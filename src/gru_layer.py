"""
gru_layer.py
------------
GRULayer: stacks multiple GRUCell instances with dropout between layers.

Forward:  X (T, input_dim) -> H (T, hidden_dim)  [all hidden states]
Backward: dH (T, hidden_dim) -> dX (T, input_dim)

Dropout is applied to the OUTPUT of each layer except the last,
and only when training=True (Srivastava et al., 2014).
"""

import numpy as np
from .gru_cell import GRUCell


class GRULayer:
    """
    A stack of `num_layers` GRU cells, each sharing the same hidden_dim.

    Parameters
    ----------
    input_dim  : int   — dimensionality of inputs (embed_dim for layer 0)
    hidden_dim : int   — H in the design doc
    num_layers : int   — number of stacked GRU cells (default 2)
    keep_prob  : float — dropout keep probability (1 - dropout_rate)
    seed       : int
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        keep_prob: float = 0.8,
        seed: int = 0,
    ):
        self.num_layers = num_layers
        self.H = hidden_dim
        self.keep_prob = keep_prob
        self.rng = np.random.default_rng(seed)

        # Build cells: layer 0 takes embed_dim, subsequent layers take H
        self.cells: list[GRUCell] = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(GRUCell(in_dim, hidden_dim, seed=seed + i))

        # Dropout masks stored per layer (for backward pass)
        self._masks: list[np.ndarray | None] = [None] * num_layers

        # Hidden state across the sequence — initialised to zero each forward
        self._h_seq: list[list[np.ndarray]] = []  # [layer][timestep]

    # ──────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────

    def forward(
        self, X: np.ndarray, h0: np.ndarray | None = None, training: bool = True
    ) -> np.ndarray:
        """
        Parameters
        ----------
        X        : (T, input_dim) — sequence of embeddings
        h0       : (num_layers, H) or None — initial hidden states
        training : bool — whether to apply dropout

        Returns
        -------
        H_out : (T, H) — hidden states of the LAST layer across time
        """
        T = X.shape[0]

        if h0 is None:
            h0 = np.zeros((self.num_layers, self.H))

        self._h_seq = []  # reset cache

        layer_input = X  # (T, input_dim)

        for layer_idx, cell in enumerate(self.cells):
            h_prev = h0[layer_idx]          # (H,)
            h_states = []                   # will hold T hidden states

            for t in range(T):
                h_t = cell.forward(layer_input[t], h_prev)
                h_states.append(h_t)
                h_prev = h_t

            H_layer = np.stack(h_states)    # (T, H)
            self._h_seq.append(H_layer)

            # Apply dropout between layers (not after the last layer)
            if layer_idx < self.num_layers - 1 and training:
                mask = (self.rng.random(H_layer.shape) < self.keep_prob
                        ).astype(np.float64) / self.keep_prob
                self._masks[layer_idx] = mask
                layer_input = H_layer * mask
            else:
                self._masks[layer_idx] = None
                layer_input = H_layer

        return layer_input  # (T, H) — output of last layer

    # ──────────────────────────────────────────
    # Backward
    # ──────────────────────────────────────────

    def backward(self, dH_out: np.ndarray) -> np.ndarray:
        """
        Backpropagate through all layers and all timesteps.

        Parameters
        ----------
        dH_out : (T, H) — upstream gradient w.r.t. last layer's H

        Returns
        -------
        dX : (T, input_dim) — gradient w.r.t. the original input X
        """
        T = dH_out.shape[0]
        dLayer = dH_out  # gradient flowing into the current layer from above

        for layer_idx in reversed(range(self.num_layers)):
            cell = self.cells[layer_idx]
            mask = self._masks[layer_idx]

            # Apply dropout mask in backward (if one was applied in forward)
            if mask is not None:
                dLayer = dLayer * mask

            dh_next = np.zeros(self.H)
            dInput = np.zeros_like(self._h_seq[layer_idx - 1]
                                   if layer_idx > 0
                                   else np.zeros((T, cell.input_dim)))

            # BPTT: backprop through time within this layer
            for t in reversed(range(T)):
                dh_t = dLayer[t] + dh_next          # combine upstream + recurrent
                dx_t, dh_next = cell.backward(dh_t)
                # dInput stores gradient w.r.t. this layer's input at each step
                if layer_idx == 0:
                    dInput[t] = dx_t
                else:
                    dInput[t] = dx_t

            dLayer = dInput

        return dLayer   # (T, input_dim)

    # ──────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────

    def params(self) -> dict[str, np.ndarray]:
        out = {}
        for i, cell in enumerate(self.cells):
            for k, v in cell.params().items():
                out[f"layer{i}_{k}"] = v
        return out

    def grads(self) -> dict[str, np.ndarray]:
        out = {}
        for i, cell in enumerate(self.cells):
            for k, v in cell.grads().items():
                out[f"layer{i}_{k}"] = v
        return out

    def zero_grad(self) -> None:
        for cell in self.cells:
            cell.zero_grad()

    def state_dict(self) -> dict[str, np.ndarray]:
        out = {}
        for i, cell in enumerate(self.cells):
            for k, v in cell.state_dict().items():
                out[f"layer{i}_{k}"] = v
        return out

    def load_state_dict(self, d: dict[str, np.ndarray]) -> None:
        for i, cell in enumerate(self.cells):
            cell_dict = {
                k.replace(f"layer{i}_", ""): v
                for k, v in d.items()
                if k.startswith(f"layer{i}_")
            }
            cell.load_state_dict(cell_dict)

    @property
    def last_hidden(self) -> np.ndarray:
        """Return the final hidden state of each layer: (num_layers, H)."""
        return np.stack([h[-1] for h in self._h_seq])
