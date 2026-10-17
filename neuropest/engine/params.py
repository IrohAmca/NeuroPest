from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Refractory time (ms) a silenced neuron is held at: while refractory a neuron integrates nothing and cannot cross the
# threshold, on every engine. Finite, because WGSL may assume no inf / NaN; in float32 1e30 - dt == 1e30, so it never ends.
REF_CLAMP = np.float32(1e30)


@dataclass(frozen=True)
class LIFParams:
    """Constants of Shiu et al. 2024 (github.com/philshiu/Drosophila_brain_model, model.py)."""
    v_rest: float = -52.0   # mV; also the reset potential
    v_th: float = -45.0     # mV
    tau_m: float = 20.0     # ms
    tau_syn: float = 5.0    # ms
    t_ref: float = 2.2      # ms; v and g are frozen while refractory
    t_dly: float = 1.8      # ms synaptic delay (quantized to the time step)
    w_syn: float = 0.275    # mV per synapse
    tau_adapt: float = 300.0  # ms; decay of the spike-frequency adaptation (only on neurons with net.extra['adapt_mv'])
