from __future__ import annotations

from dataclasses import dataclass


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
