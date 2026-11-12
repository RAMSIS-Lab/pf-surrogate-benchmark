"""Shared metric helpers: angle referencing to the slack bus, per-sample errors."""
import numpy as np


def sample_errors(vm_pred, va_pred_deg, vm_true, va_true_deg, slack, bus_type):
    """All arrays [S, nb]; slack is [S] bus index; bus_type [S, nb] in {PQ:1, PV:2, REF:3}.
    Angles are referenced to the slack bus (prediction and label separately)."""
    idx = np.arange(len(slack))
    va_p = va_pred_deg - va_pred_deg[idx, slack][:, None]
    va_t = va_true_deg - va_true_deg[idx, slack][:, None]
    dvm = np.abs(vm_pred - vm_true)
    dva = np.abs(va_p - va_t)
    out = {
        "vm_mae": dvm.mean(1), "va_mae_deg": dva.mean(1),
        "vm_max": dvm.max(1), "va_max_deg": dva.max(1),
    }
    for name, code in (("pq", 1), ("pv", 2)):
        m = bus_type == code
        cnt = np.maximum(m.sum(1), 1)
        out[f"vm_mae_{name}"] = (dvm * m).sum(1) / cnt
        out[f"va_mae_deg_{name}"] = (dva * m).sum(1) / cnt
    return out
