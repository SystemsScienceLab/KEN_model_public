# =============================================================================
# Lead model conceptualization and development (theory, methodology, and
# implementation), software architecture & model coding (all components),
# data acquisition & processing, team supervision: Michael Miess
# Initial software architecture development, first coding, technical consulting:
# Joel Foramitti
# Lead of energy modeling: Ansir Ilyas
# Lead of water modeling: Dan Wang
# Model consulting & support: Asjad Naqvi
# Project conceptualization, funding & supervision: Yoshihide Wada
# =============================================================================

"""
Remittances Module: Forward projection of personal remittances paid
====================================================================

Projects personal remittances (outflows from Saudi Arabia to the rest of
the world) over the simulation horizon.  Remittances are a key component
of the secondary income balance in the Balance of Payments.

Growth modes (controlled by p.remittances_follow_wage_growth):

  1. remittances_follow_wage_growth = True (default):
     Remittances grow with the nominal wage bill growth rate:
       gW_nominal = W[t] / W[t-1] - 1
     This captures real GDP growth, inflation, and employment dynamics.
     Economically sound: remittances sent by foreign workers scale with
     their compensation.  An additional manual adjustment parameter
     p.remittances_growth_adjustment is layered on top.

  2. remittances_follow_wage_growth = False (exogenous trend):
     Remittances grow from the 2021 base year at the calibrated
     compound annual growth rate pc.gRemittances_avg, plus the manual
     adjustment parameter p.remittances_growth_adjustment.

The module also records:
  - remittances_growth[t]: the applied growth rate of remittances
"""
import numpy as np

from .model_classes import ModelVariables, ModelParameters
from .calibration import ParametersCalibrated


def remittances_module(v: ModelVariables, p: ModelParameters, pc: ParametersCalibrated, t: int):
    """
    Compute remittances for time step t.

    Must be called AFTER wages W[t] are computed (section 3.3 of model.py),
    so that wage-bill growth is available for Mode A.

    Parameters
    ----------
    v : ModelVariables    — model state arrays (modified in-place)
    p : ModelParameters   — scenario parameters (enable_remittances, etc.)
    pc : ParametersCalibrated — calibrated parameters (remittances_base_year, gRemittances_avg)
    t : int               — current time step (1-indexed, t=1 is 2021)
    """

    if not p.enable_remittances:
        # If remittances are disabled, they stay at zero
        return

    if t == 1:
        # Base year: use calibrated 2021 value
        v.remittances[t] = pc.remittances_base_year
        v.remittances_growth[t] = 0.0
        return

    # ─────────────────────────────────────────────────────────────────────
    # Determine remittances growth rate
    # ─────────────────────────────────────────────────────────────────────
    if p.remittances_follow_wage_growth:
        # Mode A: grow with nominal wage bill
        # W[t] is already computed in section 3.3 of model.py
        if v.W[t - 1] > 0:
            gW_nominal = v.W[t] / v.W[t - 1] - 1.0
        else:
            gW_nominal = 0.0
        g_remit = np.maximum(gW_nominal + p.remittances_growth_adjustment,0.0)  # Ensure remittances growth is not negative
    else:
        # Mode B: exogenous trend from calibrated average growth rate
        g_remit = np.maximum(pc.gRemittances_avg + p.remittances_growth_adjustment,0.0)  # Ensure remittances growth is not negative

    # Saudization policy: rising domestic labour share → fewer foreign workers → smaller
    # fraction of the wage bill is remitted abroad each year.
    # Compound formula: (1+g_eff) = (1+g_raw) × (1 − saud_rate)
    saud_rate = getattr(p, 'remittances_saudization_rate', 0.0)
    if saud_rate > 0.0:
        g_remit = (1.0 + g_remit) * (1.0 - saud_rate) - 1.0

    v.remittances_growth[t] = g_remit
    v.remittances[t] = v.remittances[t - 1] * (1.0 + g_remit)

    # Floor: remittances cannot be negative
    v.remittances[t] = max(v.remittances[t], 0.0)
