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
FDI Module: Foreign Direct Investment decomposition and accounting
=================================================================

Decomposes total investment I_[t] into:
  - I_domestic_[t]: domestically financed investment
  - FDI_net_flow_[t]: net foreign direct investment inflows

FDI growth modes (controlled by p.FDI_follow_domestic_growth):

  1. FDI_follow_domestic_growth = True (default for active scenarios):
     FDI tracks the actual realized *sectoral* investment growth rate,
     gI_realized_[s] = I_[t,s] / I_[t-1,s] - 1, so that each sector's
     FDI grows in line with that sector's investment dynamics.  The
     FDI_target growth rate is layered on top during the vision_2030_timing
     window, causing the FDI share to *increase* in scenarios with
     FDI_target > 0.  A per-sector cap (FDI_MAX_SECTORAL_SHARE, default 50%)
     prevents any single sector from being overly FDI-dependent.

  2. FDI_follow_domestic_growth = False (legacy mode):
     FDI grows from its 2021 base-year value at the fixed annual rate
     calibrated from FDI_target.  Growth is applied for the duration of
     vision_2030_timing, after which FDI stays constant.

The module also tracks:
  - FDI_gross_flow_[t]: gross FDI inflows (scaled proportionally to net)
  - FDI_stock_[t]: cumulative FDI position (stock(t-1) + gross(t))
  - Aggregate totals and the FDI share of investment

The sectoral distribution of FDI follows positive sectoral investment I_[t],
ensuring aggregate FDI targets are always met.

Capital account and current account entries are also computed here for
balance-of-payments consistency.
"""

import numpy as np
from .model_classes import ModelVariables, ModelParameters
from .calibration import ParametersCalibrated


def fdi_module(v: ModelVariables, p: ModelParameters, pc: ParametersCalibrated, t: int):
    """
    Compute FDI flows and decompose investment into domestic + FDI components.

    Must be called AFTER total investment I_[t] and I_total[t] have been computed
    in investment_module, but BEFORE the private/public investment split, so that
    FDI can reduce the domestically-financed share.

    Parameters
    ----------
    v : ModelVariables    — model state arrays (modified in-place)
    p : ModelParameters   — scenario parameters (enable_FDI, FDI_target, etc.)
    pc : ParametersCalibrated — calibrated parameters (FDI base year, growth rates)
    t : int               — current time step (1-indexed, t=1 is 2021)
    """

    if not p.enable_FDI:
        # If FDI is disabled, all investment is domestic, FDI variables stay at zero
        v.I_domestic_[t] = v.I_[t].copy()
        return

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Determine FDI flows by sector
    # ─────────────────────────────────────────────────────────────────────────
    # Two growth modes, controlled by p.FDI_follow_domestic_growth:
    #
    # Mode A (FDI_follow_domestic_growth = True):
    #   FDI tracks the actual realized *sectoral* investment growth rate,
    #   gI_realized_[s] = I_[t,s] / I_[t-1,s] - 1, so that each sector's
    #   FDI grows in line with that sector's investment dynamics.  The
    #   FDI_target growth rate is layered on top during the vision_2030_timing
    #   window (share increases when FDI_target > 0).
    #   Per-sector cap: FDI cannot exceed FDI_MAX_SECTORAL_SHARE of I_[t,s].
    #   Aggregate cap: total FDI cannot exceed 95% of I_total[t].
    #
    # Mode B (FDI_follow_domestic_growth = False — legacy):
    #   FDI grows from its base-year value at the fixed FDI_target annual rate.
    #   After vision_2030_timing, FDI stays constant at the level reached.
    #   Distributed proportionally to positive sectoral I_[t].

    # Maximum FDI share of any individual sector's investment
    FDI_MAX_SECTORAL_SHARE = 0.50

    # FDI target factor: active during vision_2030_timing window only
    if t <= p.vision_2030_timing:
        fdi_target_factor = 1 + pc.FDI_yearly_target_growth_rate
    else:
        fdi_target_factor = 1.0

    if p.FDI_follow_domestic_growth:
        # ── Mode A: sectoral realized I_ growth + FDI target premium ──
        if t == 1:
            # First period: distribute base-year aggregate to sectors
            fdi_net_aggregate = pc.FDI_base_year_net_total
            I_positive = np.maximum(v.I_[t], 0.0)
            I_positive_total = np.sum(I_positive)
            if I_positive_total > 0:
                v.FDI_net_flow_[t] = fdi_net_aggregate * (I_positive / I_positive_total)
            else:
                v.FDI_net_flow_[t] = np.zeros(pc.S)
        else:
            # Subsequent periods: grow each sector's FDI by its realized I_ growth
            I_prev = v.I_[t-1]
            gI_realized_ = np.where(I_prev > 0, v.I_[t] / I_prev - 1.0, 0.0)
            v.FDI_net_flow_[t] = v.FDI_net_flow_[t-1] * (1 + gI_realized_) * fdi_target_factor
            # Floor: no negative FDI
            v.FDI_net_flow_[t] = np.maximum(v.FDI_net_flow_[t], 0.0)

        # Per-sector cap: FDI ≤ FDI_MAX_SECTORAL_SHARE × positive I_[t,s]
        max_fdi_per_sector = FDI_MAX_SECTORAL_SHARE * np.maximum(v.I_[t], 0.0)
        v.FDI_net_flow_[t] = np.minimum(v.FDI_net_flow_[t], max_fdi_per_sector)

        # Aggregate cap: total FDI ≤ 95% of I_total
        fdi_total = np.sum(v.FDI_net_flow_[t])
        if v.I_total[t] > 0 and fdi_total > v.I_total[t] * 0.95:
            v.FDI_net_flow_[t] *= (v.I_total[t] * 0.95 / fdi_total)

    else:
        # ── Mode B: fixed-rate growth from base year (legacy) ──
        if t <= p.vision_2030_timing:
            fdi_net_aggregate = pc.FDI_base_year_net_total * (1 + pc.FDI_yearly_target_growth_rate) ** (t - 1)
        else:
            fdi_net_aggregate = pc.FDI_base_year_net_total * (1 + pc.FDI_yearly_target_growth_rate) ** (p.vision_2030_timing - 1)

        # Cap aggregate FDI: cannot exceed 95% of total investment
        if v.I_total[t] > 0:
            fdi_net_aggregate = min(fdi_net_aggregate, v.I_total[t] * 0.95)

        # Distribute to sectors proportionally to positive I_[t]
        I_positive = np.maximum(v.I_[t], 0.0)
        I_positive_total = np.sum(I_positive)
        if I_positive_total > 0:
            v.FDI_net_flow_[t] = fdi_net_aggregate * (I_positive / I_positive_total)
        else:
            v.FDI_net_flow_[t] = np.zeros(pc.S)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Decompose investment: domestic = total - FDI
    # ─────────────────────────────────────────────────────────────────────────
    v.I_domestic_[t] = v.I_[t] - v.FDI_net_flow_[t]

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Gross FDI flows (scale proportionally from base-year gross/net ratio)
    # ─────────────────────────────────────────────────────────────────────────
    # Gross FDI = Net FDI * (base_year_gross / base_year_net) for each sector
    # Use absolute values to handle sign issues; if no base-year net, use net = gross
    fdi_net_base_total = np.abs(pc.FDI_base_year_net_total)
    fdi_gross_base_total = np.sum(np.abs(pc.FDI_base_year_gross_))
    if fdi_net_base_total > 0:
        gross_to_net_ratio = fdi_gross_base_total / fdi_net_base_total
    else:
        gross_to_net_ratio = 1.0

    v.FDI_gross_flow_[t] = np.abs(v.FDI_net_flow_[t]) * gross_to_net_ratio

    # ─────────────────────────────────────────────────────────────────────────
    # 5. FDI stock accumulation: Stock(t) = Stock(t-1) + Gross_inflow(t)
    # ─────────────────────────────────────────────────────────────────────────
    if t == 1:
        # Initialize stock from calibrated base-year values
        v.FDI_stock_[t] = pc.FDI_base_year_stock_ + v.FDI_gross_flow_[t]
    else:
        v.FDI_stock_[t] = v.FDI_stock_[t - 1] + v.FDI_gross_flow_[t]

    # ─────────────────────────────────────────────────────────────────────────
    # 5b. Net FDI stock accumulation: NetStock(t) = NetStock(t-1) + max(net_flow, 0)
    # Tracks only net inflows — avoids gross/net ratio amplification.
    # Used in model.py for FDI income repatriation calculations.
    # ─────────────────────────────────────────────────────────────────────────
    if t == 1:
        # Initialize net stock: base-year net aggregate distributed by investment shares
        fdi_net_base = pc.FDI_base_year_net_total
        I_pos = np.maximum(v.I_[t], 0.0)
        I_pos_total = np.sum(I_pos)
        base_distribution = fdi_net_base * (I_pos / I_pos_total) if I_pos_total > 0 else np.zeros(pc.S)
        v.FDI_net_stock_[t] = np.maximum(base_distribution, 0.0) + np.maximum(v.FDI_net_flow_[t], 0.0)
    else:
        v.FDI_net_stock_[t] = v.FDI_net_stock_[t - 1] + np.maximum(v.FDI_net_flow_[t], 0.0)

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Aggregate (macro) FDI variables
    # ─────────────────────────────────────────────────────────────────────────
    v.FDI_net_total[t] = np.sum(v.FDI_net_flow_[t])
    v.FDI_gross_total[t] = np.sum(v.FDI_gross_flow_[t])
    v.FDI_stock_total[t] = np.sum(v.FDI_stock_[t])
    v.FDI_net_stock_total[t] = np.sum(v.FDI_net_stock_[t])

    # FDI share of total investment
    if v.I_total[t] > 0:
        v.FDI_share_of_investment[t] = v.FDI_net_total[t] / v.I_total[t]
    else:
        v.FDI_share_of_investment[t] = 0.0


