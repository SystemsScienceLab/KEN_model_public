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

from .model_classes import ModelConfig, ModelVariables, ModelResults, ModelParameters
from .calibration import ParametersCalibrated
from .ar_model_estimation import estimate_ar_model, estimate_ar_model_log, estimate_ar_model_growthrates, forecast, determine_optimal_lag_order, determine_optimal_lag_order_log, determine_optimal_lag_order_growthrates, identify_structural_break
import numpy as np
import pandas as pd
import warnings

###################################################################################################################################################################################################################################################################################################


def exports_module(v: ModelVariables, p: ModelParameters, pc: ParametersCalibrated, t: int, verbose=False):
    ###################################################################################################################################################################################################################################################################################################
    #####################################################################################################################################################################################################################
    # Oil exports
    #####################################################################################################################################################################################################################
    if p.Oil_export_timeseries_model:
        Oil_exp_model_fit = estimate_ar_model_growthrates(
            pc.oil_activities, pc.Oil_optimal_lag_aic, pc.begin_time_series_estimation_oil)
        # Pick the latest value.
        v.Oil_growth_rate_forecast[t] = forecast(
            Oil_exp_model_fit, steps=1).iloc[-1]
        if p.Oil_exports_random_fluctuations:
            # Add random noise with historical volatility of oil activities growth
            if isinstance(pc.oil_activities, pd.DataFrame):
                growth_df = pc.oil_activities.select_dtypes(
                    include=[np.number]).pct_change()
                hist_growth = growth_df.to_numpy().ravel()
            else:
                growth_s = pd.to_numeric(
                    pc.oil_activities.pct_change(), errors='coerce')
                hist_growth = growth_s.to_numpy()
            hist_growth = hist_growth[np.isfinite(hist_growth)]
            if t == 1:
                if verbose:
                    print(
                        "hist_growth historical growth rates OIL ACTIVITY=OIL EXPORTS vector", hist_growth)
            n_obs = int(hist_growth.size)
            # The estimated standard error is large; cap the fluctuations at a maximum
            sigma = float(np.nanstd(hist_growth)) if n_obs > 0 else 0.0
            # A parameter regulates how much the fluctuations of oil prices are allowed to vary, to reflect different uncertainties of future oil markets
            sigma = sigma * p.oil_exports_uncertainty_change
            max_shock = 0.5 * p.oil_exports_uncertainty_change
            min_shock = - 0.5 * p.oil_exports_uncertainty_change
            shock = np.maximum(np.minimum(float(np.random.normal(
                loc=0.0, scale=sigma)), max_shock), min_shock) if sigma > 0 else 0.0
            if verbose:
                print("Shock added to oil growth rate forecast (last 5 periods)", shock)
            gEXP_oil_shocked = v.Oil_growth_rate_forecast[t] + shock
            # To avoid downward bias, take the average of last 5 periods oil exports to correct up and down
            if t > 1:
                start = max(1, t - 5)
                base_EX_oil = np.maximum(
                    (np.mean([v.EX_oil_[τ] for τ in range(start, t)], axis=0)), 0)
            else:
                base_EX_oil = v.EX_oil_[t-1]
        # Assumption: Oil exports cannot be smaller than zero, and are bounded above by last period's sectoral domestic GDP.
        # This acts as a supply constraint: export growth is limited by past output growth.
            v.EX_oil_[t] = np.minimum(np.maximum(base_EX_oil * (1 + gEXP_oil_shocked + p.oil_growth_adjustment), 0), v.X_[
                                      t-1]*(1 + v.Oil_growth_rate_forecast[t] + p.oil_growth_adjustment))
        else:
            v.EX_oil_[t] = v.EX_oil_[
                t-1] * (1 + v.Oil_growth_rate_forecast[t] + p.oil_growth_adjustment)
        pc.oil_activities = pd.concat(
            [pc.oil_activities, pd.Series([v.EX_oil_[t]])], ignore_index=True)

    else:
        if p.Oil_exports_random_fluctuations:
            # Add random noise with historical volatility of oil activities growth
            if isinstance(pc.oil_activities, pd.DataFrame):
                growth_df = pc.oil_activities.select_dtypes(
                    include=[np.number]).pct_change()
                hist_growth = growth_df.to_numpy().ravel()
            else:
                growth_s = pd.to_numeric(
                    pc.oil_activities.pct_change(), errors='coerce')
                hist_growth = growth_s.to_numpy()
            hist_growth = hist_growth[np.isfinite(hist_growth)]
            if t == 1:
                if verbose:
                    print("hist_growth historical growth rates vector", hist_growth)
            n_obs = int(hist_growth.size)
            # The estimated standard error is large; cap the fluctuations at a maximum
            sigma = float(np.nanstd(hist_growth)) if n_obs > 0 else 0.0
            max_shock = 0.5
            min_shock = - 0.5
            shock = np.maximum(np.minimum(float(np.random.normal(
                loc=0.0, scale=sigma)), max_shock), min_shock) if sigma > 0 else 0.0
            if verbose:
                print("Shock added to oil growth rate forecast (last 5 periods)", shock)
            v.oil_exports_shock[t] = shock
            gEXP_oil_shocked = pc.gEXP_oil_avg + shock
        # To avoid downward bias, take the average of last 5 periods oil exports to correct up and down
            if t > 1:
                start = max(1, t - 5)
                base_EX_oil = np.maximum(
                    (np.mean([v.EX_oil_[τ] for τ in range(start, t)], axis=0)), 0)
            else:
                base_EX_oil = v.EX_oil_[t-1]
            # Assumption: Oil exports cannot be smaller than zero, but also CANNOT be larger than sectoral domestic GDP of last period
            v.EX_oil_[t] = np.minimum(np.maximum(base_EX_oil * (1 + gEXP_oil_shocked + p.oil_growth_adjustment), 0), v.X_[
                                      t-1]*(1 + pc.gEXP_oil_avg + p.oil_growth_adjustment))
            # Append to last data series
            pc.oil_activities = pd.concat(
                [pc.oil_activities, pd.Series([v.EX_oil_[t]])], ignore_index=True)
        # Case: no variability in oil exports
        else:
            v.EX_oil_[t] = v.EX_oil_[t-1] * \
                (1 + pc.gEXP_oil_avg + p.oil_growth_adjustment)

    # Set a condition that oil exports are not negative
    v.EX_oil_[t] = np.maximum(0, v.EX_oil_[t])
    v.ex_oil_[t] = v.EX_oil_[t] / v.p_[t]  # Oil export share of total output

    #####################################################################################################################################################################################################################
    # Non-oil exports
    #####################################################################################################################################################################################################################
    # USUALLY, non-oil exports are set as fixed (baseline), or grow with a steady rate determined as scenario choice from Vision 2030 or more ambitious goals.
    # Therefore, usually, this time series is NOT activated.
    if p.Non_oil_export_timeseries_model:
        Non_oil_exp_model_fit = estimate_ar_model_growthrates(
            pc.non_oil_export_timeseries, pc.non_oil_optimal_lag_aic, pc.begin_time_series_estimation_nonoil_exp)
        # Pick the latest value.
        v.Exports_nonoil_growth_rate_forecast[t] = forecast(
            Non_oil_exp_model_fit, steps=1).iloc[-1]
        v.EX_non_oil_[t] = v.EX_non_oil_[
            t-1] * (1 + float(v.Exports_nonoil_growth_rate_forecast[t]) + p.non_oil_export_growth_adjustment)
        pc.non_oil_export_timeseries = pd.concat(
            [pc.non_oil_export_timeseries, pd.Series([v.EX_non_oil_[t]])], ignore_index=True)
    else:  # Standard case: no time series model applied to non-oil exports
        if p.Multiple_V2030_nonoil_Exports_growth:
            v.EX_non_oil_[t] = v.EX_non_oil_[t-1] * (1 + pc.gEXP_non_oil_avg + p.non_oil_export_growth_adjustment +
                                                     pc.sectoral_yearly_growth_rates_non_oil_exports_ * p.V2030_non_oil_export_scaling_factor)
        else:
            v.EX_non_oil_[t] = v.EX_non_oil_[
                t-1] * (1 + pc.gEXP_non_oil_avg + p.non_oil_export_growth_adjustment)
    # Non-oil export share of total output
    v.ex_non_oil_[t] = v.EX_non_oil_[t] / v.p_[t]

    #####################################################################################################################################################################################################################
    # Total exports
    #####################################################################################################################################################################################################################
    v.EX_[t] = v.EX_non_oil_[t] + v.EX_oil_[t]
    v.ex_[t] = v.EX_[t] / v.p_[t]  # Export share of total output

    ###################################################################################################################################################################################################################################################################################################
    return v.EX_[t], v.ex_[t], v.EX_oil_[t], v.ex_oil_[t], v.EX_non_oil_[t], v.ex_non_oil_[t], v.oil_exports_shock[t]
    ###################################################################################################################################################################################################################################################################################################
