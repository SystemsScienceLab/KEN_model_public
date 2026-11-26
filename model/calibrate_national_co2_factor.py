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

"""Fit the two common scaling factors of the energy calibration (calibration.py, section 5.5b).

1. ParametersCalibrated.NATIONAL_CO2_SCALING_FACTOR: scales the oil and gas intensities of the non-power
   sectors so that national energy CO2 in the Baseline scenario matches the Energy Institute series
   2021-2025 (least squares). The power sector follows the historical generation data in these years,
   so only the non-power level is fitted.
2. ParametersCalibrated.ELECTRICITY_SCALING_FACTOR: scales the electricity intensities of all sectors so
   that sectoral electricity demand in the Baseline scenario equals observed generation in 2025, the
   last historical year. From 2026 demand then continues from the observed level.

Both factors are fitted on the Baseline and are the same in all scenarios. Because they also enter the
calibration of energy-efficiency costs and investment, the fit is repeated a few times until it is stable.

Run from the repository root:  python -m model.calibrate_national_co2_factor
The printed factors are the values of the two constants in calibration.py.
"""
import warnings
import numpy as np

from model.parameters_baseline import ModelParameters
from model.calibration import ParametersCalibrated
from model.model import run_model, ModelConfig

SCENARIO = {'Baseline_scenario': True, 'Vision_2030_scenario': False,
            'Transformation_scenario': False, 'Zero_growth_rates_steady_state': False}
YEARS = [2021, 2022, 2023, 2024, 2025]   # t = 1..5


def run_baseline(co2_factor, elec_factor):
    ParametersCalibrated.NATIONAL_CO2_SCALING_FACTOR = co2_factor
    ParametersCalibrated.ELECTRICITY_SCALING_FACTOR = elec_factor
    p = ModelParameters.default_values()
    p.Warnings_ON = False
    pc = ParametersCalibrated(p, scenario_config=SCENARIO, verbose=False)
    pc.force_hist_national_co2 = False   # let the model compute 2021-2025, to fit the CO2 factor
    res = run_model(ModelConfig(p, pc, 41))
    e = pc.electricity_sector_idx
    t = list(range(1, len(YEARS) + 1))
    # Energy CO2 only (oil and gas combustion); process emissions are calibrated separately
    total = (res.macro['oil_emissions_total'] + res.macro['gas_emissions_total']).loc[t].to_numpy(dtype=float)
    power = (res.sectoral['oil_emissions'].iloc[:, e] + res.sectoral['gas_emissions'].iloc[:, e]).loc[t].to_numpy(dtype=float)
    observed = np.array([pc.national_energy_co2_hist_mt[y] for y in YEARS])
    # Sectoral electricity demand in the last historical year from real output (2021 prices), before the
    # model rescales it to the observed total
    t_last = pc.hist_power_last_t
    X = res.sectoral['X'].loc[t_last].to_numpy(dtype=float)
    price = res.sectoral['p'].loc[t_last].to_numpy(dtype=float)
    price_2021 = res.sectoral['p'].loc[1].to_numpy(dtype=float)
    x_real = np.divide(X * price_2021, price, out=np.zeros_like(X), where=price != 0)
    reduction = res.sectoral['energy_reduction_factor'].loc[t_last].to_numpy(dtype=float)
    elec_demand = float(np.sum(x_real * pc.electricity_intensities_ * (1.0 - reduction)))
    elec_observed = float(pc.hist_power_total_gen_gj[t_last - 1])
    return total, power, observed, elec_demand, elec_observed


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    np.seterr(all='ignore')
    co2_factor = ParametersCalibrated.NATIONAL_CO2_SCALING_FACTOR
    elec_factor = ParametersCalibrated.ELECTRICITY_SCALING_FACTOR
    for _ in range(4):
        total, power, observed, elec_demand, elec_observed = run_baseline(co2_factor, elec_factor)
        non_power = total - power
        target_non_power = observed - power
        # least-squares multiplier on the current non-power level
        co2_factor *= float(np.sum(target_non_power * non_power) / np.sum(non_power ** 2))
        elec_factor *= elec_observed / elec_demand
    total, power, observed, elec_demand, elec_observed = run_baseline(co2_factor, elec_factor)
    print(f'Fitted NATIONAL_CO2_SCALING_FACTOR = {co2_factor:.3f}')
    for y, m, o in zip(YEARS, total, observed):
        print(f'  {y}: model {m:6.1f} Mt, Energy Institute {o:6.1f} Mt, {100 * (m / o - 1):+.1f}%')
    print(f'Fitted ELECTRICITY_SCALING_FACTOR = {elec_factor:.3f}')
    print(f'  2025: sectoral electricity demand {elec_demand / 3.6e6:6.1f} TWh, observed generation '
          f'{elec_observed / 3.6e6:6.1f} TWh, {100 * (elec_demand / elec_observed - 1):+.2f}%')
