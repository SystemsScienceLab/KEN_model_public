"""Fit the scaling factor for the oil and gas intensities of the non-power sectors.

The factor (ParametersCalibrated.NATIONAL_CO2_SCALING_FACTOR, calibration.py) is chosen so that national
energy CO2 in the Baseline scenario matches the Energy Institute series 2021-2025 (least squares). The
power sector follows the historical generation data in these years, so only the non-power level is fitted.
Because the factor also enters the calibration of energy-efficiency costs, the fit is repeated a few
times until it is stable.

Run from the repository root:  python -m model.calibrate_national_co2_factor
Then copy the printed factor into calibration.py.
"""
import warnings
import numpy as np

from model.parameters_baseline import ModelParameters
from model.calibration import ParametersCalibrated
from model.model import run_model, ModelConfig

SCENARIO = {'Baseline_scenario': True, 'Vision_2030_scenario': False,
            'Transformation_scenario': False, 'Zero_growth_rates_steady_state': False}
YEARS = [2021, 2022, 2023, 2024, 2025]   # t = 1..5


def run_baseline(factor):
    ParametersCalibrated.NATIONAL_CO2_SCALING_FACTOR = factor
    p = ModelParameters.default_values()
    p.Warnings_ON = False
    pc = ParametersCalibrated(p, scenario_config=SCENARIO, verbose=False)
    pc.force_hist_national_co2 = False   # let the model compute 2021-2025, to fit the factor
    res = run_model(ModelConfig(p, pc, 41))
    e = pc.electricity_sector_idx
    t = list(range(1, len(YEARS) + 1))
    total = res.macro['emissions_gross_total'].loc[t].to_numpy()
    power = (res.sectoral['oil_emissions'].iloc[:, e] + res.sectoral['gas_emissions'].iloc[:, e]).loc[t].to_numpy()
    observed = np.array([pc.national_energy_co2_hist_mt[y] for y in YEARS])
    return total, power, observed


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    np.seterr(all='ignore')
    factor = ParametersCalibrated.NATIONAL_CO2_SCALING_FACTOR
    for _ in range(4):
        total, power, observed = run_baseline(factor)
        non_power = total - power
        target_non_power = observed - power
        # least-squares multiplier on the current non-power level
        factor *= float(np.sum(target_non_power * non_power) / np.sum(non_power ** 2))
    total, power, observed = run_baseline(factor)
    print(f'Fitted NATIONAL_CO2_SCALING_FACTOR = {factor:.3f}')
    for y, m, o in zip(YEARS, total, observed):
        print(f'  {y}: model {m:6.1f} Mt, Energy Institute {o:6.1f} Mt, {100 * (m / o - 1):+.1f}%')
