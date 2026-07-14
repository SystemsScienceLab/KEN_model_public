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

from model.model import run_model, ModelConfig
from model.calibration import ParametersCalibrated
import numpy as np
import pandas as pd
import datetime
import dataclasses
from types import SimpleNamespace

from model.parameters_baseline import ModelParameters as BaseModelParameters
from model.parameters_Vision2030_scenario import ModelParameters as Vision2030ModelParameters
from model.parameters_transformation_scenario import ModelParameters as TransformationModelParameters
from model.parameters_zero_growth_steadystate_calibration import ModelParameters as ZeroGrowthSteadyStateModelParameters
from typing import Literal
import warnings


ScenarioType = Literal["Baseline", "Vision_2030",
                       "Transformation", "Steady_state"]


class ExperimentRunner:

    scenario_params = {
        "Baseline": BaseModelParameters,
        "Vision_2030": Vision2030ModelParameters,
        "Transformation": TransformationModelParameters,
        "Steady_state": ZeroGrowthSteadyStateModelParameters
    }

    def __init__(self, startyear: int = 1, endyear: int = 41):
        self.startyear = startyear
        self.endyear = endyear
        # Suppress numpy "divide by zero" / "invalid value encountered in divide" warnings
        # (division by zero occurs in the model by design; the warning is not actionable).
        np.seterr(divide='ignore', invalid='ignore')

    def single_run(self, scenario: str, seed: int, parameter_overrides: dict = None):

        np.random.seed(seed)

        # Load and calibrate parameters
        parameters = self.scenario_params[scenario].default_values()
        # Apply parameter overrides if provided
        if parameter_overrides:
            for param_name, param_value in parameter_overrides.items():
                if hasattr(parameters, param_name):
                    setattr(parameters, param_name, param_value)
        
        # Create scenario configuration dict to pass to calibration
        scenario_config = {
            'Baseline_scenario':                scenario == 'Baseline',
            'Vision_2030_scenario':             scenario == 'Vision_2030',
            'Transformation_scenario':          scenario == 'Transformation',
            'Zero_growth_rates_steady_state':   scenario == 'Steady_state'
        }
        
        parameters_calibrated = pc = ParametersCalibrated(
            parameters, verbose=False, scenario_config=scenario_config)
        config = ModelConfig(parameters, parameters_calibrated, self.endyear)

        # Set to ignore all WARNINGS
        parameters.Warnings_ON = False

        # Run simulation
        results = run_model(config, verbose=False)

        run_report = {
            "results_macro": results.macro,
            "results_sectoral": results.sectoral,
            "startyear": self.startyear,
            "endyear": self.endyear,
            "parameters": SimpleNamespace(**dataclasses.asdict(parameters)),
            "parameters_calibrated": parameters_calibrated,
            "scenario": scenario,
            "seed": seed,
            "timestamp": datetime.datetime.now()
        }

        return run_report

    def run(self, scenarios: list[ScenarioType], iterations: int = 10, base_seed=None, verbose=True, parameter_overrides: dict = None):
        all_run_reports = {}

        seed_index = 0
        if base_seed is None:
            base_seed = np.random.randint(0, 1000000)

        n_runs = len(scenarios) * iterations
        n_warnings = 0

        for scenario in scenarios:
            scenario_reports = []
            for _ in range(iterations):
                seed_index += 1
                seed = base_seed + seed_index

                if verbose:
                    print(
                        f"\r>> Running simulation {seed_index}/{n_runs}"
                        f" ({scenario}, seed={seed})                   ",
                        end='', flush=True
                    )

                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always")
                    run_report = self.single_run(scenario, seed, parameter_overrides)

                run_report["warnings"] = caught_warnings.copy()
                n_warnings += len(caught_warnings)

                scenario_reports.append(run_report)
            all_run_reports[scenario] = scenario_reports

        if verbose:
            print(
                f"\n>> All simulations completed ({n_warnings} warnings)")

        return all_run_reports

    def describe_results(self, run_report):
        results = run_report["results"]
        parameters = run_report["parameters"]
        parameters_calibrated = run_report["parameters_calibrated"]
        startyear = run_report["startyear"]
        endyear = run_report["endyear"]

        print("VALIDATION BY GROWTH RATES comparison model to data")
        print(
            f"This is the average empirical GDP growth rate from 2013-2023 in %: {parameters_calibrated.gGY_avg*100:.2f}%")
        df = results.macro.copy()
        df['gY'] = df['Y'].pct_change()*100
        Y_startyear = results.macro.loc[1, 'Y']
        Y_endyear = results.macro.loc[:, 'Y'].iloc[-1]
        years = endyear - startyear
        Y_avg_growth_rate_model = (Y_endyear / Y_startyear) ** (1 / years) - 1
        print(
            f"This is average MODEL GDP growth by taking first and last value: {Y_avg_growth_rate_model * 100:.2f}%")
        print(
            f"This is the average MODEL GDP growth rates over the whole time horizon: {df['gY'].mean():.2f}%")

        print("")
        print("Exogenous growth rates of components of GDP 2013 - 2023, all in percent")
        print(
            f"Investment private, gI_avg, calibrated trend growth rate: {parameters_calibrated.gI_avg*100:.2f}%")
        print(
            f"Average investment growth in desalination capacity until 2030: {parameters_calibrated.gDesalCap_avg*100:.2f}%")
        print(
            f"Exports oil, gEXP_oil_avg + oil growth adjustment parameter: {(parameters_calibrated.gEXP_oil_avg + parameters.oil_growth_adjustment)*100:.2f}%")
        print(
            f"Exports non-oil, gEXP_non_oil_avg incl. exogenous adjustment factor: {(parameters_calibrated.gEXP_non_oil_avg + parameters.non_oil_export_growth_adjustment)*100:.2f}%")
        print(
            f"Wages, gW + wage population growth adjustment parameter: {((df['gW'].mean() + parameters.wage_population_growth_adjustment)*100):.2f}%")
        print(
            f"Imports, MEAN average growth rate: {((df['gIM']).mean()*100):.2f}%")
        print(
            f"Growth rate Government consumption as in IOTs, gGY_avg: {((parameters_calibrated.gGY_avg*100)):.2f}%")

        print("")
        print("Average growth rates from the time series modeling")
        print("MEAN Y GDP Output growth_rate_forecast", (df['Y_growth_rate_forecast'].mean(
        )*100).round(1))
        print("MEAN Oil_growth_rate_forecast incl. oil growth adjustment parameter",
              (df['Oil_growth_rate_forecast'].mean()*100 + parameters.oil_growth_adjustment).round(1), '%')
        print("MEAN NON Oil_growth_rate_forecast incl. NON oil growth adjustment parameter",
              (df['Exports_nonoil_growth_rate_forecast'].mean()*100 + parameters.oil_growth_adjustment*100).round(1), '%')
        print("MEAN GY_growth_rate_forecast",
              (df['GY_growth_rate_forecast'].mean()*100).round(1), '%')

        print("Endogenous investment growth rates PERCENT SECTOR MEAN for each sector, over all times",
              results.sectoral['gI_endog'].mean()*100)
        print("Endogenous investment growth rates PERCENT TIME MEAN OVER ALL SECTORS at time t",
              results.sectoral['gI_endog'].mean(axis=1)*100)
