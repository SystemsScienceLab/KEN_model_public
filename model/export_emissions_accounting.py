"""
Emissions embodied in exported crude oil: accounting extension outside the model.

Reproduces Table "export_emissions" of the main text and the sensitivity table of the
Supplementary Information from a saved results file.

Method
------
Nominal crude oil exports (sector "Extraction of crude petroleum and natural gas") are
converted to barrels with the 2021 export unit value and the crude sector's model price:

    barrels[t] = EX_crude[t] / (P_2021 * p_crude[t])

P_2021 is the 2021 input-output crude export value divided by the observed 2021 crude export
volume (6.2 mb/d). It is the same for all scenarios, so the export unit value is held
constant in real terms at its 2021 level. Barrels are multiplied by a combustion emission
factor of 0.43 tCO2 per barrel.

Usage (from the repository root, KEN_model conda environment):
    python -m model.export_emissions_accounting [saved_results/results_YYYYMMDD_HHMMSS.pkl]
"""

import glob
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CRUDE_EXPORTS_2021_MBD = 6.2    # Saudi crude oil exports 2021, million barrels per day
EMISSION_FACTOR = 0.43          # tCO2 per barrel of crude oil combusted
SAR_PER_USD = 3.75              # Exchange rate peg
PEAK_CRUDE_EXPORTS_MBD = 7.6    # Historical peak of annual crude exports (2016)
N_YEARS = 40                    # 2021-2060 (t = 1..40)

SCENARIOS = ["Baseline", "Vision_2030", "Transformation"]


def load_results(path=None):
    if path is None:
        path = sorted(glob.glob("saved_results/results_*.pkl"))[-1]
    with open(path, "rb") as f:
        return pickle.load(f), path


def main(path=None):
    results, path = load_results(path)
    print(f"Results file: {path}")

    pc = results["Baseline"][0]["parameters_calibrated"]
    crude = pc.crude_oil_sector
    barrels_2021 = CRUDE_EXPORTS_2021_MBD * 1e6 * 365
    gt_2021 = barrels_2021 * EMISSION_FACTOR / 1e9

    # 2021 IOT value of crude exports (1000 SAR, prices = 1): common anchor for all scenarios
    EX_crude_2021 = pc.EX_oil_[crude]
    unit_value_sar = EX_crude_2021 * 1e3 / barrels_2021
    print(f"Crude export value 2021 (IOT): {EX_crude_2021 / 1e6:.1f} bn SAR")
    print(f"Implied export unit value 2021: {unit_value_sar:.0f} SAR/bbl "
          f"= {unit_value_sar / SAR_PER_USD:.1f} USD/bbl")

    t = np.arange(1, N_YEARS + 1)
    EX_baseline = results["Baseline"][0]["results_sectoral"]["EX_oil"].iloc[:N_YEARS, crude].values
    variants, territorial, paths = {}, {}, {}
    for sc in SCENARIOS:
        macro = results[sc][0]["results_macro"]
        sectoral = results[sc][0]["results_sectoral"]
        EX = sectoral["EX_oil"].iloc[:N_YEARS, crude].values     # nominal crude exports
        p_crude = sectoral["p"].iloc[:N_YEARS, crude].values
        deflator = macro["deflator_gdp"].values[:N_YEARS]
        cpi = macro["price_index_cons"].values[:N_YEARS]
        ex_macro = macro["ex_oil"].values[:N_YEARS]              # real, crude + refined

        # Export volume relative to 2021 under alternative price assumptions
        central = EX / p_crude / EX_crude_2021
        v = {
            "Central: crude sector price": central,
            "GDP deflator": EX / deflator / EX_crude_2021,
            "Consumer price index": EX / cpi / EX_crude_2021,
            "No deflation (nominal)": EX / EX_crude_2021,
            "Oil price +2.0%/yr": EX / 1.02 ** t / EX_crude_2021,
            f"Oil price +{pc.inflation_avg * 100:.2f}%/yr (hist. non-oil deflator)":
                EX / (1 + pc.inflation_avg) ** t / EX_crude_2021,
            f"Volume capped at {PEAK_CRUDE_EXPORTS_MBD} mb/d":
                np.minimum(central, PEAK_CRUDE_EXPORTS_MBD / CRUDE_EXPORTS_2021_MBD),
            # Lower bound: one oil price path for all scenarios, rising with the baseline's
            # nominal crude exports, so that baseline export volume stays at its 2021 level
            "Oil price follows baseline nominal exports (lower bound)": EX / EX_baseline,
            "Previous convention: crude + refined, own-2021 anchor": ex_macro / ex_macro[0],
        }
        variants[sc] = {k: x * gt_2021 for k, x in v.items()}
        paths[sc] = variants[sc]["Central: crude sector price"]
        territorial[sc] = (macro["emissions_gross_total"].values[:N_YEARS].sum(),
                           macro["emissions_net_total"].values[:N_YEARS].sum())

    # Territorial emissions: detect unit (t or Mt) and convert to Gt
    scale = 1e9 if territorial["Baseline"][0] > 1e6 else 1e3
    territorial = {sc: (g / scale, n / scale) for sc, (g, n) in territorial.items()}
    terr_avoided = territorial["Baseline"][0] - territorial["Transformation"][0]

    print("\nTerritorial cumulative emissions 2021-2060, GtCO2 (gross, net)")
    for sc in SCENARIOS:
        print(f"  {sc:15s} {territorial[sc][0]:6.1f} {territorial[sc][1]:6.1f}")

    print("\nCentral estimate, annual GtCO2 (mb/d)")
    for sc in SCENARIOS:
        x = paths[sc]
        print(f"  {sc:15s} " + "  ".join(
            f"{2020 + i}: {x[i - 1]:.3f} ({x[i - 1] / gt_2021 * CRUDE_EXPORTS_2021_MBD:.1f})"
            for i in (1, 10, 20, 30, 40)) + f"  | first decade {x[:10].sum():.1f}")

    rows = []
    for k in variants["Baseline"]:
        b, v30, tr = (variants[sc][k].sum() for sc in SCENARIOS)
        rows.append({"Variant": k, "Baseline": b, "Vision 2030": v30, "Transformation": tr,
                     "Avoided V2030": b - v30, "Avoided Transf.": b - tr,
                     "x territorial": (b - tr) / terr_avoided,
                     "Baseline 2060 mb/d": variants["Baseline"][k][-1] / gt_2021 * CRUDE_EXPORTS_2021_MBD})
    print("\nCumulative emissions from combustion of exported crude 2021-2060, GtCO2")
    print(pd.DataFrame(rows).set_index("Variant").round(1).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
