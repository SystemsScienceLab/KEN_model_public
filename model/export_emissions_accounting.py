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
# Real oil price trend 1990-2023 for the lower estimate: Brent 23.73 USD/bbl in 1990 (Energy Institute,
# Statistical Review 2023) and 82.49 USD/bbl in 2023 (annual average) = 3.85% a year nominal; US CPI
# 130.7 (1990) and 304.7 (2023) = 2.60% a year; real trend = 1.0385 / 1.0260 - 1 = 1.2% a year.
REAL_OIL_PRICE_TREND = 0.012
N_YEARS = 40                    # 2021-2060 (t = 1..40)

SCENARIOS = ["Baseline", "Vision_2030", "Transformation"]


def load_results(path=None):
    if path is None:
        path = sorted(glob.glob("saved_results/results_*.pkl"))[-1]
    with open(path, "rb") as f:
        return pickle.load(f), path


def compute(results):
    """Annual emissions from the combustion of exported crude under all price assumptions (GtCO2 per
    year, dictionary variants[scenario][assumption]) and territorial emissions from the model run
    (cumulative sums in territorial, annual gross and net series in annual, model units)."""
    pc = results["Baseline"][0]["parameters_calibrated"]
    crude = pc.crude_oil_sector
    barrels_2021 = CRUDE_EXPORTS_2021_MBD * 1e6 * 365
    gt_2021 = barrels_2021 * EMISSION_FACTOR / 1e9

    # 2021 IOT value of crude exports (1000 SAR, prices = 1): common anchor for all scenarios
    EX_crude_2021 = pc.EX_oil_[crude]
    unit_value_sar = EX_crude_2021 * 1e3 / barrels_2021
    t = np.arange(1, N_YEARS + 1)
    EX_baseline = results["Baseline"][0]["results_sectoral"]["EX_oil"].iloc[:N_YEARS, crude].values
    variants, territorial, paths, annual = {}, {}, {}, {}
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
            "Real oil price +1.2%/yr, 1990-2023 trend (lower bound)": EX / p_crude / (1 + REAL_OIL_PRICE_TREND) ** (t - 1) / EX_crude_2021,
            "Oil price follows baseline nominal exports": EX / EX_baseline,
            "Alternative convention: crude + refined, own-2021 anchor": ex_macro / ex_macro[0],
        }
        variants[sc] = {k: x * gt_2021 for k, x in v.items()}
        paths[sc] = variants[sc]["Central: crude sector price"]
        territorial[sc] = (macro["emissions_gross_total"].values[:N_YEARS].sum(),
                           macro["emissions_net_total"].values[:N_YEARS].sum())
        annual[sc] = (np.array([float(np.sum(x)) for x in macro["emissions_gross_total"].values[:N_YEARS]]),
                      np.array([float(np.sum(x)) for x in macro["emissions_net_total"].values[:N_YEARS]]))

    return dict(pc=pc, gt_2021=gt_2021, EX_crude_2021=EX_crude_2021, unit_value_sar=unit_value_sar,
                variants=variants, territorial=territorial, paths=paths, annual=annual)


UPPER = "Central: crude sector price"
LOWER = "Real oil price +1.2%/yr, 1990-2023 trend (lower bound)"


def plot_export_emissions(results, save_path=None):
    """Figure: annual territorial emissions (solid lines) and emissions from the combustion of exported
    crude by scenario: upper estimate (dashed, deflation with the crude sector's model price) and lower
    estimate (dotted, oil price rising with baseline export revenue), shaded in between,
    2021-2060, GtCO2 per year. One panel, scenario colours as in the other result figures. The range
    of the export estimate under other price assumptions is in the table printed by main()."""
    import matplotlib.pyplot as plt
    c = compute(results)
    years = np.arange(2021, 2021 + N_YEARS)
    colors = {"Baseline": "#d62728", "Vision_2030": "#ff7f0e", "Transformation": "#1f77b4"}
    scale = 1e9 if c["territorial"]["Baseline"][0] > 1e6 else 1e3
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ends = []
    for sc in SCENARIOS:
        gross = c["annual"][sc][0] / scale
        export = c["variants"][sc][UPPER]
        name = {"Baseline": "Baseline", "Vision_2030": "Vision 2030", "Transformation": "Transformation"}[sc]
        lo = c["variants"][sc][LOWER]
        ax.fill_between(years, np.minimum(lo, export), np.maximum(lo, export), color=colors[sc], alpha=0.15, linewidth=0)
        ax.plot(years, gross, color=colors[sc], linewidth=2.5, label=f"{name} - territorial emissions")
        ax.plot(years, export, color=colors[sc], linewidth=2.5, linestyle="--", label=f"{name} - exported crude, upper estimate")
        ax.plot(years, lo, color=colors[sc], linewidth=1.8, linestyle=":", label=f"{name} - exported crude, lower estimate")
        ends.append((gross[-1], f"{gross[-1]:.2f}", colors[sc]))
        ends.append((export[-1], f"{export[-1]:.2f}", colors[sc]))
        ends.append((lo[-1], f"{lo[-1]:.2f}", colors[sc]))
        ax.scatter([2060, 2060, 2060], [gross[-1], export[-1], lo[-1]], color=colors[sc], s=30, zorder=5)
    # end-point labels, spread so that they do not overlap
    ax.set_ylim(bottom=0)
    y0, y1 = ax.get_ylim()
    gap = (y1 - y0) * 0.045
    ends.sort(key=lambda e: e[0])
    pos = []
    for y, _, _ in ends:
        pos.append(y if not pos or y - pos[-1] >= gap else pos[-1] + gap)
    for (y, txt, col), yp in zip(ends, pos):
        ax.annotate(txt, xy=(2060, y), xytext=(2061.0, yp), color=col, fontweight="bold", fontsize=14,
                    va="center", ha="left", annotation_clip=False)
    # legend: colours = scenarios, line type = account
    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="grey", alpha=0.3, label="Range between lower and upper estimate"))
    ax.legend(handles=handles, fontsize=11, loc="upper left", ncol=2)
    ax.set_title("Territorial emissions and emissions from exported crude", fontweight="bold", fontsize=16)
    ax.set_xlabel("Year", fontweight="bold", fontsize=16)
    ax.set_ylabel("GtCO$_2$ per year", fontweight="bold", fontsize=16)
    ax.set_xticks(np.arange(2025, 2061, 5))
    ax.set_xlim(2021, 2060)
    ax.tick_params(labelsize=16)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()


def main(path=None):
    results, path = load_results(path)
    print(f"Results file: {path}")

    c = compute(results)
    pc, gt_2021, EX_crude_2021, unit_value_sar = c["pc"], c["gt_2021"], c["EX_crude_2021"], c["unit_value_sar"]
    variants, territorial, paths = c["variants"], c["territorial"], c["paths"]
    print(f"Crude export value 2021 (IOT): {EX_crude_2021 / 1e6:.1f} bn SAR")
    print(f"Implied export unit value 2021: {unit_value_sar:.0f} SAR/bbl "
          f"= {unit_value_sar / SAR_PER_USD:.1f} USD/bbl")

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
