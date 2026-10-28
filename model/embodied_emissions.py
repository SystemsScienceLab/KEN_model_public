"""Emissions embodied in oil exports: paper Section 4.6, Table 4 (accounting extension outside the model).

Crude oil exports (sector 'Extraction of crude petroleum and natural gas') are deflated with the crude
sector's price, converted to barrels by anchoring 2021 to Saudi crude exports, and multiplied by a
combustion emission factor. Range: lower bound = oil price rises with the Baseline's nominal exports
(Baseline volume held at its 2021 level); upper bound = model-consistent deflation (sector price p).

Usage (run_model_COMPARE_scenarios.ipynb):
    from model.embodied_emissions import embodied_emissions_table
    display(embodied_emissions_table(exp_reports))
"""
import pandas as pd

CRUDE_EXPORTS_2021_BPD = 6.2e6   # Saudi crude exports 2021, barrels/day (OPEC 2022; JODI 2023)
EF_T_PER_BBL = 0.43              # combustion emission factor, tCO2 per barrel (U.S. EPA 2024, 2026)
SAR_PER_USD = 3.75
OPEC_BASKET_2021_USD = 69.9      # OPEC Reference Basket 2021 average, USD/bbl (OPEC 2022)
NCV_CRUDE_GJ_PER_T = 42.3        # IPCC 2006 net calorific value of crude oil
BBL_PER_T = 7.33                 # barrels per tonne of crude (typical)
IPCC_EF_RANGE_T_PER_TJ = (71.1, 75.5)   # IPCC 2006 95% interval for crude oil (default 73.3)
YEARS_T = list(range(1, 41))     # t = 1..40 is 2021..2060
CRUDE = 'Extraction of crude petroleum and natural gas'


def _crude(report):
    i = report['parameters_calibrated'].sectors.index(CRUDE)
    ex = report['results_sectoral']['EX'].iloc[:, i].astype(float)   # thousand SAR, nominal
    p = report['results_sectoral']['p'].iloc[:, i].astype(float)     # sector price index
    return ex, p


def _mean_runs(runs, f):
    return pd.concat([f(r) for r in runs], axis=1).mean(axis=1)


def check_assumptions(exp_reports):
    """Print a check of the assumptions of Section 4.6 against the model and the literature."""
    report = next(iter(exp_reports.values()))[0]
    ef_model = report['parameters_calibrated'].oil_emission_factor * 1e6       # tCO2/GJ
    gj_per_bbl = NCV_CRUDE_GJ_PER_T / BBL_PER_T
    lo, hi = (x / 1000 * gj_per_bbl for x in IPCC_EF_RANGE_T_PER_TJ)
    ex, p = _crude(report)
    barrels_2021 = CRUDE_EXPORTS_2021_BPD * 365
    usd_per_bbl = ex.loc[1] / p.loc[1] * 1000 / SAR_PER_USD / barrels_2021
    print('ASSUMPTION CHECK')
    print(f'  Model oil emission factor (IPCC Tier 1): {ef_model * 1000:.1f} tCO2/TJ')
    print(f'  {EF_T_PER_BBL} tCO2/bbl implies {EF_T_PER_BBL / ef_model:.2f} GJ/bbl; crude oil has about {gj_per_bbl:.2f} GJ/bbl (NCV)')
    print(f'  IPCC 95% range: {lo:.3f}-{hi:.3f} tCO2/bbl -> {EF_T_PER_BBL} is '
          f'{"inside" if lo <= EF_T_PER_BBL <= hi else "OUTSIDE"} the range')
    print(f'  Implied 2021 export unit value: {usd_per_bbl:.1f} USD/bbl (OPEC basket 2021: {OPEC_BASKET_2021_USD} USD/bbl)')
    print(f'  Exported crude 2021: {barrels_2021 * EF_T_PER_BBL / 1e9:.2f} GtCO2\n')


def embodied_emissions_table(exp_reports, check=True):
    """Table 4: cumulative emissions 2021-2060 (GtCO2), territorial (model output) and embodied in
    exported crude (range: lower to upper bound, see module docstring)."""
    if check:
        check_assumptions(exp_reports)
    barrels_2021 = CRUDE_EXPORTS_2021_BPD * 365
    base_ex = _mean_runs(exp_reports['Baseline'], lambda r: _crude(r)[0]) if 'Baseline' in exp_reports else None
    cols = {}
    for sc, runs in exp_reports.items():
        ex = _mean_runs(runs, lambda r: _crude(r)[0])
        real = _mean_runs(runs, lambda r: _crude(r)[0] / _crude(r)[1])
        hi = barrels_2021 * EF_T_PER_BBL * real / real.loc[1] / 1e9                # GtCO2 per year
        lo = barrels_2021 * EF_T_PER_BBL * ex / base_ex / 1e9 if base_ex is not None else hi
        gross = _mean_runs(runs, lambda r: r['results_macro']['emissions_gross_total'].astype(float)) / 1e3
        net = _mean_runs(runs, lambda r: r['results_macro']['emissions_net_total'].astype(float)) / 1e3
        cols[sc] = {'gross': gross.loc[YEARS_T].sum(), 'net': net.loc[YEARS_T].sum(),
                    'lo': lo.loc[YEARS_T].sum(), 'hi': hi.loc[YEARS_T].sum(), 'hi_2060': hi.loc[YEARS_T[-1]]}
    b = cols.get('Baseline')
    rng = lambda lo, hi: f'{lo:.1f}–{hi:.1f}'
    avoided = lambda sc, s: '—' if sc == 'Baseline' or b is None else s
    table = pd.DataFrame({sc: {
        'Territorial cumulative emissions, gross': f"{c['gross']:.1f}",
        'Territorial cumulative emissions, net': f"{c['net']:.1f}",
        'Combustion of exported crude, cumulative': rng(c['lo'], c['hi']),
        'Avoided vs. baseline, territorial gross': avoided(sc, f"{b['gross'] - c['gross']:.1f}" if b else ''),
        'Avoided vs. baseline, territorial net': avoided(sc, f"{b['net'] - c['net']:.1f}" if b else ''),
        'Avoided vs. baseline, exported crude': avoided(sc, rng(b['lo'] - c['lo'], b['hi'] - c['hi']) if b else ''),
        'Exported crude in 2060 (GtCO2/yr, upper)': f"{c['hi_2060']:.2f}",
    } for sc, c in cols.items()})
    table.index.name = 'Cumulative 2021–2060, GtCO2'
    return table
