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

# --- Assumptions (paper Section 4.6), with sources ---------------------------------------------------
# Saudi crude oil exports in 2021: 6.22 million barrels per day (Joint Organisations Data Initiative, JODI,
# annual average 2021, reported by Arab News, 20 Feb 2022, "Saudi crude oil exports fell in 2021 even as
# OPEC+ ease output caps"; OPEC Annual Statistical Bulletin 2022, Table 5.5, gives about 6.2 mb/d).
CRUDE_EXPORTS_2021_BPD = 6.22e6
# Combustion emission factor of crude oil, U.S. EPA Greenhouse Gas Equivalencies Calculator ("barrels of
# oil consumed"): 5.80 MMBtu/barrel x 20.31 kg C/MMBtu x 44/12 kg CO2/kg C = 0.432 t CO2 per barrel
# (EPA, Inventory of U.S. Greenhouse Gas Emissions and Sinks, Annex 2). Heat content is a higher heating value.
EPA_MMBTU_PER_BBL = 5.80
EPA_KG_C_PER_MMBTU = 20.31
EF_T_PER_BBL = EPA_MMBTU_PER_BBL * EPA_KG_C_PER_MMBTU * 44 / 12 / 1000      # = 0.432 tCO2/bbl
# Checks
GJ_PER_MMBTU = 1.055056
NCV_PER_GCV_OIL = 0.95           # net/gross calorific value of oil (IEA/OECD convention)
IPCC_EF_RANGE_T_PER_TJ = (71.1, 75.5)   # IPCC 2006 Guidelines, Vol. 2, Table 1.4: crude oil 73.3 (71.1-75.5) tCO2/TJ (NCV)
SAR_PER_USD = 3.75               # fixed exchange rate (Saudi Central Bank)
OPEC_BASKET_2021_USD = 69.89     # OPEC Reference Basket, 2021 average, USD/bbl (OPEC Annual Statistical Bulletin 2022)
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
    ef_model = report['parameters_calibrated'].oil_emission_factor * 1e6 * 1000  # tCO2/TJ (NCV)
    gj_hhv_per_bbl = EPA_MMBTU_PER_BBL * GJ_PER_MMBTU
    ef_epa_ncv = EF_T_PER_BBL / (gj_hhv_per_bbl * NCV_PER_GCV_OIL) * 1000      # tCO2/TJ (NCV)
    lo, hi = IPCC_EF_RANGE_T_PER_TJ
    ex, p = _crude(report)
    barrels_2021 = CRUDE_EXPORTS_2021_BPD * 365
    usd_per_bbl = ex.loc[1] / p.loc[1] * 1000 / SAR_PER_USD / barrels_2021
    ok = lambda c: 'OK' if c else 'CHECK'
    print('ASSUMPTION CHECK (paper Section 4.6)')
    print(f'  Crude exports 2021: {CRUDE_EXPORTS_2021_BPD / 1e6:.2f} million bbl/day (JODI) = {barrels_2021 / 1e9:.2f} billion bbl')
    print(f'  Emission factor: {EF_T_PER_BBL:.3f} tCO2/bbl (EPA) = {ef_epa_ncv:.1f} tCO2/TJ on a net-calorific basis; '
          f'IPCC 2006: 73.3 ({lo}-{hi}); model: {ef_model:.1f}  [{ok(lo <= ef_epa_ncv <= hi)}]')
    print(f'  Implied 2021 export unit value: {usd_per_bbl:.1f} USD/bbl; OPEC basket 2021: {OPEC_BASKET_2021_USD} USD/bbl  '
          f'[{ok(abs(usd_per_bbl / OPEC_BASKET_2021_USD - 1) < 0.05)}]')
    print(f'  Emissions of exported crude, 2021: {barrels_2021 * EF_T_PER_BBL / 1e9:.2f} GtCO2\n')


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
