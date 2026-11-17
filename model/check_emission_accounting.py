"""Read-only check of the emission and energy accounting of a saved KEN run (November 2026).

Run from the repository root:  python -m model.check_emission_accounting saved_results/<file>.pkl

Prints four blocks:
  1. 2021: model against the national inventory (Kingdom of Saudi Arabia, First Biennial Transparency
     Report, National GHG Inventory Document, UNFCCC, March 2025) by inventory category.
  2. 2060: gross emissions by group of sectors, process emissions, capture and removals, net with and
     without CCU, per scenario.
  3. Electricity balance: years in which the demand cap binds, electricity added by electrification,
     electricity of the sectors that are not electrified, desalination kWh per m3.
  4. Oil and gas use in 2021 (EJ, combustion).

What a pass looks like:
  - 2021 national total equals the inventory total, combustion equals the Energy Institute value.
  - Inventory categories within about 15% (the allocation of GCAM energy to IO sectors differs, see notes).
  - The electricity demand cap binds in no year (block 3), electricity of the non-electrified sectors in
    the transformation scenario is not below the baseline, desalination needs at least 2 kWh per m3.
  - Air and water transport are not electrification targets.
"""
import sys
import pickle
import numpy as np

# Inventory 2021, Gg CO2e (AR5). Table numbers of the BTR1 inventory document.
BTR_2021 = {
    "total excl. LULUCF (sum of Table 1 sectors)": 717_207,
    "1.A.1 energy industries (Table 6)": 332_814,
    "   electricity generation": 205_579,
    "   combined heat and power": 101_811,
    "   petroleum refining": 25_424,
    "1.A.2 manufacturing and construction (Table 7)": 84_040,
    "1.A.3 transport (Table 8, road 128,779)": 133_600,
    "1.A.4 other sectors (Table 9)": 10_060,
    "1.B fugitive, oil and gas (Table 10)": 17_312,
    "IPPU (Table 1)": 84_270,
    "agriculture (Table 1)": 9_205,
    "waste (Table 1)": 44_110,
    "memo: international bunkers (Table 5)": 12_960,
}


def groups(sectors, pc):
    ix = sectors.index
    power = [pc.electricity_sector_idx]
    refining = [ix('Manufacture of coke and refined petroleum products'), ix('Extraction of crude petroleum and natural gas')]
    transport = [i for i, n in enumerate(sectors) if n.startswith(('Land transport', 'Water transport', 'Air transport', 'Warehousing', 'Postal'))]
    trade = [i for i, n in enumerate(sectors) if 'trade' in n.lower()]
    used = set(power + refining + transport + trade)
    manu = [i for i, n in enumerate(sectors) if i not in used and (n.startswith('Manufacture') or 'Mining' in n or 'quarrying' in n.lower()
            or n.startswith(('Construction', 'Civil', 'Specialized', 'Repair and installation')))]
    used |= set(manu)
    other = [i for i in range(len(sectors)) if i not in used]
    return {"power": power, "refining and extraction": refining, "manufacturing, mining, construction": manu,
            "transport sectors": transport, "trade (road fuel proxy)": trade, "other services": other}


def combustion(r, sc, t):
    x = r[sc][0]
    s, pc = x['results_sectoral'], x['parameters_calibrated']
    O, G = np.asarray(s['oil_use_final']), np.asarray(s['gas_use_final'])
    return O[t] * pc.oil_emission_factor + G[t] * pc.gas_emission_factor, O[t], G[t]


def main(path):
    r = pickle.load(open(path, 'rb'))
    scen = [k for k in ('Baseline', 'Vision_2030', 'Transformation') if k in r]
    x0 = r[scen[0]][0]
    pc = x0['parameters_calibrated']
    sectors = pc.sectors
    g = groups(sectors, pc)
    T60 = 2060 - 2021

    print("=" * 90)
    print("1. 2021: model against the national inventory (Mt CO2e)")
    e, O, G = combustion(r, scen[0], 0)
    m = x0['results_macro']
    P = np.asarray(x0['results_sectoral']['process_emissions'])[0]
    model_rows = {
        "total excl. LULUCF (sum of Table 1 sectors)": m['emissions_gross_total'].iloc[0],
        "1.A.1 energy industries (Table 6)": e[g["power"]].sum() + e[g["refining and extraction"]].sum(),
        "   electricity generation": e[g["power"]].sum(),
        "   petroleum refining": e[g["refining and extraction"]].sum(),
        "1.A.2 manufacturing and construction (Table 7)": e[g["manufacturing, mining, construction"]].sum(),
        "1.A.3 transport (Table 8, road 128,779)": e[g["transport sectors"]].sum() + e[g["trade (road fuel proxy)"]].sum(),
        "1.A.4 other sectors (Table 9)": e[g["other services"]].sum(),
    }
    print(f"{'item':58s} {'inventory':>10s} {'model':>8s}")
    for k, v in BTR_2021.items():
        mv = model_rows.get(k)
        print(f"{k:58s} {v / 1000:10.1f} {'' if mv is None else f'{mv:8.1f}'}")
    print(f"{'model: combustion CO2 (Energy Institute 572.4)':58s} {'':10s} {e.sum():8.1f}")
    print(f"{'model: process and other (residual)':58s} {'':10s} {P.sum():8.1f}")
    print("Notes: model transport = transport sectors + trade (road fuel of GCAM is allocated to trade and service")
    print("sectors through their purchases of refined products); CHP heat of the inventory is mostly in model refining.")

    print("=" * 90)
    print("2. 2060: gross emissions by group, capture and removals, net (Mt CO2e)")
    hdr = ["power", "refining and extraction", "manufacturing, mining, construction", "transport sectors",
           "trade (road fuel proxy)", "other services"]
    for sc in scen:
        x = r[sc][0]; m = x['results_macro']
        e, _, _ = combustion(r, sc, T60)
        proc = np.asarray(x['results_sectoral']['process_emissions'])[T60].sum()
        ccs, ccu = m['flow_remove_ccs'].iloc[T60], m['flow_reuse_recycle'].iloc[T60] if 'flow_reuse_recycle' in m else np.nan
        nbs, net = m['flow_remove_nature'].iloc[T60], m['emissions_net_total'].iloc[T60]
        print(f"-- {sc}")
        for k in hdr:
            print(f"   {k:40s} {e[g[k]].sum():8.1f}")
        print(f"   {'process and other':40s} {proc:8.1f}")
        print(f"   {'gross':40s} {m['emissions_gross_total'].iloc[T60]:8.1f}")
        print(f"   capture and removals: CCS {ccs:.1f}, CCU {ccu:.1f}, afforestation {nbs:.1f}")
        print(f"   {'net':40s} {net:8.1f}")
        print(f"   {'net without CCU':40s} {net + ccu:8.1f}")

    print("=" * 90)
    print("3. Electricity balance")
    tg_ref = r[scen[-1]][0]['parameters_calibrated'].electrification_target_sectors   # transformation targets
    for sc in scen:
        x = r[sc][0]; m = x['results_macro']; s = x['results_sectoral']; pcs = x['parameters_calibrated']; p = x['parameters']
        el = m['electricity_use_total'].values
        gfc = m['Y_growth_rate_forecast'].values
        E = np.asarray(s['electricity_use'])
        tg = pcs.electrification_target_sectors
        on = getattr(p, 'enable_industrial_electrification', False)
        # cap binds when total demand equals last year's demand times (1 + forecast); with electrification
        # the cap applies to the non-electrified part, so this test is only indicative there
        binds = sum(abs(el[t] / el[t - 1] - 1 - gfc[t]) < 1e-6 for t in range(6, T60 + 1))
        non_tg = E[T60].sum() - E[T60][tg_ref].sum()   # sectors that the transformation does not electrify
        W = m['water_use_desal_tot'].values if 'water_use_desal_tot' in m else None
        desal = E[T60][pcs.sector_desal] / 3.6e6 / (W[T60] / 1e9) if W is not None and W[T60] > 0 else np.nan
        print(f"-- {sc}: electricity 2060 {el[T60] / 3.6e6:7.0f} TWh, cap binds in {binds} of {T60 - 5} years,"
              f" electrification {'on' if on else 'off'}, desalination {desal:.2f} kWh/m3")
        if on:
            print(f"   target sectors ({len(tg)}): {', '.join(pcs.sectors[i][:28] for i in tg)}")
        print(f"   electricity of the transformation's target sectors {E[T60][tg_ref].sum() / 3.6e6:.0f} TWh,"
              f" of all other sectors {non_tg / 3.6e6:.0f} TWh")
    print("   (pass: in the transformation, 'all other sectors' is not far below the baseline value)")

    print("=" * 90)
    print("4. Oil and gas use 2021 (EJ, combustion incl. power-plant fuel)")
    _, O, G = combustion(r, scen[0], 0)
    ei = pc.electricity_sector_idx
    print(f"   oil {O.sum() / 1e9:.2f} (power {O[ei] / 1e9:.2f}), gas {G.sum() / 1e9:.2f} (power {G[ei] / 1e9:.2f})")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    main(sys.argv[1])
