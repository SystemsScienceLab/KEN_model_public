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
FDI Input Data Visualization
Bar charts comparing aggregate FDI sectors vs. disaggregated model-sector attribution.
Confirms that the sector mapping preserves totals.

Usage from notebook:
    from model.visuals_FDI_input import plot_fdi_attribution, plot_fdi_decomposition
    plot_fdi_attribution(pc)
    plot_fdi_decomposition(pc)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def _build_comparison_df(pc, fdi_type='net', year=2023):
    """
    Build a DataFrame with one row per FDI aggregate sector,
    columns: 'Aggregate' (original data), 'Disaggregated' (re-aggregated from model sectors).

    Parameters
    ----------
    pc : ParametersCalibrated
    fdi_type : str, one of 'net', 'stock', 'gross'
    year : int
    """
    if fdi_type == 'net':
        agg_df = pc.ts_fdi_aggregate
        disagg_df = pc.ts_fdi
    elif fdi_type == 'stock':
        agg_df = pc.ts_fdi_stock_aggregate
        disagg_df = pc.ts_fdi_stock
    elif fdi_type == 'gross':
        agg_df = pc.ts_fdi_gross_aggregate
        disagg_df = pc.ts_fdi_gross
    else:
        raise ValueError(f"Unknown fdi_type: {fdi_type}")

    # Original aggregate values (drop Total)
    agg_row = agg_df.drop(columns=['Total'], errors='ignore').loc[year]

    # Re-aggregate the disaggregated data by FDI sector
    sector_to_fdi = {name: pc.fdi_sector_mapping[idx]
                     for idx, name in enumerate(pc.sectors)}
    reagg = disagg_df.loc[year].copy()
    reagg.index = [sector_to_fdi[s] for s in disagg_df.columns]
    # Some sectors map to None; drop those for the comparison
    reagg = reagg[reagg.index.notna()]
    reagg = reagg.groupby(reagg.index).sum()

    # Align on common sectors
    common = sorted(set(agg_row.index) & set(reagg.index),
                    key=lambda s: abs(agg_row[s]), reverse=True)

    comparison = pd.DataFrame({
        'Aggregate': [agg_row[s] for s in common],
        'Disaggregated': [reagg[s] for s in common],
    }, index=common)

    return comparison


def plot_fdi_attribution(pc, year=2023, figsize_per_chart=(18, 7)):
    """
    Create bar charts for FDI net flows, stock, and gross inflows.

    For each chart:
      - Paired bars: Aggregate FDI sector value vs. re-aggregated model-sector value.
      - A final pair of bars showing the grand total of both.

    Parameters
    ----------
    pc : ParametersCalibrated instance (must have ts_fdi*, fdi_sector_mapping, sectors)
    year : int, the year to visualize (default 2023)
    figsize_per_chart : tuple, figure size for each chart
    """
    chart_specs = [
        ('net',   f'FDI Net Inflows {year}: Aggregate Sectors vs. Model-Sector Attribution'),
        ('stock', f'FDI Stock {year}: Aggregate Sectors vs. Model-Sector Attribution'),
        ('gross', f'FDI Gross Inflows {year}: Aggregate Sectors vs. Model-Sector Attribution'),
    ]

    for fdi_type, title in chart_specs:
        comp = _build_comparison_df(pc, fdi_type=fdi_type, year=year)

        # Append grand total row
        total_row = pd.DataFrame({
            'Aggregate': [comp['Aggregate'].sum()],
            'Disaggregated': [comp['Disaggregated'].sum()],
        }, index=['GRAND TOTAL'])
        comp_with_total = pd.concat([comp, total_row])

        n = len(comp_with_total)
        x = np.arange(n)
        bar_width = 0.38

        fig, ax = plt.subplots(figsize=figsize_per_chart)

        bars1 = ax.bar(x - bar_width / 2, comp_with_total['Aggregate'] / 1e6,
                       bar_width, label='Aggregate FDI Sectors',
                       color='steelblue', edgecolor='white', linewidth=0.5)
        bars2 = ax.bar(x + bar_width / 2, comp_with_total['Disaggregated'] / 1e6,
                       bar_width, label='Re-aggregated from Model Sectors',
                       color='darkorange', edgecolor='white', linewidth=0.5)

        # Highlight the GRAND TOTAL bars
        bars1[-1].set_edgecolor('black')
        bars1[-1].set_linewidth(2)
        bars2[-1].set_edgecolor('black')
        bars2[-1].set_linewidth(2)

        # Labels
        ax.set_ylabel('SAR (Millions)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        labels = list(comp_with_total.index)
        # Shorten long labels for readability
        short_labels = []
        for lbl in labels:
            if len(lbl) > 28 and lbl != 'GRAND TOTAL':
                short_labels.append(lbl[:26] + '…')
            else:
                short_labels.append(lbl)
        ax.set_xticklabels(short_labels, rotation=55, ha='right', fontsize=9)

        # Bold the GRAND TOTAL tick label
        tick_labels = ax.get_xticklabels()
        tick_labels[-1].set_fontweight('bold')
        tick_labels[-1].set_fontsize(11)

        ax.legend(fontsize=11, loc='upper right')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda val, pos: f'{val:,.0f}'))

        # Add a subtle grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)

        # Annotate the difference on GRAND TOTAL bars
        agg_total = comp_with_total.loc['GRAND TOTAL', 'Aggregate']
        dis_total = comp_with_total.loc['GRAND TOTAL', 'Disaggregated']
        diff = dis_total - agg_total
        max_bar = max(agg_total, dis_total) / 1e6
        ax.annotate(f'Diff = {diff:,.0f} SAR',
                    xy=(n - 1, max_bar),
                    xytext=(n - 1, max_bar * 1.05),
                    ha='center', fontsize=10, fontweight='bold',
                    color='green' if abs(diff) < 1 else 'red')

        plt.tight_layout()
        plt.show()

        # Print numeric summary
        print(f"\n{'─' * 70}")
        print(f"  {title}")
        print(f"{'─' * 70}")
        print(f"  {'Sector':<45} {'Aggregate':>15} {'Disaggregated':>15} {'Diff':>12}")
        print(f"  {'─' * 87}")
        for sector in comp_with_total.index:
            a = comp_with_total.loc[sector, 'Aggregate']
            d = comp_with_total.loc[sector, 'Disaggregated']
            diff_s = d - a
            marker = '  ✓' if abs(diff_s) < 1 else f'  ✗ {diff_s:+,.0f}'
            print(f"  {sector:<45} {a:>15,.0f} {d:>15,.0f}{marker}")
        print()


def plot_fdi_decomposition(pc, year=2023, figsize=(22, 10)):
    """
    Stacked bar chart showing the decomposition of each aggregate FDI sector
    into its constituent model sectors (for net FDI flows only).

    Each bar represents one aggregate FDI sector from ts_fdi_aggregate.
    The bar is subdivided into coloured segments showing how the aggregate
    value was distributed across the individual model sectors via dX weights.

    Parameters
    ----------
    pc : ParametersCalibrated instance
    year : int, the year to visualize (default 2023)
    figsize : tuple, figure size
    """
    import matplotlib.cm as cm
    from matplotlib.patches import Patch

    # --- 1. Build the decomposition data ---
    # For each aggregate FDI sector, collect all model sectors and their allocated values
    agg_df = pc.ts_fdi_aggregate.drop(columns=['Total'], errors='ignore')
    disagg_row = pc.ts_fdi.loc[year]  # Series, indexed by model sector names

    # Reverse mapping: model sector index -> FDI sector name
    fdi_sector_mapping = pc.fdi_sector_mapping

    # Group model sectors by their FDI aggregate sector
    fdi_groups = {}  # fdi_sector_name -> list of (model_sector_name, value)
    for idx, model_name in enumerate(pc.sectors):
        fdi_sec = fdi_sector_mapping[idx]
        if fdi_sec is None:
            continue
        if fdi_sec not in fdi_groups:
            fdi_groups[fdi_sec] = []
        fdi_groups[fdi_sec].append((idx, model_name, disagg_row[model_name]))

    # Order FDI sectors by absolute aggregate value (descending)
    agg_row = agg_df.loc[year]
    fdi_sector_order = sorted(fdi_groups.keys(),
                              key=lambda s: abs(agg_row[s]) if s in agg_row.index else 0,
                              reverse=True)

    # --- 2. Determine the maximum number of model sectors in any group (for colour map) ---
    max_subsectors = max(len(fdi_groups[s]) for s in fdi_sector_order)
    # Use a qualitative colour map with enough distinct colours
    cmap = cm.get_cmap('tab20', max(max_subsectors, 20))

    # --- 3. Build stacked bar data ---
    n_bars = len(fdi_sector_order)
    x = np.arange(n_bars)
    bar_width = 0.65

    fig, ax = plt.subplots(figsize=figsize)

    # We'll collect legend entries for the model sectors
    # To avoid a huge legend, we only label sectors with significant share
    legend_handles = []
    legend_labels_set = set()

    for bar_idx, fdi_sec in enumerate(fdi_sector_order):
        members = fdi_groups[fdi_sec]
        # Sort members by absolute value descending so largest segments are at bottom
        members_sorted = sorted(members, key=lambda m: abs(m[2]), reverse=True)

        # Separate positive and negative components
        pos_members = [(idx, name, val) for idx, name, val in members_sorted if val >= 0]
        neg_members = [(idx, name, val) for idx, name, val in members_sorted if val < 0]

        # Stack positive values upward from 0
        bottom_pos = 0.0
        for i, (midx, mname, mval) in enumerate(pos_members):
            color = cmap(i % 20)
            short_name = mname if len(mname) <= 35 else mname[:33] + '…'
            ax.bar(bar_idx, mval / 1e6, bar_width,
                   bottom=bottom_pos / 1e6,
                   color=color, edgecolor='white', linewidth=0.4)
            # Label segments that are large enough to be visible
            seg_height = mval / 1e6
            if abs(seg_height) > (abs(agg_row[fdi_sec]) / 1e6) * 0.08 and abs(seg_height) > 0.5:
                label_y = (bottom_pos + mval / 2) / 1e6
                ax.text(bar_idx, label_y, f'[{midx}]',
                        ha='center', va='center', fontsize=6.5,
                        fontweight='bold', color='black',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7, linewidth=0))
            bottom_pos += mval

        # Stack negative values downward from 0
        bottom_neg = 0.0
        for i, (midx, mname, mval) in enumerate(neg_members):
            color = cmap((len(pos_members) + i) % 20)
            ax.bar(bar_idx, mval / 1e6, bar_width,
                   bottom=bottom_neg / 1e6,
                   color=color, edgecolor='white', linewidth=0.4)
            seg_height = mval / 1e6
            if abs(seg_height) > 0.5:
                label_y = (bottom_neg + mval / 2) / 1e6
                ax.text(bar_idx, label_y, f'[{midx}]',
                        ha='center', va='center', fontsize=6.5,
                        fontweight='bold', color='black',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7, linewidth=0))
            bottom_neg += mval

        # Draw a thin horizontal marker at the aggregate value for reference
        agg_val = agg_row[fdi_sec] / 1e6
        ax.plot([bar_idx - bar_width / 2 - 0.05, bar_idx + bar_width / 2 + 0.05],
                [agg_val, agg_val],
                color='black', linewidth=1.5, linestyle='-', zorder=5)

    # --- 4. X-axis labels ---
    short_fdi_labels = []
    for s in fdi_sector_order:
        if len(s) > 30:
            short_fdi_labels.append(s[:28] + '…')
        else:
            short_fdi_labels.append(s)
    ax.set_xticks(x)
    ax.set_xticklabels(short_fdi_labels, rotation=55, ha='right', fontsize=9)

    # --- 5. Formatting ---
    ax.set_ylabel('SAR (Millions)', fontsize=12)
    ax.set_title(f'FDI Net Inflows {year}: Decomposition of Aggregate Sectors into Model Sectors\n'
                 f'(Segment labels show [model sector index]; black line = aggregate total)',
                 fontsize=13, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, pos: f'{val:,.0f}'))
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.axhline(y=0, color='black', linewidth=0.8, zorder=3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()

    # --- 6. Print a detailed legend table ---
    print(f"\n{'═' * 100}")
    print(f"  FDI Net Inflows {year}: Model-Sector Decomposition Legend")
    print(f"{'═' * 100}")
    for fdi_sec in fdi_sector_order:
        members = fdi_groups[fdi_sec]
        members_sorted = sorted(members, key=lambda m: abs(m[2]), reverse=True)
        agg_val = agg_row[fdi_sec]
        print(f"\n  ▸ {fdi_sec}  (Aggregate: {agg_val:,.0f} SAR)")
        print(f"    {'Idx':<6} {'Model Sector':<65} {'Value (SAR)':>15} {'Share':>8}")
        print(f"    {'─' * 95}")
        for midx, mname, mval in members_sorted:
            share = (mval / agg_val * 100) if agg_val != 0 else 0
            print(f"    [{midx:<3}]  {mname:<65} {mval:>15,.0f} {share:>7.1f}%")
    print()
