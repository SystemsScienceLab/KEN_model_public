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

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- GLOBAL FONT SIZE SETTINGS ---
# This ensures that ALL plots automatically inherit size 16 
# for labels, ticks, and legends, keeping everything uniform.
plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=16)

# Mapping from internal scenario key   human-readable legend label.
# Only scenarios that need a different display name require an entry.
SCENARIO_LABELS = {
    "Transformation": "Transformation",
}


 # 
# PRIVATE HELPERS - overlap resolution & GDP-share endpoint annotations
 # 

def _resolve_overlaps_pts(points, min_gap):
    """Nudge overlapping annotation y-positions upward to prevent collisions."""
    if not points:
        return []
    pts = sorted([dict(p) for p in points], key=lambda p: p['y'])
    for i in range(1, len(pts)):
        if pts[i]['y'] - pts[i - 1]['y'] < min_gap:
            pts[i]['y'] = pts[i - 1]['y'] + min_gap
    return pts


def _add_endpoint_annotations(ax, start_annots, end_annots, years_ref, caption_text='% of GDP Y (Tn SAR for 2021 and 2060)', caption_loc='bottom_left', start_x_offset=1.5):
    """
    Place % of GDP labels at the left (start) and right (end) of plotted lines.
    """
    if years_ref is None or not (start_annots or end_annots):
        return

    all_pts = list(start_annots or []) + list(end_annots or [])
    all_y = [p['real_y'] for p in all_pts]
    y_range = (max(all_y) - min(all_y)) if len(all_y) > 1 else (
        abs(max(all_y, key=abs)) * 0.3 if all_y else 1.0)
    if y_range <= 0:
        y_range = max(abs(v) for v in all_y) * 0.3 if any(v != 0 for v in all_y) else 1.0
    min_gap = max(y_range * 0.07, 1e-9)

    x_start = years_ref[0]
    x_end   = years_ref[-1]

    # Snap start annotation y-positions between ytick marks to avoid visual overlap with axis labels
    if start_annots:
        _ylim = ax.get_ylim()
        _yticks = [t for t in ax.get_yticks() if _ylim[0] <= t <= _ylim[1]]
        if len(_yticks) >= 2:
            _sorted_ticks = np.sort(np.array(_yticks, dtype=float))
            _tick_gap = float(np.median(np.diff(_sorted_ticks)))
            _snap_thresh = _tick_gap * 0.28
            for _p in start_annots:
                for _i, _tk in enumerate(_sorted_ticks):
                    if abs(_p['y'] - _tk) < _snap_thresh:
                        if _i + 1 < len(_sorted_ticks):
                            _p['y'] = (_sorted_ticks[_i] + _sorted_ticks[_i + 1]) / 2.0
                        elif _i > 0:
                            _p['y'] = (_sorted_ticks[_i - 1] + _sorted_ticks[_i]) / 2.0
                        break

    def _place(annots, x_base, x_text, ha):
        for p in _resolve_overlaps_pts(annots, min_gap):
            needs_arrow = abs(p['y'] - p['real_y']) > min_gap * 0.35
            ax.annotate(
                p['label'],
                xy=(x_base, p['real_y']),
                xytext=(x_text, p['y']),
                color=p['color'], fontsize=16, fontweight='bold',
                ha=ha, va='center',
                arrowprops=dict(arrowstyle='-', color=p['color'],
                                lw=0.7, alpha=0.5) if needs_arrow else None,
            )

    if start_annots:
        _place(start_annots, x_start, x_start - start_x_offset, 'right')
    if end_annots:
        _place(end_annots, x_end, x_end + 1.5, 'left')

    # Extend x-axis so labels don't get clipped
    cur = ax.get_xlim()
    ax.set_xlim(
        min(cur[0], x_start - (start_x_offset + 2.5)) if start_annots else cur[0],
        max(cur[1], x_end   + 5.0) if end_annots   else cur[1],
    )
    
    # Check if this axis is the top-left (first) subplot in a grid
    if caption_text:
        is_first_subplot = True
        if ax is not None and hasattr(ax, 'get_subplotspec'):
            row_idx = ax.get_subplotspec().rowspan.start
            col_idx = ax.get_subplotspec().colspan.start
            if row_idx > 0 or col_idx > 0:
                is_first_subplot = False
                
        # Only draw the text if it is the first subplot (or a standalone figure)
        if is_first_subplot:
            if caption_loc == 'top_right':
                ax.text(0.99, 0.99, caption_text,
                        transform=ax.transAxes, fontsize=14, color='black',
                        style='italic', va='top', ha='right')
            else:
                ax.text(0.01, 0.01, caption_text,
                        transform=ax.transAxes, fontsize=14, color='black',
                        style='italic', va='bottom', ha='left')

 # 
# PRIVATE HELPER - panel drawing
 # 

def _plot_panel(ax, reports, variables, title, scenario_colors, ylabel="Trillion SAR", zero_line=False, legend_fontsize=16, gdp_share=False, show_legend=True, legend_loc='best', caption_text='% of GDP Y (Tn SAR for 2021 and 2060)', caption_loc='bottom_left', start_filter_scenario=None, start_ratio_overrides=None, start_x_offset=1.5):
    """
    Draw one thematic panel for scenario-comparison multi-figure plots.
    """
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    line_styles = ['-', '--', '-.', ':']

    years = None
    _gdp_entries = []  # (color, mean_vals, Y_mean) when gdp_share=True

    for s_idx, (scenario, runs) in enumerate(reports.items()):
        color = scenario_colors.get(scenario, default_colors[s_idx % len(default_colors)])

        for v_idx, (col_name, var_label) in enumerate(variables):
            ls = line_styles[v_idx]
            data = []
            y_for_share = []
            for run_report in runs:
                macro = run_report["results_macro"]
                if col_name not in macro.columns:
                    continue
                data.append(macro[col_name].values / 1_000_000_000)
                if gdp_share and 'Y' in macro.columns:
                    y_for_share.append(macro['Y'].values / 1_000_000_000)

            if not data:
                continue

            if years is None:
                years = list(range(2021, 2021 + len(data[0])))

            arr = np.array(data)
            mean_vals = np.mean(arr, axis=0)
            std_vals  = np.std(arr, axis=0)

            # Legend label: include variable name only when there are multiple vars per panel
            if len(variables) == 1:
                lbl = SCENARIO_LABELS.get(scenario, scenario)
            else:
                lbl = f"{SCENARIO_LABELS.get(scenario, scenario)} - {var_label}"

            ax.plot(years, mean_vals, label=lbl, color=color,
                    linestyle=ls, linewidth=2.0)
            ax.fill_between(years,
                            mean_vals - std_vals,
                            mean_vals + std_vals,
                            alpha=0.12, color=color)

            if gdp_share and y_for_share:
                Y_mean = np.mean(np.array(y_for_share), axis=0)
                _gdp_entries.append((color, mean_vals, Y_mean, scenario, v_idx))

    if years is not None:
        custom_ticks = [2021] + list(range(2025, years[-1] + 1, 5))
        ax.set_xticks(custom_ticks)
        ax.set_xticklabels(custom_ticks, rotation=0, fontsize=16)

    if zero_line:
        ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')

    ax.set_title(title, fontsize=14, fontweight='bold')
    
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=16)
        
    if show_legend:
        ax.legend(loc=legend_loc, fontsize=legend_fontsize, frameon=True)
        
    ax.grid(True, alpha=0.3)

    # GDP-share endpoint annotations 
    if gdp_share and _gdp_entries and years:
        _has_filter = start_filter_scenario is not None
        # show start labels for single-variable panels OR when a filter scenario is set
        show_start = _has_filter or (len(_gdp_entries) <= len(reports))
        start_ann, end_ann = [], []
        for color, mean_v, Y_mean, _scen, _v_idx in _gdp_entries:
            if Y_mean[0] > 0 and Y_mean[-1] > 0:
                # End annotations: always show for all scenarios; no T-value suffix when filtered
                _abs_sfx_e = '' if _has_filter else (f" ({mean_v[-1]:.1f}T)" if show_start else '')
                end_ann.append({'y': mean_v[-1], 'real_y': mean_v[-1],
                                'label': f"{mean_v[-1] / Y_mean[-1] * 100:.1f}%{_abs_sfx_e}",
                                'color': color})
                # Start annotations: only when show_start; filtered to one scenario if set
                if show_start and (not _has_filter or _scen == start_filter_scenario):
                    # Resolve override: None=compute, False=skip, float=ratio, (float,str)=ratio+suffix
                    _override = (start_ratio_overrides[_v_idx]
                                 if (start_ratio_overrides is not None and _v_idx < len(start_ratio_overrides))
                                 else None)
                    if _override is not False:
                        if isinstance(_override, tuple):
                            _start_pct, _lbl_sfx = _override[0] * 100, _override[1]
                        elif _override is not None:
                            _start_pct, _lbl_sfx = _override * 100, ''
                        else:
                            _start_pct, _lbl_sfx = mean_v[0] / Y_mean[0] * 100, ''
                        start_ann.append({'y': mean_v[0], 'real_y': mean_v[0],
                                          'label': f"{_start_pct:.1f}%{_lbl_sfx}",
                                          'color': color})
        _add_endpoint_annotations(ax, start_ann, end_ann, years, caption_text=caption_text, caption_loc=caption_loc, start_x_offset=start_x_offset)

def _plot_panel_raw(ax, reports, col, title, ylabel, scenario_colors,
                    scale=1.0, zero_line=False, reference_line=None, show_legend=True, legend_loc='best', legend_fontsize=16):
    """
    Draw one panel for a single macro variable across scenarios.

    Unlike _plot_panel (which divides by 1,000,000,000), this helper multiplies
    the raw stored value by `scale`.  Use scale=100 for decimal percentage or
    decimal index-base-100 conversions (inflation rates, price indices).
    """
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    years = None

    for s_idx, (scenario, runs) in enumerate(reports.items()):
        color = scenario_colors.get(scenario, default_colors[s_idx % len(default_colors)])
        data = []
        for rr in runs:
            macro = rr["results_macro"]
            if col not in macro.columns:
                continue
            data.append(macro[col].values * scale)
            if years is None:
                years = list(range(2021, 2021 + len(macro[col])))
        if not data:
            continue
        arr = np.array(data)
        mean_v = np.mean(arr, axis=0)
        std_v  = np.std(arr, axis=0)
        ax.plot(years, mean_v, label=SCENARIO_LABELS.get(scenario, scenario), color=color, linewidth=2.0)
        ax.fill_between(years, mean_v - std_v, mean_v + std_v,
                        alpha=0.12, color=color)

    if years is not None:
        ticks = [2021] + list(range(2025, years[-1] + 1, 5))
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, rotation=0, fontsize=16)

    if zero_line:
        ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
    if reference_line is not None:
        ax.axhline(y=reference_line, color='grey', linewidth=0.8,
                   linestyle='--', label=f'Base ({reference_line:.0f})')

    ax.set_title(title, fontsize=14, fontweight='bold')
    
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=16)
        
    if show_legend:
        ax.legend(loc=legend_loc, fontsize=legend_fontsize, frameon=True)
        
    ax.grid(True, alpha=0.3)


# 
# MACRO SCENARIO COMPARISON  (6-panel)
# 

def plot_macro_comparison(reports):
    """
    Six-panel scenario comparison inspired by the 'Multiple Plots for comparison'
    block in postprocessing.py (lines 101-149).

    Each panel shows 1-2 key variables; colour = scenario, line-style = variable.
    Solid line = first variable listed, dashed = second variable.
    All monetary values in Trillion SAR.
    """
    scenario_colors = {
        "Baseline":        "#d62728",
        "Vision_2030":     "#ff7f0e",
        "Net_zero":        "#2ca02c",
        "Transformation":  "#1f77b4",
        "Steady_state":    "#7f7f7f",
    }

    sns.set_style("whitegrid")
    fig, axs = plt.subplots(3, 2, figsize=(18, 22))
    ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = axs

    # start_ratio_overrides: calibrated GASTAT 2021 ratios used as fixed start labels
    # False=skip label, (ratio, suffix)=ratio with appended text; overrides bypass model t=1 deviation
    _gdp_start_overrides = [(0.4828, ' (non-oil)'), (0.2806, ' (oil)'), (0.2366, ' (other)')]
    _plot_panel(ax1, reports,
                [('Y_non_oil',   'Non-oil GDP'),
                 ('Y_oil',       'Oil GDP'),
                 ('Y_other_gdp', 'Other (Gov+Taxes)')],
                title='GDP - Non-oil, Oil & Other',

                scenario_colors=scenario_colors, ylabel="Trillion SAR", zero_line=False,
                legend_fontsize=16, show_legend=True, legend_loc='upper left',
                gdp_share=True, start_filter_scenario='Baseline',
                start_ratio_overrides=_gdp_start_overrides, start_x_offset=4.0)

    _plot_panel(ax2, reports,
                [('C',        'Consumption C'),
                 ('I_total',  'Investment')],
                title='Demand - Consumption & Investment',
                scenario_colors=scenario_colors, ylabel="", zero_line=False, 
                legend_fontsize=16, show_legend=True, legend_loc='best',
                gdp_share=True)

    _plot_panel(ax3, reports,
                [('YD_wage',   'Wage income'),
                 ('YD_profit', 'Profit income')],
                title='Households - Disposable Income',
                scenario_colors=scenario_colors, ylabel="Trillion SAR", zero_line=False, 
                show_legend=True, legend_loc='best', legend_fontsize=16,
                gdp_share=True)

    _plot_panel(ax4, reports,
                [('EX',     'Total Exports'),
                 ('IM',     'Imports'),
                 ('EX_oil', 'Oil Exports')],
                title='Trade - Exports & Imports',
                scenario_colors=scenario_colors, ylabel="", zero_line=False, 
                show_legend=True, legend_loc='best', legend_fontsize=16,
                gdp_share=True)

    _plot_panel(ax5, reports,
                [('Gov_rev', 'Revenue'),
                 ('Gov_exp', 'Total expenditure')],
                title='Government - Revenue & Total Expenditure',
                scenario_colors=scenario_colors, ylabel="Trillion SAR", zero_line=False, 
                show_legend=True, legend_loc='best', legend_fontsize=16,
                gdp_share=True)

    _plot_panel(ax6, reports,
                [('P', 'Profits P'),
                 ('W', 'Wages W')],
                title='Firms - Profits & Wages',
                scenario_colors=scenario_colors, ylabel="", zero_line=False, 
                show_legend=True, legend_loc='best', legend_fontsize=16,
                gdp_share=True)

    for _i, _ax in enumerate(axs.flatten()):
        _ax.text(-0.05, 1.05, f"({chr(ord('a') + _i)})", transform=_ax.transAxes,
                 fontsize=16, fontweight='bold', va='top')
    plt.tight_layout()
    plt.show()


 #
# GDP EXPENDITURE IDENTITY DECOMPOSITION  (standalone, one panel per scenario)
 #

# Component colours. These are keyed by GDP COMPONENT, not by scenario, so the
# module-level scenario_colors cannot be reused here. Validated as an adjacent-pair
# categorical set on a white surface: worst CVD dE 9.1 (>=8 target), worst
# normal-vision dE 19.6 (>=15 floor). Three of the five sit below 3:1 contrast on
# white, so the direct share labels drawn inside each band are mandatory, not
# decorative - identity must never rest on colour alone.
GDP_COMPONENT_COLORS = {
    'Y_C':  '#2a78d6',   # blue
    'Y_G':  '#eb6834',   # orange
    'Y_I':  '#1baf7a',   # aqua
    'Y_EX': '#eda100',   # yellow
    'Y_IM': '#e87ba4',   # magenta
}

# Stacked in GDP identity order: Y_ = Y_C_ + Y_G_ + Y_I_ + Y_EX_ - Y_IM_
GDP_COMPONENT_LABELS = [
    ('Y_C',  'Consumption C'),
    ('Y_G',  'Government consumption G'),
    ('Y_I',  'Investment I (demand)'),
    ('Y_EX', 'Exports EX'),
]


def plot_gdp_identity_decomposition(reports, min_label_share=4.0):
    """
    Show that the GDP expenditure components add up EXACTLY to total GDP.

    One panel per scenario, stacked VERTICALLY so each verification is large and legible.
    Everything is expressed as a share of nominal GDP, so the positive components stack to
    (100 + import share) % and imports are drawn as a deduction below zero:
    stack top - import depth = 100% = GDP.

    Series are taken from the dedicated identity columns Y_C, Y_G, Y_I, Y_EX, Y_IM
    (model_classes.ModelVariables, written in model.py section 12). Those are exact by
    construction. Do NOT rebuild this from the general macro columns: 'Gov_exp' is a
    FISCAL total that also carries subsidies, public investment GI (already inside
    investment), bond interest and repayments, and would overstate G by 9-12 pp of GDP.

    min_label_share: bands thinner than this (in pp of GDP) get no inline label, to
    avoid unreadable overlapping text.
    """
    sns.set_style("whitegrid")

    # ---- pass 1: shares per scenario, so all panels can share one y-scale ----
    # Different y-scales across panels of the SAME measure invite misreading, so the
    # limits are computed globally before anything is drawn.
    all_shares = {}
    for scenario, runs in reports.items():
        shares = {}
        for col in [c for c, _ in GDP_COMPONENT_LABELS] + ['Y_IM']:
            per_run = [rr["results_macro"][col].values / rr["results_macro"]['Y'].values * 100
                       for rr in runs if col in rr["results_macro"].columns]
            shares[col] = np.mean(np.array(per_run), axis=0) if per_run else None
        all_shares[scenario] = shares

    _tops = [sum(s[c] for c, _ in GDP_COMPONENT_LABELS).max()
             for s in all_shares.values() if all(v is not None for v in s.values())]
    _bots = [(-s['Y_IM']).min()
             for s in all_shares.values() if all(v is not None for v in s.values())]
    _ylim = ((min(_bots) * 1.18, max(_tops) * 1.06) if _tops and _bots else None)

    # Vertically stacked: one full-width row per scenario, so each verification is large.
    # The suptitle / legend / caption bands are reserved in INCHES and converted to figure
    # fractions, because the figure height grows with the number of scenarios and fixed
    # fractional offsets would drift as panels are added.
    n = len(reports)
    _panel_h, _top_band, _bottom_band = 5.2, 0.85, 1.85
    _fig_h = _panel_h * n + _top_band + _bottom_band
    fig, axs = plt.subplots(n, 1, figsize=(15, _fig_h), squeeze=False, sharey=True)
    axs = axs[:, 0]

    worst_resid_pct = 0.0

    for ax_idx, (scenario, runs) in enumerate(reports.items()):
        ax = axs[ax_idx]
        shares = all_shares[scenario]

        if any(v is None for v in shares.values()):
            ax.text(0.5, 0.5, f"Identity columns missing\nfor {scenario}",
                    ha='center', va='center', transform=ax.transAxes, fontsize=16)
            continue

        years = list(range(2021, 2021 + len(shares['Y_C'])))

        # --- positive stack, in identity order ---
        ax.stackplot(years,
                     *[shares[c] for c, _ in GDP_COMPONENT_LABELS],
                     labels=[lbl for _, lbl in GDP_COMPONENT_LABELS],
                     colors=[GDP_COMPONENT_COLORS[c] for c, _ in GDP_COMPONENT_LABELS],
                     edgecolor='white', linewidth=2.0)   # 2px surface gap between fills

        # --- imports as a deduction below zero (national accounts convention) ---
        ax.fill_between(years, 0, -shares['Y_IM'],
                        color=GDP_COMPONENT_COLORS['Y_IM'], label='Imports IM (deduction)',
                        edgecolor='white', linewidth=2.0)

        # --- the identity itself: computed sum, which must lie ON the 100% line ---
        computed = (shares['Y_C'] + shares['Y_G'] + shares['Y_I']
                    + shares['Y_EX'] - shares['Y_IM'])
        # The computed sum is drawn in the surface colour ON TOP of the solid GDP line, so
        # the GDP line reads as dashed exactly where the two coincide - and separates
        # visibly if the identity ever breaks again. It carries no legend entry of its own:
        # a white line is invisible against the legend surface, so the GDP line's label
        # explains the dashes instead.
        ax.axhline(100, color='#0b0b0b', linewidth=3.0, zorder=6,
                   label='GDP Y = 100%  (dashes = computed C + G + I + EX - IM)')
        ax.plot(years, computed, color='#fcfcfb', linewidth=1.4, linestyle=(0, (4, 3)),
                zorder=7, label='_nolegend_')

        resid_pct = float(np.max(np.abs(computed - 100.0)))
        worst_resid_pct = max(worst_resid_pct, resid_pct)

        # --- direct labels: mandatory relief for the sub-3:1 component colours ---
        for col, _ in GDP_COMPONENT_LABELS:
            base = np.zeros(len(years))
            for c2, _ in GDP_COMPONENT_LABELS:
                if c2 == col:
                    break
                base = base + shares[c2]
            for x_idx, ha in ((0, 'left'), (len(years) - 1, 'right')):
                if shares[col][x_idx] < min_label_share:
                    continue
                ax.text(years[x_idx], base[x_idx] + shares[col][x_idx] / 2,
                        f"{shares[col][x_idx]:.0f}%", ha=ha, va='center',
                        fontsize=13, fontweight='bold', color='#0b0b0b')
        for x_idx, ha in ((0, 'left'), (len(years) - 1, 'right')):
            ax.text(years[x_idx], -shares['Y_IM'][x_idx] / 2,
                    f"-{shares['Y_IM'][x_idx]:.0f}%", ha=ha, va='center',
                    fontsize=13, fontweight='bold', color='#0b0b0b')

        # --- formatting, matching the conventions used elsewhere in this module ---
        ax.set_title(SCENARIO_LABELS.get(scenario, scenario.replace('_', ' ')),
                     fontsize=18, fontweight='bold')
        ax.set_ylabel("Share of nominal GDP (%)", fontsize=16, fontweight='bold')
        custom_ticks = [2021] + list(range(2025, years[-1] + 1, 5))
        ax.set_xticks(custom_ticks)
        ax.set_xticklabels(custom_ticks, rotation=0, fontsize=14)
        ax.axhline(0, color='#c3c2b7', linewidth=1.0)
        ax.set_xlim(years[0], years[-1])
        if _ylim:
            ax.set_ylim(*_ylim)
        # Stacked vertically: the year axis is labelled once, under the bottom panel only.
        # This must follow set_xticklabels, which would otherwise restore the hidden labels.
        if ax_idx == n - 1:
            ax.set_xlabel("Year", fontsize=16, fontweight='bold')
        else:
            ax.tick_params(labelbottom=False)

        ax.text(-0.05, 1.05, f"({chr(ord('a') + ax_idx)})", transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top')

    # One figure-level legend below the panels: inside an axes it would cover the 2021
    # share labels, which are the mandatory relief for the low-contrast component colours.
    # ncol=3 rather than one row: the figure is now narrow, and the GDP-line label is long.
    _handles, _labels = axs[0].get_legend_handles_labels()
    fig.legend(_handles, _labels, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, 0.80 / _fig_h), fontsize=14, frameon=False)

    fig.suptitle("GDP expenditure identity verification:\n"
                 "Y = C + G + I + EX - IM", fontsize=20, fontweight='bold')
    fig.text(0.005, 0.005,
             "Stacked positive components reach (100 + import share) %; imports are drawn as a "
             "deduction below zero, so stack top - import depth = 100% of GDP.\n"
             "The dashed line is the sum computed from the plotted series and should coincide with "
             "the 100% GDP line.\n"
             "G is government CONSUMPTION (the GDP component), not total government expenditure "
             "Gov_exp, which also contains subsidies, public investment GI, and debt service.",
             fontsize=12, ha='left', va='bottom', color='#52514e')

    # The numeric counterpart to the dashed overlay: reported to the console rather than on
    # the figure, so the panel caption stays clean. Should be ~1e-14 pp (machine precision).
    print(f"GDP identity verification: worst deviation of (C + G + I + EX - IM) from GDP "
          f"across all scenarios and years = {worst_resid_pct:.2e} pp of GDP")

    plt.tight_layout(rect=[0, _bottom_band / _fig_h, 1, 1 - _top_band / _fig_h])
    plt.show()


 #
# BALANCE OF PAYMENTS SCENARIO COMPARISON  (4-panel, 2 x 2)
 #

 # 
# BALANCE OF PAYMENTS SCENARIO COMPARISON  (4-panel, 2 x 2)
 # 

def plot_bop_comparison(reports):
    """
    Four-panel scenario comparison for the Balance of Payments,
    simplified from postprocessing.py (lines 162-193) for readability.

    Shows the most informative BoP aggregates; drops derived/redundant series
    (secondary_income_balance, primary_income_balance, Gov_ext_assets_income,
    Gov_ext_assets_change, financial_account, BoP_check) to keep panels uncluttered.
    All monetary values in Trillion SAR.
    """
    scenario_colors = {
        "Baseline":        "#d62728",
        "Vision_2030":     "#ff7f0e",
        "Net_zero":        "#2ca02c",
        "Transformation":  "#1f77b4",
        "Steady_state":    "#7f7f7f",
    }

    default_colors_bop = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    def _custom_loop(ax, reports, var_specs, zero_line=False, gdp_share=False, ylabel="Trillion SAR", caption_text='% of GDP Y (Tn SAR for 2021 and 2060)', caption_loc='bottom_left', return_gdp_entries=False):
        """Draw scenario lines with per-variable sign and linestyle; return year list.
        If return_gdp_entries=True, skip the internal _add_endpoint_annotations call and
        return (_years, _gdp_entries) so the caller can merge with other series entries."""
        _years = None
        _gdp_entries = []  # (color, mean_v, Y_mean) when gdp_share=True

        for col, lbl, ls, sign in var_specs:
            for s_idx, (scenario, runs) in enumerate(reports.items()):
                color = scenario_colors.get(scenario, default_colors_bop[s_idx % len(default_colors_bop)])
                data = []
                y_for_share = []
                for rr in runs:
                    macro = rr["results_macro"]
                    if col not in macro.columns:
                        continue
                    data.append(macro[col].values * sign / 1_000_000_000)
                    if _years is None:
                        _years = list(range(2021, 2021 + len(macro[col])))
                    if gdp_share and 'Y' in macro.columns:
                        y_for_share.append(macro['Y'].values / 1_000_000_000)
                if not data:
                    continue
                arr    = np.array(data)
                mean_v = np.mean(arr, axis=0)
                std_v  = np.std(arr,  axis=0)
                lw = 2.4 if col == 'current_account' else 1.7
                ax.plot(_years, mean_v, label=f"{SCENARIO_LABELS.get(scenario, scenario)} - {lbl}",
                        color=color, linestyle=ls, linewidth=lw)
                ax.fill_between(_years, mean_v - std_v, mean_v + std_v,
                                alpha=0.08, color=color)
                if gdp_share and y_for_share:
                    Y_mean = np.mean(np.array(y_for_share), axis=0)
                    _gdp_entries.append((color, mean_v, Y_mean))
        if zero_line:
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        if _years is not None:
            _ticks = [2021] + list(range(2025, _years[-1] + 1, 5))
            ax.set_xticks(_ticks)
            ax.set_xticklabels(_ticks, rotation=0, fontsize=16)
            
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=16)
            
        ax.legend(loc='lower left', fontsize=16, frameon=True)
        ax.grid(True, alpha=0.3)
        # GDP-share endpoint annotations (end-only; multiple variables per panel)
        if return_gdp_entries:
            return _years, _gdp_entries
        if gdp_share and _gdp_entries and _years:
            end_ann = []
            for color, mean_v, Y_mean in _gdp_entries:
                if Y_mean[-1] > 0:
                    end_ann.append({'y': mean_v[-1], 'real_y': mean_v[-1],
                                    'label': f"{mean_v[-1] / Y_mean[-1] * 100:.1f}%",
                                    'color': color})
            _add_endpoint_annotations(ax, [], end_ann, _years, caption_text=caption_text, caption_loc=caption_loc)
        return _years

    sns.set_style("whitegrid")

    # Figure 1: 2x2 main BoP panels  
    fig, axs = plt.subplots(2, 2, figsize=(20, 14))
    ((ax1, ax2), (ax3, ax4)) = axs

    # Panel 1: Current Account + Trade Balance   
    _plot_panel(ax1, reports,
                [('current_account', 'Current account'),
                 ('trade_balance',   'Trade balance')],
                title='Current Account & Trade Balance',
                scenario_colors=scenario_colors, ylabel="Trillion SAR", zero_line=True, legend_fontsize=16,
                gdp_share=True, caption_text='% of GDP Y (Tn SAR for 2060)', caption_loc='top_right')

    # Panel 2: Foreign Presence Flows - FDI capital & remittances   
    _p2_years, _p2_loop_gdp = _custom_loop(ax2, reports, [
        ('FDI_net_total', 'FDI net capital inflows (+)', '-',  1.0),
        ('remittances',   '\u2212Remittances (outflow)', '--', -1.0),
    ], zero_line=True, gdp_share=True, ylabel="", return_gdp_entries=True)
    _p2_manual_gdp = []  # (color, mean_v, Y_mean) for manually-plotted FDI profit repat.
    for s_idx, (scenario, runs) in enumerate(reports.items()):
        color = scenario_colors.get(scenario, default_colors_bop[s_idx % len(default_colors_bop)])
        data = []
        y_for_share = []
        for rr in runs:
            macro = rr["results_macro"]
            need = ['Gov_ext_assets_income', 'Bond_external_interest_cost', 'primary_income_balance']
            if not all(c in macro.columns for c in need):
                continue
            fdi_repat = (macro['Gov_ext_assets_income'].values
                         - macro['Bond_external_interest_cost'].values
                         - macro['primary_income_balance'].values)
            data.append(-fdi_repat / 1_000_000_000)
            if 'Y' in macro.columns:
                y_for_share.append(macro['Y'].values / 1_000_000_000)
        if not data:
            continue
        arr    = np.array(data)
        mean_v = np.mean(arr, axis=0)
        std_v  = np.std(arr,  axis=0)
        ax2.plot(_p2_years, mean_v,
                 label=f"{scenario} - \u2212FDI profit repat. (outflow)",
                 color=color, linestyle='-.', linewidth=1.7)
        ax2.fill_between(_p2_years, mean_v - std_v, mean_v + std_v,
                         alpha=0.06, color=color)
        if y_for_share:
            _p2_manual_gdp.append((color, mean_v, np.mean(np.array(y_for_share), axis=0)))
    # Build combined end-annotation list and call _add_endpoint_annotations once
    if _p2_years:
        _p2_all_end_ann = []
        for color, mean_v, Y_mean in (_p2_loop_gdp + _p2_manual_gdp):
            if Y_mean[-1] > 0:
                _p2_all_end_ann.append({'y': mean_v[-1], 'real_y': mean_v[-1],
                                        'label': f"{mean_v[-1] / Y_mean[-1] * 100:.1f}%",
                                        'color': color})
        if _p2_all_end_ann:
            _add_endpoint_annotations(ax2, [], _p2_all_end_ann, _p2_years)
    ax2.legend(loc='lower left', fontsize=16, frameon=True)
    ax2.set_title('Foreign Presence Flows: FDI Capital & Remittances', fontsize=14, fontweight='bold')

    # Panel 3: Government external wealth 
    _custom_loop(ax3, reports, [
        ('Gov_ext_assets', 'Gov. gross external assets (SAMA + PIF)', '-',   1.0),
        ('Bond_external', '\u2212External bonds (liability)',   '--', -1.0),
    ], zero_line=True, gdp_share=True, ylabel="Trillion SAR")
    ax3.set_title('Government External Wealth', fontsize=14, fontweight='bold')

    # Panel 4: Sovereign External Cash Flow - Sources & Uses 
    _p4_years, _p4_loop_gdp = _custom_loop(ax4, reports, [
        ('FDI_net_total', 'FDI net capital inflows (+)', '-',  1.0),
        ('Gov_ext_assets_income',    'Gov. ext. asset income (SAMA/PIF returns)', '--',  1.0),
    ], zero_line=True, gdp_share=True, ylabel="", return_gdp_entries=True)
    _p4_manual_gdp = []  # (color, mean_v, Y_mean) for manually-plotted agg. outflows
    for s_idx, (scenario, runs) in enumerate(reports.items()):
        color = scenario_colors.get(scenario, default_colors_bop[s_idx % len(default_colors_bop)])
        data = []
        y_for_share = []
        for rr in runs:
            macro = rr["results_macro"]
            need = ['Gov_ext_assets_income', 'primary_income_balance', 'remittances']
            if not all(c in macro.columns for c in need):
                continue
            agg_out = (macro['Gov_ext_assets_income'].values
                       - macro['primary_income_balance'].values
                       + macro['remittances'].values)
            data.append(-agg_out / 1_000_000_000)
            if 'Y' in macro.columns:
                y_for_share.append(macro['Y'].values / 1_000_000_000)
        if not data:
            continue
        arr    = np.array(data)
        mean_v = np.mean(arr, axis=0)
        std_v  = np.std(arr,  axis=0)
        ax4.plot(_p4_years, mean_v,
                 label=f"{scenario} - \u2212Agg. ext. outflows (FDI repat+bond int+remit)",
                 color=color, linestyle='-.', linewidth=1.7)
        ax4.fill_between(_p4_years, mean_v - std_v, mean_v + std_v,
                         alpha=0.06, color=color)
        if y_for_share:
            _p4_manual_gdp.append((color, mean_v, np.mean(np.array(y_for_share), axis=0)))
    # Build combined end-annotation list and call _add_endpoint_annotations once
    if _p4_years:
        _p4_all_end_ann = []
        for color, mean_v, Y_mean in (_p4_loop_gdp + _p4_manual_gdp):
            if Y_mean[-1] > 0:
                _p4_all_end_ann.append({'y': mean_v[-1], 'real_y': mean_v[-1],
                                        'label': f"{mean_v[-1] / Y_mean[-1] * 100:.1f}%",
                                        'color': color})
        if _p4_all_end_ann:
            _add_endpoint_annotations(ax4, [], _p4_all_end_ann, _p4_years)
    ax4.legend(loc='lower left', fontsize=16, frameon=True)
    ax4.set_title('Sovereign External Cash Flow: Sources & Uses', fontsize=14, fontweight='bold')

    for _i, _ax in enumerate(axs.flatten()):
        _ax.text(-0.05, 1.05, f"({chr(ord('a') + _i)})", transform=_ax.transAxes,
                 fontsize=16, fontweight='bold', va='top')
    plt.tight_layout()
    plt.show()

    # Figure 2 (standalone): BoP Flows Decomposition 
    fig5, ax5 = plt.subplots(figsize=(14, 8))
    sns.set_style("whitegrid")

    _p5_years = _custom_loop(ax5, reports, [
        ('current_account',                'Current account (aggregate)', '-',  1.0),
        ('Bond_external_routine_issuance', 'Bond financing (+deficit/\u2212surplus)', '-.', 1.0),
        ('trade_balance',                  'Trade balance (goods & services)', ':', 1.0),
    ], zero_line=True, gdp_share=True, ylabel="Trillion SAR", caption_text='% of GDP Y (Tn SAR for 2060)', caption_loc='top_right')
    for s_idx, (scenario, runs) in enumerate(reports.items()):
        color = scenario_colors.get(scenario, default_colors_bop[s_idx % len(default_colors_bop)])
        data = []
        for rr in runs:
            macro = rr["results_macro"]
            need = ['current_account', 'trade_balance']
            if not all(c in macro.columns for c in need):
                continue
            # Non-trade CA flows = primary income balance + secondary income balance
            # = (GEA_income  FDI profit repat.  bond ext. int.) + (remittances)
            # Computed cleanly as CA  trade_balance (avoids mixing in FA items)
            non_trade_ca = (macro['current_account'].values - macro['trade_balance'].values) / 1_000_000_000
            data.append(non_trade_ca)
        if not data:
            continue
        arr    = np.array(data)
        mean_v = np.mean(arr, axis=0)
        std_v  = np.std(arr,  axis=0)
        ax5.plot(_p5_years, mean_v,
                 label=f"{scenario} - Non-trade CA flows\n"
                       f"  (GEA returns \u2212 FDI profit repat. \u2212 ext. bond interest \u2212 remittances)",
                 color=color, linestyle='--', linewidth=1.7)
        ax5.fill_between(_p5_years, mean_v - std_v, mean_v + std_v,
                         alpha=0.06, color=color)
    ax5.legend(loc='lower left', fontsize=16, frameon=True)
    ax5.set_title('BoP Flows Decomposition', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Figure 3 (standalone): BoP-Relevant Stocks 
    fig6, ax6 = plt.subplots(figsize=(14, 8))
    sns.set_style("whitegrid")

    _custom_loop(ax6, reports, [
        ('Gov_ext_assets',       'Gov. gross external assets (SAMA + PIF)',     '-',  1.0),
        ('Bond_external',       '\u2212Ext. bonds (liability)',   '--', -1.0),
        ('FDI_net_stock_total', 'FDI net stock (foreign equity)', '-.', 1.0),
    ], zero_line=True, gdp_share=True, ylabel="Trillion SAR", caption_text='% of GDP Y (Tn SAR for 2060)', caption_loc='top_right')
    ax6.set_title('BoP-Relevant Stocks', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


 # 
# ORIGINAL SINGLE-VARIABLE EXPERIMENT PLOT
 # 

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_experiment_variable(reports, variable_name, variable_label=None, unit="", ax=None, show_legend=True, show_annotation_caption=False):
    """Plot experiment results showing different scenarios with mean and standard deviation."""

    if variable_label is None:
        variable_label = variable_name

    # 1. Define Consistent Color Mapping
    scenario_colors = {
        "Baseline":        "#d62728",  # Red
        "Vision_2030":     "#ff7f0e",  # Orange
        "Net_zero":        "#2ca02c",  # Green
        "Transformation":  "#1f77b4",  # Blue
        "Steady_state":    "#7f7f7f"   # Grey
    }
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Set seaborn style
    sns.set_style("whitegrid")

    is_standalone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        is_standalone = True

    for idx, (scenario, runs) in enumerate(reports.items()):
        data = []
        
        # Assign color based on the scenario name, or fallback to default
        color = scenario_colors.get(scenario, default_colors[idx % len(default_colors)])

        for run_report in runs:
            results_macro = run_report["results_macro"]
            var_data = results_macro[variable_name].values
            
            # --- CONVERSION LOGIC ---
            # Input is Thousand SAR. Output is Trillion SAR.
            # 1 Trillion = 1,000,000,000 Thousands.
            var_data = var_data / 1_000_000_000 
            # ------------------------
            
            data.append(var_data)

        endyear = run_report["endyear"]
        # Ensure 'years' matches the length of the data array
        years = list(range(2021, 2021 + len(data[0])))

        # Convert to numpy array for easier computation
        data = np.array(data)

        # Calculate mean and standard deviation across runs
        mean_values = np.mean(data, axis=0)
        std_values = np.std(data, axis=0)

        # Plot mean line with the specific color
        ax.plot(years, mean_values, label=SCENARIO_LABELS.get(scenario, scenario),
                color=color, linewidth=2.5)

        # Plot standard deviation as shaded area
        ax.fill_between(years,
                        mean_values - std_values,
                        mean_values + std_values,
                        alpha=0.2,
                        color=color)

        # --- 2060 DOT MARKER ---
        if 2060 in years:
            idx_2060 = years.index(2060)
            val_2060 = mean_values[idx_2060]
            ax.scatter([2060], [val_2060], color=color, s=40, zorder=5)

    # -- GDP-share endpoint annotations --
    # Pre-collect Vision 2030 start value for Y (used to show consistent start label for Baseline)
    _vision_y_start = None
    if variable_name == 'Y':
        for _s, _r in reports.items():
            if _s == 'Vision_2030':
                _vd = [_rr["results_macro"][variable_name].values / 1_000_000_000
                       for _rr in _r if variable_name in _rr["results_macro"].columns]
                if _vd:
                    _vision_y_start = float(np.mean(np.array(_vd), axis=0)[0])
                break

    _gdp_s_start, _gdp_s_end = [], []
    for _s_idx, (_scenario, _runs) in enumerate(reports.items()):
        _color = scenario_colors.get(_scenario, default_colors[_s_idx % len(default_colors)])
        _vdata, _ydata = [], []
        for _rr in _runs:
            _m = _rr["results_macro"]
            if variable_name not in _m.columns or 'Y' not in _m.columns:
                continue
            _vdata.append(_m[variable_name].values / 1_000_000_000)
            _ydata.append(_m['Y'].values / 1_000_000_000)
        if not _vdata or not _ydata:
            continue
        _mv = np.mean(np.array(_vdata), axis=0)
        _ym = np.mean(np.array(_ydata), axis=0)
        
        if variable_name != 'Y':
            # For standard variables, show % and Trillion value at both ends
            if _ym[0] > 0:
                _gdp_s_start.append({'y': _mv[0], 'real_y': _mv[0],
                                     'label': f"{_mv[0] / _ym[0] * 100:.1f}% ({_mv[0]:.1f}T)",
                                     'color': _color})
            if _ym[-1] > 0:
                _gdp_s_end.append({'y': _mv[-1], 'real_y': _mv[-1],
                                   'label': f"{_mv[-1] / _ym[-1] * 100:.1f}% ({_mv[-1]:.1f}T)",
                                   'color': _color})
        else:
            # For GDP (Y), add absolute Trillion values at both start and end
            # Use Vision 2030 start value for Baseline to ensure consistent start labels
            _start_t = (_vision_y_start if (_scenario == 'Baseline' and _vision_y_start is not None)
                        else _mv[0])
            _gdp_s_start.append({'y': _mv[0], 'real_y': _mv[0],
                                 'label': f"{_start_t:.1f}T",
                                 'color': _color})
            _gdp_s_end.append({'y': _mv[-1], 'real_y': _mv[-1],
                               'label': f"{_mv[-1]:.1f}T",
                               'color': _color})
            
    _add_endpoint_annotations(ax, _gdp_s_start, _gdp_s_end, years,
                               caption_text='' if variable_name == 'Y' else '% of GDP Y (Tn SAR for 2021 and 2060)')

    # Formatting
    ax.set_xlabel("Year", fontsize=16, fontweight='bold')
    
    # --- CUSTOM X-AXIS TICKS ---
    # Create a list starting with 2021, then every 5 years from 2025 onwards
    custom_ticks = [2021] + list(range(2025, years[-1] + 1, 5))
    ax.set_xticks(custom_ticks)
    # Changed fontsize to 14 for the x-ticks only
    ax.set_xticklabels(custom_ticks, rotation=0, fontsize=14)

    # --- REMOVE Y LABEL IF IT IS IN THE SECOND COLUMN OF A SUBPLOT GRID ---
    if ax is not None and hasattr(ax, 'get_subplotspec'):
        # Check if the axis is part of a multi-column subplot and not in the first column
        col_idx = ax.get_subplotspec().colspan.start
        
        # Hide unit for anything not in the first column
        if col_idx > 0:
            unit = ""
            
    if unit:
        ax.set_ylabel(unit, fontsize=16, fontweight='bold')
        
    ax.set_title(f"{variable_label}", fontsize=16, fontweight='bold')
    
    # Show legend based on the parameter passed into the function
    if show_legend:
        ax.legend(loc='best', fontsize=16, frameon=True, shadow=True)
        
    # Show annotation caption based on the parameter passed into the function
    # Only show caption if it's not the 'Y' plot, change color to black, remove "Y" from text.
    if show_annotation_caption and variable_name != 'Y':
        ax.text(0.01, 0.01, '% of GDP (Tn SAR for 2021 and 2060)',
                transform=ax.transAxes, fontsize=14, color='black',
                style='italic', va='bottom', ha='left')
        
    ax.grid(True, alpha=0.3)

    if is_standalone:
        plt.tight_layout()
        plt.show()
 # 
# FINANCIAL & PHYSICAL STOCKS SCENARIO COMPARISON  (3 x 2)
 # 

def plot_stocks_comparison(reports):
    """
    Four-panel scenario comparison of major financial stocks.

    Panels
    ------
    (a) Capital Stock (Nominal & Real) & Firm Loans
    (b) Household Total & Financial Wealth
    (c) Government Stocks (Bonds, GEA, Net Wealth)
    (d) FDI Stock (net) & GEA Gov Assets

    Color  = scenario  (consistent with other experiment_plots functions).
    Style  = variable within each panel (solid, dashed, dash-dot, dotted).
    Shaded = +/-1 std dev across stochastic runs.
    """
    scenario_colors = {
        "Baseline":        "#d62728",
        "Vision_2030":     "#ff7f0e",
        "Net_zero":        "#2ca02c",
        "Transformation":  "#1f77b4",
        "Steady_state":    "#7f7f7f",
    }
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    line_styles = ['-', '--', '-.', ':']

    def _extract(reports, col, scale=1.0):
        """Return {scenario: (years, mean_arr, std_arr)} for a given column."""
        out = {}
        for s_idx, (scenario, runs) in enumerate(reports.items()):
            data = []
            years = None
            for rr in runs:
                macro = rr["results_macro"]
                if col not in macro.columns:
                    continue
                vals = macro[col].values * scale
                data.append(vals)
                if years is None:
                    years = list(range(2021, 2021 + len(vals)))
            if not data:
                continue
            arr = np.array(data)
            out[scenario] = (years, np.mean(arr, axis=0), np.std(arr, axis=0))
        return out

    def _draw(ax, reports, variables, title, ylabel,
              zero_line=False, scale=1.0):
        """Draw one standard panel (monetary, scaled by `scale`)."""
        _gdp_entries = []
        for v_idx, (col, lbl) in enumerate(variables):
            series = _extract(reports, col, scale)
            y_series = _extract(reports, 'Y', 1 / 1_000_000_000)
            ls = line_styles[v_idx % len(line_styles)]
            for s_idx, (scenario, (years, mean_v, std_v)) in enumerate(series.items()):
                color = scenario_colors.get(
                    scenario, default_colors[s_idx % len(default_colors)])
                label = f"{SCENARIO_LABELS.get(scenario, scenario)} - {lbl}" if len(variables) > 1 else SCENARIO_LABELS.get(scenario, scenario)
                ax.plot(years, mean_v, label=label, color=color,
                        linestyle=ls, linewidth=1.8)
                ax.fill_between(years, mean_v - std_v, mean_v + std_v,
                                alpha=0.10, color=color)
                if scenario in y_series:
                    _, Y_mean, _ = y_series[scenario]
                    _gdp_entries.append((color, mean_v, Y_mean, years))
        if zero_line:
            ax.axhline(0, color='black', linewidth=0.7, linestyle='--')
        _fmt_ax(ax, title, ylabel)
        # GDP-share endpoint annotations (end-only for multi-variable panels)
        if _gdp_entries:
            n_scenarios = len(reports)
            show_start = len(_gdp_entries) <= n_scenarios
            start_ann, end_ann = [], []
            years_ref = _gdp_entries[0][3]
            for color, mean_v, Y_mean, _ in _gdp_entries:
                if Y_mean[0] > 0 and Y_mean[-1] > 0:
                    _abs_sfx_s = f" ({mean_v[0]:.1f}T)" if show_start else ''
                    _abs_sfx_e = f" ({mean_v[-1]:.1f}T)" if show_start else ''
                    if show_start:
                        start_ann.append({'y': mean_v[0], 'real_y': mean_v[0],
                                          'label': f"{mean_v[0] / Y_mean[0] * 100:.1f}%{_abs_sfx_s}",
                                          'color': color})
                    end_ann.append({'y': mean_v[-1], 'real_y': mean_v[-1],
                                    'label': f"{mean_v[-1] / Y_mean[-1] * 100:.1f}%{_abs_sfx_e}",
                                    'color': color})
            _add_endpoint_annotations(ax, start_ann, end_ann, years_ref)

    def _fmt_ax(ax, title, ylabel):
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=16)
            
        ax.legend(loc='best', fontsize=16, frameon=True)
        ax.grid(True, alpha=0.3)
        # x-ticks
        handles, labels = ax.get_lines(), None
        if ax.lines:
            xdata = ax.lines[0].get_xdata()
            if len(xdata):
                x_end = int(xdata[-1])
                ticks = [2021] + list(range(2025, x_end + 1, 5))
                ax.set_xticks(ticks)
                ax.set_xticklabels(ticks, rotation=0, fontsize=16)

    sns.set_style("whitegrid")
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    ((ax1, ax2), (ax3, ax4)) = axs

    # Panel 1: Capital Stock & Firm Loans 
    loan_col = 'L' if any(
        'L' in rr["results_macro"].columns
        for runs in reports.values() for rr in runs
    ) else None
    p1_vars = [('K', 'Nominal Capital K'), ('k', 'Real Capital k')]
    if loan_col:
        p1_vars.append(('L', 'Firm Loans L'))
    _draw(ax1, reports, p1_vars,
          title='Capital Stock & Firm Loans',
          ylabel='Trillion SAR / Real Index',
          scale=1 / 1_000_000_000)

    # Panel 2: Household Total & Financial Wealth  
    _draw(ax2, reports,
          [('HH_total_wealth', 'HH Net Worth (V+D+E+OFA)'), ('V', 'HH Financial Wealth V')],
          title='Household Total & Financial Wealth',
          ylabel='',
          scale=1 / 1_000_000_000)

    # Panel 3: Government Wealth (from macro comparison) 
    _plot_panel(ax3, reports,
                [('Gov_net_wealth', 'Government Net Wealth'),
                 ('GEA_gov',        'GEA_gov (net int. invest. position)')],
                title='Stocks - Government Wealth',
                scenario_colors=scenario_colors, ylabel="Trillion SAR", zero_line=True,
                show_legend=True, legend_loc='best', legend_fontsize=16,
                gdp_share=True)

    # Panel 4: FDI & Gov. External Asset Stocks 
    _draw(ax4, reports,
          [('FDI_net_stock_total', 'FDI Stock'),
           ('Gov_ext_assets',     'Gov. External Assets')],
          title='FDI and Gov. Ext. Asset Stocks - Cumulative',
          ylabel='',
          scale=1 / 1_000_000_000)

    for _i, _ax in enumerate(axs.flatten()):
        _ax.text(-0.05, 1.05, f"({chr(ord('a') + _i)})", transform=_ax.transAxes,
                 fontsize=16, fontweight='bold', va='top')
    plt.tight_layout()
    plt.show()

    # Standalone figure: Government Stocks 
    _gov_panel_vars = [
        ('Gov_net_wealth', 'Gov. Net Wealth (GEA  Bonds)',       '-',   1.0),
        ('Gov_ext_assets', 'Gov. gross external assets (SAMA + PIF)', '--',  1.0),
        ('Bond_domestic',  'Dom. Bonds (liability)',             '-.',  -1.0),
        ('Bond_external',  'Ext. Bonds (liability)',             ':',   -1.0),
    ]
    # Narrower figure: no outside-legend space needed; legend moves inside.
    fig_gov, ax_gov = plt.subplots(figsize=(12, 8))
    sns.set_style("whitegrid")
    _gov_years_ref = None
    _gov_gdp_end = []
    for col, lbl, ls, sign in _gov_panel_vars:
        for s_idx, (scenario, runs) in enumerate(reports.items()):
            color = scenario_colors.get(scenario, default_colors[s_idx % len(default_colors)])
            data, years_tmp = [], None
            for rr in runs:
                macro = rr["results_macro"]
                if col not in macro.columns:
                    continue
                vals = macro[col].values * sign / 1_000_000_000
                data.append(vals)
                if years_tmp is None:
                    years_tmp = list(range(2021, 2021 + len(vals)))
            if not data:
                continue
            arr    = np.array(data)
            mean_v = np.mean(arr, axis=0)
            std_v  = np.std(arr,  axis=0)
            ax_gov.plot(years_tmp, mean_v,
                        label=f"{SCENARIO_LABELS.get(scenario, scenario)} - {lbl}",
                        color=color, linestyle=ls, linewidth=1.8)
            ax_gov.fill_between(years_tmp, mean_v - std_v, mean_v + std_v,
                                alpha=0.08, color=color)
            # collect for GDP-share annotations
            _vd, _yd = [], []
            for rr in runs:
                macro = rr["results_macro"]
                if col not in macro.columns or 'Y' not in macro.columns:
                    continue
                _vd.append(macro[col].values * sign / 1_000_000_000)
                _yd.append(macro['Y'].values / 1_000_000_000)
                if _gov_years_ref is None:
                    _gov_years_ref = list(range(2021, 2021 + len(macro[col])))
            if _vd and _yd:
                _mv = np.mean(np.array(_vd), axis=0)
                _ym = np.mean(np.array(_yd), axis=0)
                if _ym[-1] > 0:
                    _gov_gdp_end.append({'y': _mv[-1], 'real_y': _mv[-1],
                                         'label': f"{_mv[-1] / _ym[-1] * 100:.1f}%",
                                         'color': color})
    ax_gov.axhline(0, color='black', linewidth=0.7, linestyle='--')
    ax_gov.set_title('Government Stocks', fontsize=14, fontweight='bold')
    ax_gov.set_ylabel('Trillion SAR', fontsize=16)
    ax_gov.grid(True, alpha=0.3)
    if ax_gov.lines:
        x_end = int(ax_gov.lines[0].get_xdata()[-1])
        ticks = [2021] + list(range(2025, x_end + 1, 5))
        ax_gov.set_xticks(ticks)
        ax_gov.set_xticklabels(ticks, rotation=0, fontsize=16)
    # Legend inside upper-left: 2-column layout keeps it compact and leaves the
    # right side of the plot clear for the % GDP endpoint annotations.
    ax_gov.legend(loc='upper left', ncol=2,
                  fontsize=13, frameon=True, framealpha=0.85,
                  borderaxespad=0.5, handlelength=1.8, columnspacing=1.0)
    if _gov_gdp_end and _gov_years_ref:
        _add_endpoint_annotations(ax_gov, [], _gov_gdp_end, _gov_years_ref)
    plt.tight_layout()
    plt.show()


 # 
# TOP SECTORS BY GDP Y - SCENARIO COMPARISON
 # 

 # 
# TOP SECTORS BY GDP Y - SCENARIO COMPARISON
 # 

def plot_sectoral_gdp_comparison(reports):
    """
    Show exactly 12 sectors in a 4x3 grid:
    - 6 fixed key sectors (petroleum, machinery, construction, real estate,
      public administration, education)
    - 3 extra sectors from Transformation (largest Y at finalal period)
    - 3 extra sectors from Transformation selected by a rank-sum of relative
      growth (Y_end/Y_start) and absolute growth (Y_end  Y_start), i.e. the
      sectors that grew the fastest AND reached meaningful GDP size - the type
      of structural transformation driven by Vision 2030 / industrial policy
      (e.g. ICT, professional services, tourism-related).

    Each scenario line is annotated with the sector's GDP share (%) at the
    start and end of the simulation period, placed to the left and right of the
    plot area respectively (overlap-resolved, similar to the energy graphs).

    Color  = scenario.
    Shaded = +/-1 std dev across stochastic runs.
    """
    scenario_colors = {
        "Baseline":        "#d62728",
        "Vision_2030":     "#ff7f0e",
        "Net_zero":        "#2ca02c",
        "Transformation":  "#1f77b4",
        "Steady_state":    "#7f7f7f",
    }
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # 1. Build full sector-name lookup from first available run 
    all_sector_names = {}
    for runs in reports.values():
        if runs and "parameters_calibrated" in runs[0]:
            pc_tmp = runs[0]["parameters_calibrated"]
            all_sector_names = {i: pc_tmp.sectors[i] for i in range(len(pc_tmp.sectors))}
            break

    # 2. Resolve 6 fixed sectors by case-insensitive substring match 
    FIXED_SUBSTRINGS = [
        'petroleum',
        'machinery',
        'construction',
        'real estate',
        'public administration',
        'education',
    ]
    fixed_indices = []
    for substr in FIXED_SUBSTRINGS:
        for idx, name in all_sector_names.items():
            if substr.lower() in name.lower() and idx not in fixed_indices:
                fixed_indices.append(idx)
                break

    # 3. Get Transformation scenario data for extra-sector selection 
    sus_key = next((k for k in reports if 'transformation' in k.lower()), None)

    # 3a. 3 extra sectors by largest Y at final period (existing logic) 
    extra_by_size = []
    if sus_key and reports[sus_key]:
        rr0 = reports[sus_key][0]
        Y_df = rr0["results_sectoral"]["Y"]
        end_period = Y_df.index[-1]
        for idx in Y_df.loc[end_period].nlargest(30).index.tolist():
            if idx not in fixed_indices:
                extra_by_size.append(idx)
            if len(extra_by_size) == 3:
                break

    # 3b. 3 extra sectors by combined growth score (rank-sum method) 
    # Selects sectors that grew the most on BOTH relative and absolute terms,
    # filtering out sectors that are trivially small at the start of the period.
    extra_by_growth = []
    if sus_key and reports[sus_key]:
        rr0 = reports[sus_key][0]
        Y_df = rr0["results_sectoral"]["Y"]
        Y_start = Y_df.iloc[0]   # first period values (thousand SAR)
        Y_end   = Y_df.iloc[-1]  # last  period values (thousand SAR)

        # Minimum threshold: sector must be at least 10% of mean sector size
        # at the start to exclude near-zero/placeholder sectors
        min_y_start = Y_start[Y_start > 0].mean() * 0.10

        already_selected = set(fixed_indices + extra_by_size)
        candidates = {}
        for idx in Y_df.columns:
            if idx in already_selected:
                continue
            ys = Y_start.get(idx, 0)
            ye = Y_end.get(idx, 0)
            if ys <= min_y_start or ys <= 0 or ye <= 0:
                continue
            candidates[idx] = {'rel': ye / ys, 'abs': ye - ys}

        if candidates:
            # Rank by relative growth (descending) - lower rank number = better
            rel_sorted = sorted(candidates, key=lambda x: candidates[x]['rel'], reverse=True)
            abs_sorted = sorted(candidates, key=lambda x: candidates[x]['abs'], reverse=True)
            rel_rank = {idx: i for i, idx in enumerate(rel_sorted)}
            abs_rank = {idx: i for i, idx in enumerate(abs_sorted)}
            combined_rank = {idx: rel_rank[idx] + abs_rank[idx] for idx in candidates}
            _ranked_all = sorted(combined_rank, key=combined_rank.get)  # full sorted list

            MIN_VIABLE_RATIO = 0.05  # sector must stay above 5% of its base-year value in every run
            extra_by_growth = []
            for _cidx in _ranked_all:
                if len(extra_by_growth) == 3:
                    break
                _stable = True
                for _scenario, _runs in reports.items():
                    for _rr in _runs:
                        _Y_df = _rr["results_sectoral"]["Y"]
                        if _cidx not in _Y_df.columns:
                            continue
                        _vals = _Y_df[_cidx].values
                        _base = abs(_vals[0]) if _vals[0] != 0 else 1.0
                        if np.any(_vals < _base * MIN_VIABLE_RATIO):
                            _stable = False
                            break
                    if not _stable:
                        break
                if _stable:
                    extra_by_growth.append(_cidx)

    display_indices = fixed_indices + extra_by_size + extra_by_growth  # 12 total

    # 4. Build human-readable names for each displayed sector  
    sector_names = {
        idx: all_sector_names.get(idx, f"Sector {idx}")
        for idx in display_indices
    }

    # 5. Shorten long names for subplot titles  
    def _shorten(name):
        replacements = [
            ('and natural gas', '& gas'),
            ('Crude petroleum', 'Petroleum'),
            ('Manufacture of', 'Mfg.'),
            ('manufacturing', 'mfg.'),
            ('Activities of', ''),
            ('activities', ''),
            ('administration and defence; compulsory social security', 'admin.'),
            ('Public administration', 'Public admin.'),
            ('Real estate', 'Real estate'),
        ]
        s = name
        for old, new in replacements:
            s = s.replace(old, new)
        return s.strip()[:40]

    # 6. Overlap resolver for GDP-share annotations   
    def _resolve_overlaps(points, min_gap):
        """Nudge overlapping label positions upward so they don't collide."""
        if not points:
            return []
        points = sorted(points, key=lambda p: p['y'])
        for i in range(1, len(points)):
            if points[i]['y'] - points[i - 1]['y'] < min_gap:
                points[i]['y'] = points[i - 1]['y'] + min_gap
        return points

    # 7. Build 4x3 figure 
    N = len(display_indices)   # 12
    ncols = 3
    nrows = 4
    fig, axs = plt.subplots(nrows, ncols, figsize=(18, nrows * 5.2))
    axs_flat = np.array(axs).flatten()

    # 8. Draw one subplot per sector 
    for plot_i, sector_idx in enumerate(display_indices):
        ax = axs_flat[plot_i]
        title_str = _shorten(sector_names[sector_idx])

        start_labels = []  # GDP-share annotations at t=0
        end_labels   = []  # GDP-share annotations at t=end
        years_ref    = None

        for s_idx, (scenario, runs) in enumerate(reports.items()):
            color = scenario_colors.get(
                scenario, default_colors[s_idx % len(default_colors)])

            data         = []
            total_y_data = []
            years        = None

            for rr in runs:
                Y_df = rr["results_sectoral"]["Y"]
                if sector_idx not in Y_df.columns:
                    continue
                vals = Y_df[sector_idx].values / 1_000_000_000  # Thousand   Trillion SAR
                data.append(vals)
                if years is None:
                    years = list(range(2021, 2021 + len(vals)))
                # Total GDP for share denominator
                macro = rr.get("results_macro")
                if macro is not None and 'Y' in macro.columns:
                    total_y_data.append(macro['Y'].values / 1_000_000_000)
                else:
                    # Fallback: sum all sectoral Y
                    total_y_data.append(Y_df.sum(axis=1).values / 1_000_000_000)

            if not data or years is None:
                continue

            if years_ref is None:
                years_ref = years

            arr       = np.array(data)
            mean_v    = np.mean(arr, axis=0)
            std_v     = np.std(arr, axis=0)

            ax.plot(years, mean_v, label=SCENARIO_LABELS.get(scenario, scenario), color=color, linewidth=1.8)
            ax.fill_between(years, mean_v - std_v, mean_v + std_v,
                            alpha=0.12, color=color)

            # Compute GDP shares for start/end annotations 
            total_arr  = np.array(total_y_data)
            mean_total = np.mean(total_arr, axis=0)

            share_start = (mean_v[0]  / mean_total[0])  * 100 if mean_total[0]  > 0 else 0.0
            share_end   = (mean_v[-1] / mean_total[-1]) * 100 if mean_total[-1] > 0 else 0.0

            start_labels.append({
                'y': mean_v[0],  'real_y': mean_v[0],
                'label': f"{share_start:.1f}% ({mean_v[0]:.1f}T)", 'color': color
            })
            end_labels.append({
                'y': mean_v[-1], 'real_y': mean_v[-1],
                'label': f"{share_end:.1f}% ({mean_v[-1]:.1f}T)",  'color': color
            })

        # Place GDP-share annotations 
        if years_ref is not None and (start_labels or end_labels):
            all_vals = [p['real_y'] for p in start_labels + end_labels]
            y_range  = max(all_vals) - min(all_vals) if len(all_vals) > 1 else (max(all_vals) * 0.4 if all_vals else 1.0)
            if y_range <= 0:
                y_range = max(all_vals) * 0.2 if all_vals and max(all_vals) > 0 else 1.0
            min_gap = max(y_range * 0.07, 1e-6)

            for p_ann in _resolve_overlaps(start_labels, min_gap):
                needs_arrow = abs(p_ann['y'] - p_ann['real_y']) > min_gap * 0.3
                ax.annotate(
                    p_ann['label'],
                    xy=(years_ref[0], p_ann['real_y']),
                    xytext=(years_ref[0] - 1.8, p_ann['y']),
                    color=p_ann['color'], fontsize=16, fontweight='bold',
                    ha='right', va='center',
                    arrowprops=dict(arrowstyle="-", color=p_ann['color'],
                                   lw=0.7, alpha=0.5) if needs_arrow else None,
                )

            for p_ann in _resolve_overlaps(end_labels, min_gap):
                needs_arrow = abs(p_ann['y'] - p_ann['real_y']) > min_gap * 0.3
                ax.annotate(
                    p_ann['label'],
                    xy=(years_ref[-1], p_ann['real_y']),
                    xytext=(years_ref[-1] + 1.8, p_ann['y']),
                    color=p_ann['color'], fontsize=16, fontweight='bold',
                    ha='left', va='center',
                    arrowprops=dict(arrowstyle="-", color=p_ann['color'],
                                   lw=0.7, alpha=0.5) if needs_arrow else None,
                )

            # Extend x-axis to make room for left/right annotations
            ax.set_xlim(years_ref[0] - 4, years_ref[-1] + 5)
            
            # Only draw the text if it is the first plot in the loop
            if plot_i == 0:
                ax.text(0.01, 0.01, '% of GDP Y (Tn SAR for 2021 and 2060)',
                        transform=ax.transAxes, fontsize=14, color='black',
                        style='italic', va='bottom', ha='left')

        ax.text(-0.05, 1.05, f"({chr(ord('a') + plot_i)})", transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top')
        ax.set_title(title_str, fontsize=12, fontweight='bold', pad=4)
        
        # REMOVE Y LABEL FOR ALL EXCEPT THE FIRST COLUMN
        if plot_i % ncols == 0:
            ax.set_ylabel('Trillion SAR', fontsize=16)
            
        ax.grid(True, alpha=0.3)

        # x-ticks (Custom explicit dates)
        if years_ref is not None:
            ticks = [2021, 2030, 2040, 2050, 2060]
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticks, rotation=0, fontsize=16)
        elif ax.lines:
            xdata = ax.lines[0].get_xdata()
            if len(xdata):
                ticks = [2021, 2030, 2040, 2050, 2060]
                ax.set_xticks(ticks)
                ax.set_xticklabels(ticks, rotation=0, fontsize=16)

        # Place legend ONLY in the first graph, move it inside (top left), and increase font size to 16.
        if plot_i == 0:
            ax.legend(
                loc='upper left',
                fontsize=16,
                frameon=True,
                framealpha=0.85,
                edgecolor='#cccccc',
            )

    # 9. Tag panels with descriptive x-labels   
    for j in range(len(extra_by_size)):
        axs_flat[len(fixed_indices) + j].set_xlabel(
            '(Transformation - top sector by final GDP size)',
            fontsize=9, fontstyle='italic', color='grey')

    for j in range(len(extra_by_growth)):
        axs_flat[len(fixed_indices) + len(extra_by_size) + j].set_xlabel(
            '(Transformation - high relative & absolute growth sector)',
            fontsize=9, fontstyle='italic', color='#1f77b4')

    # Hide any unused subplots (should be none for a full 4x3)
    for j in range(N, len(axs_flat)):
        axs_flat[j].set_visible(False)

    plt.tight_layout()
    plt.show()
 # 
# INFLATION & PRICE LEVEL SCENARIO COMPARISON  (2 x 2)
 # 

def plot_inflation_comparison(reports):
    """
    Four-panel scenario comparison of aggregate inflation and price levels.

    Panels
    ------
    (top-left)    Annual GDP-deflator inflation rate (%)
    (top-right)   Annual consumer price inflation rate (%)
    (bottom-left) Cumulative GDP deflator price index (base year = 100)
    (bottom-right)Cumulative consumer price index CPI (base year = 100)

    Color  = scenario (consistent palette across all experiment_plots functions).
    Shaded = +/-1 std dev across stochastic runs.

    Note: inflation variables are stored as decimals (e.g. 0.02 = 2 %); the
    cumulative index variables start at 1.0 in the base year.  Both are
    multiplied by 100 so the plots show percentages / index-base-100.
    """
    scenario_colors = {
        "Baseline":        "#d62728",
        "Vision_2030":     "#ff7f0e",
        "Net_zero":        "#2ca02c",
        "Transformation":  "#1f77b4",
        "Steady_state":    "#7f7f7f",
    }

    sns.set_style("whitegrid")
    fig, axs = plt.subplots(2, 2, figsize=(16, 11))
    ((ax1, ax2), (ax3, ax4)) = axs

    # Annual inflation rates (%)   
    _plot_panel_raw(
        ax1, reports, col='inflation',
        title='GDP Deflator - Annual Inflation Rate',
        ylabel='Inflation Rate (%)',
        scenario_colors=scenario_colors,
        scale=100,
        zero_line=True,
    )

    _plot_panel_raw(
        ax2, reports, col='inflation_consumers',
        title='CPI - Annual Consumer Price Inflation Rate',
        ylabel='',
        scenario_colors=scenario_colors,
        scale=100,
        zero_line=True,
    )

    # Cumulative price indices (base year = 100) 
    _plot_panel_raw(
        ax3, reports, col='deflator_gdp',
        title='Cumulative GDP Deflator Price Index (base year = 100)',
        ylabel='Price Index (base = 100)',
        scenario_colors=scenario_colors,
        scale=100,
        reference_line=100,
    )

    _plot_panel_raw(
        ax4, reports, col='price_index_cons',
        title='Cumulative Consumer Price Index CPI (base year = 100)',
        ylabel='',
        scenario_colors=scenario_colors,
        scale=100,
        reference_line=100,
    )

    for _i, _ax in enumerate(axs.flatten()):
        _ax.text(-0.05, 1.05, f"({chr(ord('a') + _i)})", transform=_ax.transAxes,
                 fontsize=16, fontweight='bold', va='top')
    plt.tight_layout()
    plt.show()


 # 
# MOST DYNAMIC SECTORS - SECTORAL PRICE DEVELOPMENTS  (3 x 3)
 # 

def plot_sectoral_prices_comparison(reports, n_sectors=9):
    """
    3x3 scenario comparison of sectoral price-index developments for the
    n_sectors most price-dynamic sectors.

    Dynamism is measured as the mean absolute change from the base year to the
    final period  (|p_final  p_initial|), averaged over all scenarios and runs.
    This identifies sectors whose prices deviate most from baseline regardless
    of direction (rapid appreciation OR deflation both count as dynamic).

    Axes
    ----
    Color  = scenario.
    Grey dashed horizontal = 1.0 (base-year price level).
    Shaded = +/-1 std dev across stochastic runs.
    """
    scenario_colors = {
        "Baseline":        "#d62728",
        "Vision_2030":     "#ff7f0e",
        "Net_zero":        "#2ca02c",
        "Transformation":  "#1f77b4",
        "Steady_state":    "#7f7f7f",
    }
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # 1. Build sector-name lookup from first available run 
    all_sector_names: dict = {}
    for runs in reports.values():
        if runs and "parameters_calibrated" in runs[0]:
            pc_tmp = runs[0]["parameters_calibrated"]
            all_sector_names = {i: pc_tmp.sectors[i] for i in range(len(pc_tmp.sectors))}
            break

    # 2. Determine sector count from data   
    n_all = 0
    for runs in reports.values():
        for rr in runs:
            p_df = rr["results_sectoral"].get("p")
            if p_df is not None:
                n_all = len(p_df.columns)
                break
        if n_all:
            break

    if n_all == 0:
        print("plot_sectoral_prices_comparison: no sectoral price data ('p') found.")
        return

    # 3. Rank sectors by mean absolute price change (base   final)  
    abs_changes = np.zeros(n_all)
    count = 0
    for runs in reports.values():
        for rr in runs:
            p_df = rr["results_sectoral"].get("p")
            if p_df is None:
                continue
            p_arr = p_df.values          # shape (T, S)
            abs_changes += np.abs(p_arr[-1] - p_arr[0])
            count += 1
    if count:
        abs_changes /= count

    ranked_indices = list(np.argsort(abs_changes)[::-1])   # descending

    # Filter to valid column indices (some columns might not be 0..n_all-1)
    valid_cols = None
    for runs in reports.values():
        for rr in runs:
            p_df = rr["results_sectoral"].get("p")
            if p_df is not None:
                valid_cols = list(p_df.columns)
                break
        if valid_cols is not None:
            break

    # ranked_indices are positional; map to actual column labels
    if valid_cols is not None:
        ranked_col_labels = [valid_cols[i] for i in ranked_indices if i < len(valid_cols)]
    else:
        ranked_col_labels = ranked_indices

    display_cols = ranked_col_labels[:n_sectors]

    # 4. Shorten long sector names  
    def _shorten(name):
        replacements = [
            ('and natural gas', '& gas'),
            ('Crude petroleum', 'Petroleum'),
            ('Manufacture of', 'Mfg.'),
            ('manufacturing', 'mfg.'),
            ('Activities of', ''),
            ('activities', ''),
            ('administration and defence; compulsory social security', 'admin.'),
            ('Public administration', 'Public admin.'),
            ('Real estate', 'Real estate'),
            ('collection, treatment and supply', 'supply'),
            ('Electricity, gas, steam and air conditioning supply', 'Electricity & gas'),
        ]
        s = name
        for old, new in replacements:
            s = s.replace(old, new)
        return s.strip()[:45]

    # 5. Build 3x3 figure 
    ncols = 3
    nrows = int(np.ceil(n_sectors / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(18, nrows * 5.0))
    axs_flat = np.array(axs).flatten()

    for plot_i, col_label in enumerate(display_cols):
        ax = axs_flat[plot_i]

        # Sector name: col_label is the column name in the DataFrame (int index)
        sector_name = all_sector_names.get(col_label, f"Sector {col_label}")
        title_str = _shorten(sector_name)

        # Rank annotation
        rank_label = f"#{plot_i + 1} most dynamic"

        for s_idx, (scenario, runs) in enumerate(reports.items()):
            color = scenario_colors.get(
                scenario, default_colors[s_idx % len(default_colors)])

            data = []
            years = None
            for rr in runs:
                p_df = rr["results_sectoral"].get("p")
                if p_df is None or col_label not in p_df.columns:
                    continue
                vals = p_df[col_label].values   # price index, base
                data.append(vals)
                if years is None:
                    years = list(range(2021, 2021 + len(vals)))

            if not data or years is None:
                continue

            arr = np.array(data)
            mean_v = np.mean(arr, axis=0)
            std_v  = np.std(arr, axis=0)

            ax.plot(years, mean_v, label=SCENARIO_LABELS.get(scenario, scenario), color=color, linewidth=1.8)
            ax.fill_between(years, mean_v - std_v, mean_v + std_v,
                            alpha=0.12, color=color)

        # Base-year reference line at p = 1.0
        ax.axhline(y=1.0, color='grey', linewidth=0.9, linestyle='--',
                   label='Base year (p = 1)')

        ax.text(-0.05, 1.05, f"({chr(ord('a') + plot_i)})", transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top')
        ax.set_title(title_str, fontsize=12, fontweight='bold', pad=4)
        ax.set_xlabel(rank_label, fontsize=10, fontstyle='italic', color='dimgrey')
        
        # REMOVE Y LABEL FOR ALL EXCEPT THE FIRST COLUMN
        if plot_i % ncols == 0:
            ax.set_ylabel('Price Index (base year = 1)', fontsize=16)
            
        ax.grid(True, alpha=0.3)

        # x-ticks: every 10 years + first year; rotated to prevent overlap
        if ax.lines:
            xdata = ax.lines[0].get_xdata()
            if len(xdata):
                x_end = int(xdata[-1])
                ticks = [2021] + list(range(2030, x_end + 1, 10))
                ax.set_xticks(ticks)
                ax.set_xticklabels(ticks, rotation=45, ha='right', fontsize=12)

        # Legend inside the axes at the best non-overlapping position
        ax.legend(
            loc='best',
            fontsize=11,
            frameon=True,
            framealpha=0.85,
            edgecolor='#cccccc',
        )

    # Hide unused subplots
    for j in range(n_sectors, len(axs_flat)):
        axs_flat[j].set_visible(False)

    plt.tight_layout()
    plt.show()

# plot_material_flows_comparison and plot_material_stocks_comparison removed for public release
