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

import numpy as np
import pandas as pd
from IPython.display import display as IPdisplay
from .utils import string_to_value


# ── LaTeX symbol name map ────────────────────────────────────────────────────
# Maps internal shorthand names (as used in the +/- symbol strings) to their
# LaTeX math representations.  Used by `to_latex_symbols()` before export.
_LATEX_SYMBOL_MAP = {
    # Balance sheet stocks
    'K':                r'$K$',
    'V':                r'$V$',
    'L':                r'$L$',
    'Bond_dom':         r'$B^{dom}$',
    'Bond_ext':         r'$B^{ext}$',
    'Gov_ext_assets':   r'$A^{ext}_{gov}$',
    'FDI_stock':        r'$FDI_{stock}$',
    # Transaction flow variables
    'C':                r'$C$',
    'VAT':              r'$VAT$',
    'GY':               r'$GY$',
    'Sub':              r'$Sub$',
    'GI':               r'$GI$',
    'EX':               r'$EX$',
    'IM':               r'$IM$',
    'W':                r'$W$',
    'GP':               r'$GP$',
    'PB':               r'$PB$',
    'TH':               r'$TH$',
    'Tax_profits':      r'$Tax_{profits}$',
    'Tax_wages':        r'$Tax_{wages}$',
    'IOT_taxes':        r'$IOT_{taxes}$',
    'P_private':        r'$P_{private}$',
    'rm_V':             r'$r_m V$',
    'PB_HH_rm_V':       r'$PB_{HH}{+}r_m V$',
    'PB_rm_V':          r'$PB{+}r_m V$',
    'rl_L':             r'$r_l L$',
    'rho_L':            r'$\rho L$',
    'dom_bond_service': r'$i^{dom} B^{dom}$',
    'ext_bond_service': r'$i^{ext} B^{ext}$',
    'gov_exp_resid':    r'$Gov_{resid}$',
    'I_private':        r'$I_{private}$',
    'invest_private':   r'$I_{private}$',      # kept for backward compat
    'retained_earnings':  r'$\Pi_{retained}$',
    'distributed_profits': r'$\Pi_{dist}$',
    'new_loans':        r'$\Delta L^{+}$',
    'DeltaV':           r'$\Delta V$',
    'DeltaBond_dom':    r'$\Delta B^{dom}$',
    'DeltaBond_ext':    r'$\Delta B^{ext}$',
    'FDI_net':          r'$FDI_{net}$',
    'Gov_pif':          r'$PIF_{inv}$',
    'prod_adj_realized': r'$PA_{realized}$',
    'prod_adj_demand':   r'$PA_{demand}$',
    'FDI_income':       r'$FDI_{income}$',
    'remittances':      r'$Rem$',
    'Gov_ext_assets_income': r'$r_{gea} A^{ext}_{gov}$',
    'DeltaGov_ext_assets':  r'$\Delta A^{ext}_{gov}$',
    'PB_HH':            r'$PB_{HH}$',
    'PB_gov':           r'$PB_{gov}$',
    # Equity residuals and HH deposits
    'D_HH':             r'$D_{HH}$',
    'E_firm':           r'$E_{firm}$',
    'E_firm_HH':        r'$E_{firm,HH}$',
    'E_firm_gov':       r'$E_{firm,gov}$',
    'E_bank':           r'$E_{bank}$',
    'E_bank_HH':        r'$E_{bank,HH}$',
    'E_bank_gov':       r'$E_{bank,gov}$',
    'DeltaD_HH':        r'$\Delta D_{HH}$',
    'DeltaE_firm':      r'$\Delta E_{firm}$',
    'DeltaE_firm_HH':   r'$\Delta E_{firm,HH}$',
    'DeltaE_firm_gov':  r'$\Delta E_{firm,gov}$',
    'StatDisc':         r'$\varepsilon$',
    # OFA flows (TFM) and stocks (BSM)
    'DOFA_HH':    r'$\Delta OFA_{HH}$',
    'DOFA_FC':    r'$\Delta OFA_{FC}$',
    'DOFA_FK':    r'$\Delta OFA_{FK}$',
    'DOFA_Banks': r'$\Delta OFA_{Banks}$',
    'DOFA_Gov':   r'$\Delta OFA_{Gov}$',
    'DOFA_ROW':   r'$\Delta OFA_{ROW}$',
    'OFA_HH':     r'$OFA_{HH}$',
    'OFA_Firms':  r'$OFA_{Firms}$',
    'OFA_Banks':  r'$OFA_{Banks}$',
    'OFA_Gov':    r'$OFA_{Gov}$',
    'OFA_ROW':    r'$OFA_{ROW}$',
    # Generic / row labels
    'NW':       r'$NW$',
    'DeltaNW':  r'$\Delta NW$',
    'NLNB': r'$NLNB$',
    'S/D':  r'$S/D$',
    '0':    r'$0$',
}

# Row index label map: display label → LaTeX label
_LATEX_ROW_MAP = {
    # Balance sheet rows
    "Capital (K)":          r"Capital ($K$)",
    "HH Portfolio Claims (V)": r"HH Portfolio Claims ($V$)",
    "HH Deposits (D_HH)": r"HH Deposits ($D_{HH} = L$, endogenous money)",
    "Private Firm Equity — HH (E_f,HH)":  r"Firm Equity HH ($E_{firm,HH}$)",
    "Private Firm Equity — Gov (E_f,Gov)": r"Firm Equity Gov ($E_{firm,gov}$)",
    "Bank Equity (E_b)":    r"Bank Equity ($E_{bank} = B^{dom}{-}V$)",
    "Loans (L)":            r"Loans ($L$)",
    "Dom Gov Bonds":        r"Dom.\ Gov.\ Bonds ($B^{dom}$)",
    "Ext Gov Bonds":        r"Ext.\ Gov.\ Bonds ($B^{ext}$)",
    "Gov Ext Assets":        r"Gov.\ Gross External Assets ($A^{ext}_{gov}$)",
    "FDI Net Stock":        r"FDI Net Stock ($FDI_{stock}$)",
    "Sum: Net worth (NW)":  r"Sum: Net worth ($NW$)",
    # Transaction flow rows
    "Household Tax (TH)":           r"Household Tax ($TH$)",
    "Tax on Profits":               r"Tax on Profits ($Tax_{profits}$)",
    "Tax on Wages":                 r"Tax on Wages ($Tax_{wages}$)",
    "IOT Taxes":                    r"IOT Taxes ($IOT_{taxes}$)",
    "i on Loans":                   r"$i$ on Loans ($r_l L$)",
    "Bank Profits + return to HH portfolio claims": r"Bank Profits + return to HH portfolio claims ($PB_{HH}{+}r_m V{\to}HH$, $PB_{gov}{\to}Gov$)",
    "Dom Bond Service (i+rho)":     r"Dom.\ Bond Service ($i^{dom}{+}\rho$)",
    "Ext Bond Service (i+rho)":     r"Ext.\ Bond Service ($i^{ext}{+}\rho$)",
    "Gov Exp Residual":             r"Gov.\ Exp.\ Residual ($Gov_{resid}$)",
    "Priv. Investment (dom. financed)": r"Priv.\ Investment (dom.\ financed) ($I^{dom}_{private}$)",
    "Distributed Profits":          r"Distributed Profits ($\Pi_{dist}$)",
    "Retained Earnings":            r"Retained Earnings ($\Pi_{retained}$)",
    "Production Adj. realized":     r"Production Adj.\ realized",
    "Production Adj. demand":       r"Production Adj.\ demand",
    "New Loans":                    r"New Loans ($\Delta L^{+}$)",
    "Loan Repayment":               r"Loan Repayment ($\rho L$)",
    "Change in Household Wealth (V)": r"Change in HH Wealth ($\Delta V$)",
    "Change in Dom. Bonds":         r"Change in Dom.\ Bonds ($\Delta B^{dom}$)",
    "Change in Ext. Bonds":         r"Change in Ext.\ Bonds ($\Delta B^{ext}$)",
    "Priv. Investment (FDI from RoW)": r"Priv.\ Investment (FDI from RoW) ($FDI_{net}$)",
    "Gov PIF Investment":           r"Gov.\ PIF Investment ($PIF_{inv}$)",
    "FDI Income Repatriation":      r"FDI Income Repatriation ($FDI_{income}$)",
    "Remittances":                  r"Remittances ($Rem$)",
    "Gov Ext Asset Income":          r"Gov.\ Ext.\ Asset Income ($r_{gea} A^{ext}_{gov}$)",
    "Change in Gov Ext Assets":      r"Change in Gov.\ Ext.\ Assets ($\Delta A^{ext}_{gov}$)",
    "Change in HH Deposits (\u0394D_HH)": r"Change in HH Deposits ($\Delta D_{HH} = \Delta L$, endogenous money)",
    "Change in Firm Equity — HH (\u0394E_f,HH) [incl. revaluation]": r"Change in Firm Equity — HH ($\Delta E_{firm,HH}$, incl.\ revaluation)",
    "Change in Firm Equity — Gov (\u0394E_f,Gov) [incl. revaluation]": r"Change in Firm Equity — Gov ($\Delta E_{firm,gov}$, incl.\ revaluation)",
    "Statistical Discrepancy":      r"Statistical Discrepancy ($\varepsilon = S/D + NLNB$)",
    "Change in Other Financial Assets (OFA) (\u0394OFA)": r"Change in Other Financial Assets (OFA) ($\Delta OFA$)",
    "Other Financial Assets (OFA) Net Stock": r"Other Financial Assets ($OFA$)",
    "Surplus/Deficit (S/D)":        r"Surplus ($+$)/Deficit ($-$) (S/D)",
    "Net Lending/Borrowing (NLNB) [\u2212 net lending, + net borrowing]": r"Net Lending/Borrowing ($NLNB$) [$-$ net lending, $+$ net borrowing]",
}


def _cell_to_latex(cell) -> str:
    """Convert a +/- symbol cell string to a LaTeX-formatted string."""
    if not isinstance(cell, str):
        return ''
    if cell == '' or cell == '0':
        return cell
    sign = ''
    name = cell
    if cell.startswith('+') or cell.startswith('-'):
        sign = cell[0]
        name = cell[1:]
    latex_name = _LATEX_SYMBOL_MAP.get(name, f'${name}$')
    return f'{sign}{latex_name}'


def _row_label_to_latex(label: str) -> str:
    """Return LaTeX-formatted row index label."""
    return _LATEX_ROW_MAP.get(label, label)


def to_latex_symbols(df_sym: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of a symbols DataFrame with all cell values and the row index
    converted to LaTeX-safe representations, ready for .to_latex(escape=False).
    """
    df = df_sym.copy()
    # Convert cell contents
    df = df.map(_cell_to_latex)
    # Convert row index labels
    df.index = [_row_label_to_latex(lbl) for lbl in df.index]
    return df


def _compute_sector_ofa(report: dict) -> pd.DataFrame:
    """
    Compute cumulative OFA net stocks per BSM sector (5 columns) for all time periods.

    OFA_i(t) = Σ_{s=1}^{t} ΔOFA_i(s),  where ΔOFA_i(s) = -(S/D_i(s) + NLNB_explicit_i(s)).

    This is the residual financial instrument that guarantees NLNB = −S/D for every sector
    in every period, encoding all unmodelled financial complexity (trade credit, interbank
    claims, pension fund shares, repo, etc.) as a single clearing variable.
    TFM Firms (Current) + Firms (Capital) are summed into BSM 'Firms' column.

    Results are cached in report['_ofa_stocks'] to avoid repeated computation.
    """
    if '_ofa_stocks' in report:
        return report['_ofa_stocks']

    macro = report['results_macro']
    bsm_sectors = ['Households', 'Firms', 'Banks', 'Government/CB', 'ROW']
    _OFA_ROW_LABEL = 'Change in Other Financial Assets (OFA) (\u0394OFA)'
    _tfm_to_bsm = {
        'Households':      'Households',
        'Firms (Current)': 'Firms',
        'Firms (Capital)': 'Firms',
        'Banks':           'Banks',
        'Government':      'Government/CB',
        'ROW':             'ROW',
    }

    ofa_flow = pd.DataFrame(0.0, index=macro.index, columns=bsm_sectors)

    for t_idx in range(1, len(macro)):
        tm_t, _ = create_transition_matrix(report, t=t_idx, display=False)
        if _OFA_ROW_LABEL in tm_t.index:
            row = tm_t.loc[_OFA_ROW_LABEL]
            for tfm_col, bsm_col in _tfm_to_bsm.items():
                if tfm_col in row.index:
                    ofa_flow.loc[macro.index[t_idx], bsm_col] += row[tfm_col]

    # Cumulative stock; period 0 stays at zero (pre-simulation baseline).
    ofa_stock = ofa_flow.cumsum()
    report['_ofa_stocks'] = ofa_stock
    return ofa_stock


def create_balance_sheet(run_report: dict, t: int = -1, display: bool = True):
    """
    Create a balance sheet (stock matrix) for the SFC model.

    Agents: Households, Firms, Banks, Government/CB, ROW.
    Rows: Capital, Deposits, Loans, Dom Gov Bonds, Ext Gov Bonds,
          Gov Ext Assets, FDI Net Stock, Net Worth.

    All rows sum to zero (SFC convention), except Capital which sums to +K.
    ROW column carries the external counterpart of each open-economy stock.

    Parameters
    ----------
    run_report : dict
        A single run report from ExperimentRunner.
    t : int
        Time index into the macro DataFrame (default -1 = final period).
    display : bool
        Whether to display the DataFrames using IPython.display.

    Returns
    -------
    balance_sheet : pd.DataFrame
        Numeric balance sheet with values.
    balance_sheet_symbols : pd.DataFrame
        Symbolic balance sheet showing variable names.
    """
    macro = run_report["results_macro"]
    pc = run_report["parameters_calibrated"]
    dt = macro.iloc[t]

    # ── Equity residuals (computed from existing stocks, no new model variables) ──
    # Capital-weighted public ownership share (from calibration; constant over time).
    # Mirrors the profit split: GP = sum(s_public_ * P). Here applied to the equity stock.
    _s_pub_agg = np.sum(pc.s_public_ * pc.K_) / np.sum(pc.K_)

    # HH deposits = total loan stock (endogenous money: loans create deposits attributed to HH).
    # D_HH = sum(L_): HH hold deposits equal to total loans outstanding.
    _D_HH = dt['D_HH']

    # Private firm equity: E_f = K − L − FDI.
    # With HH deposits (D_HH = L) on the HH side, loans (L) remain as firm liabilities → equity = K − L − FDI.
    _E_firm     = dt['K'] - dt['L'] - dt['FDI_net_stock_total']
    _E_firm_HH  = (1.0 - _s_pub_agg) * _E_firm    # Private HH equity claim
    _E_firm_gov = _s_pub_agg           * _E_firm    # Gov equity in public firms (Aramco, utilities, desal)

    # Bank equity: E_b = B^dom − V.
    # D_HH = L: bank assets = L + B^dom; liabilities = D_HH + V = L + V → equity = B^dom − V (unchanged).
    # May be negative when household deposits exceed domestic bond holdings (undercapitalisation signal).
    _E_bank     = dt['Bond_domestic'] - dt['V']
    _E_bank_HH  = (1.0 - pc.gov_bank_share) * _E_bank
    _E_bank_gov = pc.gov_bank_share            * _E_bank

    # Build data dict for string_to_value
    data = {
        'K':           dt['K'],
        'V':           dt['V'],
        'L':           dt['L'],
        'D_HH':        _D_HH,
        'Bond_dom':    dt['Bond_domestic'],
        'Bond_ext':    dt['Bond_external'],
        'Gov_ext_assets': dt['Gov_ext_assets'],
        'FDI_stock':   dt['FDI_net_stock_total'],
        'E_firm':      _E_firm,
        'E_firm_HH':   _E_firm_HH,
        'E_firm_gov':  _E_firm_gov,
        'E_bank':      _E_bank,
        'E_bank_HH':   _E_bank_HH,
        'E_bank_gov':  _E_bank_gov,
    }

    # 5 columns including ROW
    columns = ["Households", "Firms", "Banks", "Government/CB", "ROW"]

    rows = [
        "Capital (K)",
        "HH Portfolio Claims (V)",  # V = HH liquid claims on banks; not purely cash deposits (no endogenous money)
        "HH Deposits (D_HH)",           # D_HH = L: HH hold deposits equal to total loans outstanding (endogenous money)
        "Loans (L)",
        "Dom Gov Bonds",
        "Ext Gov Bonds",
        "Gov Ext Assets",
        "FDI Net Stock",
        "Private Firm Equity — HH (E_f,HH)",   # Private HH equity claim = (1 − s_pub) × (K − L − FDI)
        "Private Firm Equity — Gov (E_f,Gov)",  # Gov equity in public firms = s_pub × (K − L − FDI)  (Aramco, utilities, desal)
        "Bank Equity (E_b)",                     # E_b = B^dom − V: bank sector net worth, split HH/Gov
    ]

    # Symbolic entries: [HH, Firms, Banks, Gov/CB, ROW]
    # ROW holds the mirror position for every open-economy stock so each row sums to 0.
    # Equity rows: HH holds positive equity; Firms/Banks have corresponding negative equity obligations.
    # Ext Gov Bonds: ROW is the creditor (+Bond_ext)
    # Gov Ext Assets: ROW is the net debtor (−Gov_ext_assets)
    # FDI Net Stock: ROW is the investor (+FDI_stock)
    # HH Deposits: HH asset (+D_HH), Banks liability (−D_HH) — asset counterpart of the Loans row
    symbols = [
        ["",            "+K",           "",           "",             ""          ],  # Capital (real)
        ["+V",          "",             "-V",         "",             ""          ],  # HH Portfolio Claims
        ["+D_HH",       "",             "-D_HH",      "",             ""          ],  # HH Deposits: HH asset / Banks liability
        ["",            "-L",           "+L",         "",             ""          ],  # Loans
        ["",            "",             "+Bond_dom",  "-Bond_dom",    ""          ],  # Dom Bonds
        ["",            "",             "",           "-Bond_ext",    "+Bond_ext" ],  # Ext Bonds
        [""  ,           "",             "",           "+Gov_ext_assets", "-Gov_ext_assets"],  # Gov Ext Assets
        ["",            "-FDI_stock",   "",           "",             "+FDI_stock"],  # FDI Net Stock
        ["+E_firm_HH",  "-E_firm_HH",   "",           "",             ""          ],  # Priv. Firm Equity HH share → row sums to 0
        ["",            "-E_firm_gov",  "",           "+E_firm_gov",  ""          ],  # Priv. Firm Equity Gov share (public firms) → row sums to 0
        ["+E_bank_HH",  "",             "-E_bank",    "+E_bank_gov",  ""          ],  # Bank Equity: HH (priv) + Gov (state) own banks
    ]

    # Sum column symbols (row sums — all financial rows = "0"; Capital = "+K")
    sum_symbols = ["+K", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"]

    # ── OFA net stocks (cumulative other financial claims per sector) ─────────
    # Σ OFA_i = 0 by construction (every ΔOFA flow sums to zero across sectors).
    ofa_stocks = _compute_sector_ofa(run_report)
    ofa_t = ofa_stocks.iloc[t]
    data['OFA_HH']    = ofa_t['Households']
    data['OFA_Firms'] = ofa_t['Firms']
    data['OFA_Banks'] = ofa_t['Banks']
    data['OFA_Gov']   = ofa_t['Government/CB']
    data['OFA_ROW']   = ofa_t['ROW']
    rows.append("Other Financial Assets (OFA) Net Stock")
    symbols.append(['OFA_HH', 'OFA_Firms', 'OFA_Banks', 'OFA_Gov', 'OFA_ROW'])
    sum_symbols.append('0')

    # Convert symbols to numeric values
    numeric = [[string_to_value(s, data) for s in row] for row in symbols]

    balance_sheet_symbols = pd.DataFrame(symbols, columns=columns, index=rows)
    balance_sheet = pd.DataFrame(numeric, columns=columns, index=rows)

    # Net worth row: column sums over all 5 agent columns
    balance_sheet.loc['Sum: Net worth (NW)'] = balance_sheet.sum(axis=0)

    # Sum column: row sums (should be 0 for financial rows, K for Capital)
    balance_sheet['Sum'] = balance_sheet[columns].sum(axis=1)

    # Symbolic Sum column and NW row
    balance_sheet_symbols['Sum'] = sum_symbols
    balance_sheet_symbols.loc['Sum: Net worth (NW)'] = ['NW', 'NW', 'NW', 'NW', 'NW', 'NW']

    if display:
        print("Balance Sheet (Symbols):")
        IPdisplay(balance_sheet_symbols)
        print(f"\nBalance Sheet (Values in billion SAR, t={t}):")
        IPdisplay((balance_sheet / 1e6).round(2))

    return balance_sheet, balance_sheet_symbols


def create_transition_matrix(run_report: dict, t: int = -1, display: bool = True):
    """
    Create a transition (flow) matrix for the aggregated SFC model.

    Shows all income, expenditure, and balance-sheet-change flows between aggregated agents.

    Each row should ideally sum to zero (every payment has a receiver).
    Column sums reveal the model's stock-flow consistency.

    Parameters
    ----------
    run_report : dict
        A single run report from ExperimentRunner.
    t : int
        Time index into the macro DataFrame (default -1 = final period).
        Must be >= 1 in iloc terms (needs a previous period for lagged values).
    display : bool
        Whether to display the DataFrames using IPython.display.

    Returns
    -------
    transition_matrix : pd.DataFrame
        Numeric transition matrix with values.
    transition_matrix_symbols : pd.DataFrame
        Symbolic transition matrix showing variable names.
    """
    macro = run_report["results_macro"]
    p = run_report["parameters"]
    pc = run_report["parameters_calibrated"]

    dt = macro.iloc[t]          # Values at time t
    dtp = macro.iloc[t - 1]     # Values at time t-1

    # ── Direct macro variables ──────────────────────────────────────────────
    data = {
        'C':            dt['C'],
        'VAT':          dt['VAT'],
        'GY':           dt['GY'],
        'Sub':          dt['Sub_production'],
        'GI':           dt['GI'],
        'EX':           dt['EX'],
        'IM':           dt['IM'],
        'W':            dt['W'],
        'GP':           dt['GP'],
        'PB':           dt['PB'],
        'TH':           dt['TH'],
        'Tax_profits':  dt['Tax_profits'],
        'Tax_wages':    dt['Tax_wages'],
    }

    # ── Computed aggregates ─────────────────────────────────────────────────
    data['P_private'] = dt['P'] - dt['GP']
    data['IOT_taxes'] = (dt['Tax_products_net'] + dt['Tax_production']
                         + dt['Tax_investments'])
    # Bank profits split: gov_bank_share goes to Gov (already included in GP); private share flows to HH.
    data['PB_HH']  = (1.0 - pc.gov_bank_share) * dt['PB']
    data['PB_gov'] = pc.gov_bank_share           * dt['PB']

    # ── Flows requiring parameters and lagged values ────────────────────────
    # Interest and repayment flows (using lagged stocks, no circularity)
    data['rm_V'] = pc.rm * dtp['V']                           # Interest paid on deposits (return on HH portfolio claims)
    # Combined: bank profit share to HH + return on HH portfolio claims (V).
    # Both are returns flowing from Banks to Households; shown as a single row.
    data['PB_HH_rm_V'] = data['PB_HH'] + data['rm_V']        # HH receives profit share + deposit return
    data['PB_rm_V']    = dt['PB']      + data['rm_V']         # Banks pay out total profit + deposit return
    data['rl_L'] = pc.rl * dtp['L']                           # Interest paid on loans
    data['rho_L'] = p.ρ * dtp['L']                            # Loan principal repayment

    # Domestic bond service (Banks hold domestic bonds, receive coupon + amortisation from Gov)
    data['dom_bond_service'] = (np.sum(pc.interest_gov_bonds) + p.ρ) * dtp['Bond_domestic']

    # External bond service (ROW holds external bonds, receives coupon + principal from Gov)
    # Bond_external_interest_cost and Bond_external_repayment are computed in the model
    data['ext_bond_service'] = (dt['Bond_external_interest_cost']
                                + dt['Bond_external_repayment'])

    data['gov_exp_resid'] = pc.gov_exp_resid

    # ── Investment financing ────────────────────────────────────────────────
    # invest_profit: share of DOMESTIC private investment financed by profits (not loans).
    # All computations use I_private_dom (domestically-financed investment) as the base,
    # consistent with the model equation: L_[t] = L_[t-1]*(1-ρ) + I_private_dom*(1-invest_profit).
    data['I_private']         = dt['I_private']       # total private investment (used in investment row)
    data['I_private_dom']     = dt['I_private_dom']   # domestically-financed private investment
    # Retained earnings: portion of domestic private investment self-financed from profits.
    # This is the amount deducted from household wealth in the model: V[t] = ... - sum(I_dom * invest_profit)
    data['retained_earnings']   = dt['invest_profit'] * dt['I_private_dom']
    # Distributed profits: uses model's explicit P_distributed variable (sum(s_priv*P) - (rl+rho)*L).
    # This is what firms actually transfer to HH after debt service and retained earnings.
    # The separate 'i on Loans' and 'Loan Repayment' rows show gross Firms→Banks flows;
    # together they produce correct bilateral zero-sum rows.
    # Note: HH S/D in TFM will differ from ΔV by approximately (VAT + remittances) because
    # those flows appear in TFM current rows for HH but are not deducted in the V equation.
    data['distributed_profits'] = dt['P_distributed'] - data['retained_earnings']
    # New loans: the loan-financed portion of domestic private investment (fixes prior use of I_private)
    data['new_loans'] = (1 - dt['invest_profit']) * dt['I_private_dom']

    # ── Stock changes ───────────────────────────────────────────────────────
    data['DeltaV'] = dt['V'] - dtp['V']
    data['DeltaBond_dom'] = dt['Bond_domestic'] - dtp['Bond_domestic']
    data['DeltaBond_ext'] = dt['Bond_external'] - dtp['Bond_external']
    data['DeltaGov_ext_assets'] = dt['Gov_ext_assets'] - dtp['Gov_ext_assets']

    # HH deposit change: ΔD_HH = ΔL (endogenous money: HH acquires deposit asset as loans expand/contract).
    # D_HH = L, so ΔD_HH = new_loans − rho_L. Shown as an explicit TFM row (HH acquires/reduces deposit asset).
    data['DeltaD_HH'] = dt['D_HH'] - dtp['D_HH']

    # Firm equity changes: stock-based Δ(K − FDI), split by capital-weighted public ownership share.
    # These are the economically meaningful values; the statistical discrepancy is left to emerge
    # naturally from the model flows rather than being forced to zero by a residual.
    _s_pub_agg_tfm = np.sum(pc.s_public_ * pc.K_) / np.sum(pc.K_)
    _DeltaE_total        = (dt['K'] - dt['L'] - dt['FDI_net_stock_total']) - (dtp['K'] - dtp['L'] - dtp['FDI_net_stock_total'])
    data['DeltaE_firm']     = _DeltaE_total
    data['DeltaE_firm_HH']  = (1.0 - _s_pub_agg_tfm) * _DeltaE_total
    data['DeltaE_firm_gov'] = _s_pub_agg_tfm           * _DeltaE_total

    # ── BoP / international flows ───────────────────────────────────────────
    # FDI net inflow: foreign equity entering Saudi firms (ROW → Firms)
    data['FDI_net'] = dt['FDI_net_total']

    # Gov PIF investment: fiscal surplus invested abroad (Gov → ROW)
    data['Gov_pif'] = dt['Gov_pif_investment']
    # ── BoP income flows ────────────────────────────────────────────────────
    # FDI income repatriation: foreign investors' share of private profits leaving Saudi (HH → ROW).
    # Deducted from v.YD_profit in the model; full amount stored in v.FDI_income_payments.
    data['FDI_income'] = dt['FDI_income_payments']

    # Remittances: foreign workers send earnings abroad (HH → ROW).
    data['remittances'] = dt['remittances']

    # Gov ext asset income: Saudi Government earns investment returns on PIF/SAMA foreign assets (ROW → Gov).
    # Full amount shown (consistent with how it enters primary_income_balance in the BoP).
    data['Gov_ext_assets_income'] = dt['Gov_ext_assets_income']
    # ── Production adjustment ───────────────────────────────────────────────
    # Gap between demand-side Y and production-side value added (X − IntP).
    # Arises from agricultural sector modifications and supply constraints.
    value_added = dt['X'] - dt['IntP']
    Y_demand = dt['C'] + dt['GY'] + dt['I_total'] + dt['EX'] - dt['IM']
    Y_realized = dt['Y']
    data['prod_adj_realized'] = Y_realized - value_added
    data['prod_adj_demand'] = value_added - Y_demand  # negative when Y > VA

    # ── Matrix structure ────────────────────────────────────────────────────
    # Firms are split into two sub-accounts following SFC convention:
    #   Firms (Current): all income and expenditure flows; net = retained earnings
    #   Firms (Capital): investment expenditure financed by retained earnings + loans + FDI equity
    columns = ["Households", "Firms (Current)", "Firms (Capital)", "Banks", "Government", "ROW"]

    # ── Section 1: Cash-flow rows (income, expenditure, transfers) ──────────
    # ── Section 2: Changes in financial assets ──────────────────────────────
    rows = [
        # ── Section 1: Cash flows ───────────────────────────────────────────
        "Consumption",
        "VAT",
        "Gov Consumption",
        "Gov Subsidies",
        "Priv. Investment (dom. financed)",  # Firms (Current) revenue from selling; Firms (Capital) expenditure
        "Gov Investment",
        "Exports",
        "Imports",
        "Wages",
        "Distributed Profits",         # Private profits net of debt service paid to HH (= P_distributed − retained_earnings)
        "Retained Earnings",           # Internal Firms (Current) → Firms (Capital) transfer
        "Gov Profits",
        "Bank Profits + return to HH portfolio claims",  # Combined: (PB_HH + rm*V) → HH, PB_gov → Gov, (PB + rm*V) from Banks
        "Household Tax (TH)",
        "Tax on Profits",
        "Tax on Wages",
        "IOT Taxes",
        "i on Loans",                  # Firms (Current) pays gross loan interest to Banks
        "Dom Bond Service (i+rho)",
        "Ext Bond Service (i+rho)",
        "FDI Income Repatriation",
        "Remittances",
        "Gov Ext Asset Income",
        "Gov Exp Residual",
        "Production Adj. realized",
        "Production Adj. demand",
        # ── Section 2: Changes in financial assets ──────────────────────────
        "New Loans",                   # Firms (Capital) borrows from Banks
        "Loan Repayment",              # Firms (Capital) repays principal to Banks
        "Change in HH Deposits (ΔD_HH)",  # HH acquires/reduces deposit asset = ΔL (endogenous money: ΔD_HH = new_loans − rho_L)
        "Change in Household Wealth (V)",  # NOTE: V = HH portfolio wealth; includes revaluation effects beyond new loans — residual captured via equity rows below
        "Change in Dom. Bonds",
        "Change in Ext. Bonds",
        "Priv. Investment (FDI from RoW)",  # Equity inflow to Firms (Capital) from ROW — finances FDI-funded portion of private investment
        "Change in Gov Ext Assets",
        "Gov PIF Investment",          # Gov acquires foreign assets → financial account (asset acquisition)
        "Change in Firm Equity — HH (ΔE_f,HH) [incl. revaluation]",   # HH equity residual: forces HH S/D + NLNB = 0; diff from Δ(K−FDI) = implicit revaluation
        "Change in Firm Equity — Gov (ΔE_f,Gov) [incl. revaluation]",  # Gov equity residual: forces Gov S/D + NLNB = 0
    ]

    # Each inner list: [Households, Firms (Current), Firms (Capital), Banks, Government, ROW]
    symbols = [
        # ── Section 1: Cash flows ──────────────────────────────────────────────────────────────────────────────────────────────
        ["-C",                    "+C",                    "",                      "",                      "",                      ""                   ],  # Consumption
        ["-VAT",                  "",                      "",                      "",                      "+VAT",                  ""                   ],  # VAT
        ["",                      "+GY",                   "",                      "",                      "-GY",                   ""                   ],  # Gov Consumption
        ["",                      "+Sub",                  "",                      "",                      "-Sub",                  ""                   ],  # Gov Subsidies
        ["",                      "+I_private_dom",         "-I_private_dom",         "",                      "",                      ""                   ],  # Priv. Investment (dom. financed) — uses I_private_dom (domestically-financed share)
        ["",                      "+GI",                   "",                      "",                      "-GI",                   ""                   ],  # Gov Investment
        ["",                      "+EX",                   "",                      "",                      "",                      "-EX"                ],  # Exports
        ["",                      "-IM",                   "",                      "",                      "",                      "+IM"                ],  # Imports
        ["+W",                    "-W",                    "",                      "",                      "",                      ""                   ],  # Wages
        ["+distributed_profits",  "-distributed_profits",  "",                      "",                      "",                      ""                   ],  # Distributed Profits
        ["",                      "-retained_earnings",    "+retained_earnings",    "",                      "",                      ""                   ],  # Retained Earnings
        ["",                      "-GP",                   "",                      "",                      "+GP",                   ""                   ],  # Gov Profits
        ["+PB_HH_rm_V",           "",                      "",                      "-PB_rm_V",              "+PB_gov",               ""                   ],  # Bank Profits + rm*V: (PB_HH+rm*V)→HH, (PB+rm*V) from Banks, PB_gov→Gov (row sums to 0)
        ["-TH",                   "",                      "",                      "",                      "+TH",                   ""                   ],  # Household Tax (TH)
        ["-Tax_profits",          "",                      "",                      "",                      "+Tax_profits",          ""                   ],  # Tax on Profits
        ["-Tax_wages",            "",                      "",                      "",                      "+Tax_wages",            ""                   ],  # Tax on Wages
        ["",                      "-IOT_taxes",            "",                      "",                      "+IOT_taxes",            ""                   ],  # IOT Taxes
        ["",                      "-rl_L",                 "",                      "+rl_L",                 "",                      ""                   ],  # i on Loans (Firms pays Banks)
        ["",                      "",                      "",                      "+dom_bond_service",     "-dom_bond_service",     ""                   ],  # Dom Bond Service
        ["",                      "",                      "",                      "",                      "-ext_bond_service",     "+ext_bond_service"  ],  # Ext Bond Service
        ["-FDI_income",           "",                      "",                      "",                      "",                      "+FDI_income"        ],  # FDI Income Repatriation
        ["-remittances",          "",                      "",                      "",                      "",                      "+remittances"       ],  # Remittances
        ["",                      "",                      "",                      "",                      "+Gov_ext_assets_income",  "-Gov_ext_assets_income" ],  # Gov Ext Asset Income
        ["+gov_exp_resid",        "",                      "",                      "",                      "-gov_exp_resid",        ""                   ],  # Gov Exp Residual
        ["",                      "+prod_adj_realized",    "",                      "",                      "",                      ""                   ],  # Production Adj. realized
        ["",                      "+prod_adj_demand",      "",                      "",                      "",                      ""                   ],  # Production Adj. demand
        # ── Section 2: Changes in financial assets ─────────────────────────────────────────────────────────────────────────────
        ["",                      "",                      "+new_loans",            "-new_loans",            "",                      ""                   ],  # New Loans
        ["",                      "",                      "-rho_L",                "+rho_L",                "",                      ""                   ],  # Loan Repayment
        ["-DeltaD_HH",            "",                      "",                      "+DeltaD_HH",            "",                      ""                   ],  # Change in HH deposits: HH acquires asset (−), Banks issue liability (+)
        ["-DeltaV",               "",                      "",                      "+DeltaV",               "",                      ""                   ],  # Change in household wealth (V) — tracks ΔV stock; Banks column shows counterpart asset change
        ["",                      "",                      "",                      "+DeltaBond_dom",        "-DeltaBond_dom",        ""                   ],  # Change in Dom. Bonds
        ["",                      "",                      "",                      "",                      "-DeltaBond_ext",        "+DeltaBond_ext"     ],  # Change in Ext. Bonds
        ["",                      "",                      "+FDI_net",              "",                      "",                      "-FDI_net"           ],  # Priv. Investment (FDI from RoW)
        ["",                      "",                      "",                      "",                      "-DeltaGov_ext_assets",   "+DeltaGov_ext_assets"   ],  # Change in Gov Ext Assets
        ["",                      "",                      "",                      "",                      "-Gov_pif",              "+Gov_pif"           ],  # Gov PIF = asset acquisition (financial account)
        ["+DeltaE_firm_HH",       "",                      "-DeltaE_firm_HH",       "",                      "",                      ""                   ],  # Firm Equity HH: HH(+HH share), Firms(Capital)(−HH share) → row sums to 0
        ["",                      "",                      "-DeltaE_firm_gov",      "",                      "+DeltaE_firm_gov",      ""                   ],  # Firm Equity Gov: Firms(Capital)(−gov share), Gov(+gov share) → row sums to 0
    ]

    # ── Build matrix ─────────────────────────────────────────────────────────
    numeric = [[string_to_value(s, data) for s in row] for row in symbols]
    transition_matrix_symbols = pd.DataFrame(symbols, columns=columns, index=rows)
    transition_matrix = pd.DataFrame(numeric, columns=columns, index=rows)

    # ── Balancing adjustment ──────────────────────────────────────────────────
    # The TFM must sum to zero globally (every payment has a counterpart).
    # In practice the "Production Adj." rows are one-sided (only the Firms
    # cell is non-zero), so we absorb any residual into that catch-all row.
    global_flow_sum = float(transition_matrix[columns].sum().sum())
    firms_col = columns.index("Firms (Current)")
    pa_row = "Production Adj. realized"
    transition_matrix.at[pa_row, columns[firms_col]] -= global_flow_sum

    # ── Section boundaries ──────────────────────────────────────────────────
    cf_last = "Production Adj. demand"          # last cash-flow row
    fin_first = "New Loans"                      # first financial-changes row
    cf_rows = rows[: rows.index(cf_last) + 1]
    fin_rows = rows[rows.index(fin_first):]

    # ── Computed summary rows ───────────────────────────────────────────────
    # Surplus/Deficit: column sum of cash-flow section (positive = surplus)
    surplus_deficit = transition_matrix.loc[cf_rows, columns].sum(axis=0)

    # NLNB (explicit, before OFA): column sum of explicitly-modelled financial rows.
    _nlnb_explicit = transition_matrix.loc[fin_rows, columns].sum(axis=0)

    # ── Other Financial Assets (OFA): residual SFC clearing instrument ───────
    # For every sector: ΔOFA_i = -(S/D_i + NLNB_explicit_i)
    # This guarantees NLNB = -S/D identically for all sectors.
    # Post-Keynesian rationale: endogenous money and circuit theory guarantee
    # that every financing gap is covered by some financial instrument; OFA
    # represents all unmodelled instruments (trade credit, interbank claims,
    # pension fund shares, repo, etc.).  Σ ΔOFA_i = 0 (global flow balance).
    _OFA_ROW_LABEL = 'Change in Other Financial Assets (OFA) (\u0394OFA)'
    _delta_ofa = -(surplus_deficit + _nlnb_explicit)
    transition_matrix.loc[_OFA_ROW_LABEL, columns] = _delta_ofa.values
    transition_matrix_symbols.loc[_OFA_ROW_LABEL, columns] = [
        '+DOFA_HH', '+DOFA_FC', '+DOFA_FK', '+DOFA_Banks', '+DOFA_Gov', '+DOFA_ROW']
    fin_rows_incl_ofa = fin_rows + [_OFA_ROW_LABEL]

    # NLNB (including OFA) = _nlnb_explicit + ΔOFA = -S/D ✓
    _NLNB_LABEL = 'Net Lending/Borrowing (NLNB) [\u2212 net lending, + net borrowing]'
    nlnb_values = transition_matrix.loc[fin_rows_incl_ofa, columns].sum(axis=0)

    # ── Assign computed rows ────────────────────────────────────────────────
    transition_matrix.loc['Surplus/Deficit (S/D)', columns] = surplus_deficit.values
    transition_matrix.loc[_NLNB_LABEL, columns] = nlnb_values.values

    transition_matrix_symbols.loc['Surplus/Deficit (S/D)', columns] = 'S/D'
    transition_matrix_symbols.loc[_NLNB_LABEL, columns] = 'NLNB'

    # Statistical Discrepancy: S/D + NLNB = 0 everywhere after OFA row is included.
    # Retained as a diagnostic row — all values should display as zero.
    _STAT_DISC_LABEL = 'Statistical Discrepancy'
    stat_disc_values = surplus_deficit.values + nlnb_values.values
    transition_matrix.loc[_STAT_DISC_LABEL, columns] = stat_disc_values
    transition_matrix_symbols.loc[_STAT_DISC_LABEL, columns] = 'StatDisc'

    # ── Reorder: S/D | financial rows | OFA row | NLNB | Stat. Discrepancy ──
    final_order = cf_rows + ['Surplus/Deficit (S/D)'] + fin_rows + [_OFA_ROW_LABEL, _NLNB_LABEL, _STAT_DISC_LABEL]
    transition_matrix        = transition_matrix.reindex(final_order)
    transition_matrix_symbols = transition_matrix_symbols.reindex(final_order, fill_value='')

    # ── Row sums ────────────────────────────────────────────────────────────
    transition_matrix['Sum'] = transition_matrix[columns].sum(axis=1)

    # ── Symbol Sum column ───────────────────────────────────────────────────
    transition_matrix_symbols['Sum'] = ''
    transition_matrix_symbols.loc['Surplus/Deficit (S/D)', 'Sum'] = '0'
    transition_matrix_symbols.loc[_OFA_ROW_LABEL, 'Sum'] = '0'
    transition_matrix_symbols.loc[_NLNB_LABEL, 'Sum'] = '0'
    transition_matrix_symbols.loc[_STAT_DISC_LABEL, 'Sum'] = ''

    if display:
        print("Transaction flow matrix (Symbols):")
        IPdisplay(transition_matrix_symbols)
        print(f"\nTransaction flow matrix (Values in billion SAR, t={t}):")
        IPdisplay((transition_matrix / 1e6).round(2))

    return transition_matrix, transition_matrix_symbols
