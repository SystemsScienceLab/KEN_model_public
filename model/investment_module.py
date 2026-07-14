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

from .model_classes import ModelConfig, ModelVariables, ModelResults, ModelParameters
from .calibration import ParametersCalibrated
from .FDI_module import fdi_module
from .energy_investment_module import energy_investment_module
import numpy as np
import pandas as pd
import warnings

###################################################################################################################################################################################################################################################################################################
def investment_module(v: ModelVariables, p: ModelParameters, pc: ParametersCalibrated, t: int, verbose=False):
###################################################################################################################################################################################################################################################################################################    
    ########################################################################################################################################################################################################################
    # 1. Energy-related investments (RE, efficiency, electrification, hydrogen, CCS, CCU, storage)
    ########################################################################################################################################################################################################################
    energy_investment_module(v, p, pc, t)


    ########################################################################################################################################################################################################################
    # 2. Desalination and wastewater investment
    ########################################################################################################################################################################################################################
    if t == 1:
        v.I_[t][pc.sector_desal] = pc.I_desal
        v.I_[t][pc.sector_wwater] = pc.I_wwater
    elif ((v.pot_desal[t-1] + v.pot_wwater[t-1]) > v.water_use_total_national[t-1]) and (t > p.vision_2030_timing):
        # Case: Overcapacity (Supply > Demand) AND outside Vision 2030 period -> Stop Investment
        v.I_[t][pc.sector_desal] = 0
        v.I_[t][pc.sector_wwater] = 0
    else:
        # 1. Determine Growth Strategy
        # Use High Growth if: (Gap Exists) OR (Vision 2030) OR (High Growth Policy)
        # 2. Desalination Investment
        # Strategy:
        # - Before 2030: Force high growth (Policy Driven) IF the scenario supports it (p.desalination_high_growth == 1).
        # - After 2030: High growth only if water gap exists (Demand Driven).
        # - Otherwise: Endogenous growth.
        
        # Condition A: Policy Driven (Vision 2030 Period)
        policy_high_growth = (p.desalination_high_growth == 1) and (t <= p.vision_2030_timing)
        
        # Condition B: Demand Driven (Water Gap)
        demand_high_growth = (v.water_use_total_national[t-1] > (v.pot_desal[t-1] + v.pot_wwater[t-1]))
        
        use_high_growth = policy_high_growth or demand_high_growth

        # 2. Desalination Investment
        # Step A: Determine Base Investment
        if v.I_[t-1][pc.sector_desal] > 0:
            base_desal = v.I_[t-1][pc.sector_desal]
        else: # if no investment in previous year, calculate based on depreciation
            base_desal = pc.δ_desal * v.K_[t-1][pc.sector_desal]
            
        # Step B: Apply Growth Rate
        if use_high_growth:
            v.I_[t][pc.sector_desal] = base_desal * (1 + pc.gDesalCap_avg)
        else: 
            # Maturity Phase: Grow with the economy (Endogenous rate, typically 2-4%)
            # Once the high-growth Vision 2030 phase is over and there are no supply gaps,
            # the sector stabilizes and grows in line with overall economic investment trends.
            v.I_[t][pc.sector_desal] = base_desal * (1 + v.gI_trend_endog[t])

        # 3. Wastewater Investment
        # Step A: Determine Base Investment
        if v.I_[t-1][pc.sector_wwater] > 0:
             base_wwater = v.I_[t-1][pc.sector_wwater]
        else: # if no investment in previous year, calculate based on depreciation
             base_wwater = pc.δ_wwater * v.K_[t-1][pc.sector_wwater]
            
        # Step B: Apply Growth Rate
        if use_high_growth:
            v.I_[t][pc.sector_wwater] = base_wwater * (1 + pc.gWWaterCap_avg)
        else: 
            # Maturity Phase: Grow with the economy (Endogenous rate, typically 2-4%)
            # Same logic as desalination: Stabilize growth after 2030 unless gaps exist.
            v.I_[t][pc.sector_wwater] = base_wwater * (1 + v.gI_trend_endog[t])
    


    ########################################################################################################################################################################################################################
    # 3. General sectoral investment distribution
    ########################################################################################################################################################################################################################
# --------------------------------------------------------------------------------
    # --- UPDATED THEME 2: ELECTRICITY SECTOR BASELINE & GRID UPGRADES ---
    # --------------------------------------------------------------------------------
    # 1. Total Base trend investment (derived from historical IO tables)
    total_base_elec_inv = pc.I_[pc.electricity_gas_AC_sector] * (1 + v.gI_trend_endog[t])

    # 2. Prevent Double-Counting (Removing baseline generation)
    generation_share_of_baseline = getattr(p, 'historical_generation_capex_share', 0.45) 
    base_grid_and_ops_inv = total_base_elec_inv * (1 - generation_share_of_baseline)

    # 3. Grid Upgrade Factor & CAPEX Allocation via Dictionary Shares
    grid_upgrade_cost = 0.0
    grid_upgrade_demand_ = np.zeros(pc.S)

    if p.enable_industrial_electrification and t > 1:
        # --- CALCULATION OF GRID UPGRADE COST (116.0 SAR/GJ) ---
        # Source: SEC 2025-2030 Grid Modernization Plan ($58.7 Billion for 100GW)
        # Link: https://www.seetaoe.com/details/249898.html
        # Base cost is ~2,200 SAR/kW. Assuming 60% load factor (18.92 GJ/year) -> ~116 SAR/GJ.
        grid_upgrade_cost = (v.electrification_added_load[t] * getattr(p, 'grid_upgrade_cost_per_gj', 116.0)) / 1000.0

        # --- DEFINE SECTORAL SHARES FOR BUILDING THE GRID (Mapped to exact KEN sectors) ---
        grid_upgrade_shares = {
            'Manufacture of electrical equipment': 0.45,  # Transformers, substations, switchgear
            'Civil engineering': 0.40,                    # Towers, trenching, foundations
            'Manufacture of basic metals': 0.15           # Aluminum and copper cables
        }

        for sector_name, share in grid_upgrade_shares.items():
            if sector_name in pc.sectors:
                idx = pc.sectors.index(sector_name)
                grid_upgrade_demand_[idx] += grid_upgrade_cost * share
            else:
                fallback_idx = pc.sectors.index('Construction of buildings') if 'Construction of buildings' in pc.sectors else 0
                grid_upgrade_demand_[fallback_idx] += grid_upgrade_cost * share

    # 4. Total Electricity Sector Investment (Who OWNS the grid)
    v.I_[t][pc.electricity_gas_AC_sector] = base_grid_and_ops_inv + v.I_RE[t] + v.I_hydrogen[t] + grid_upgrade_cost
    # --------------------------------------------------------------------------------
    if t == 1:
        I_ALL_except_energy_desal_wwater = v.I_total[t-1]  - pc.I_desal - pc.I_wwater - pc.I_RE
    else:    
        # We must subtract EVERYTHING added in the previous year to find the true economic baseline
        prev_netzero_total = (np.sum(v.I_netzero_ccs_[t-1]) + np.sum(v.I_netzero_nature_[t-1]) + 
                              (np.sum(v.I_netzero_ccu_[t-1]) if hasattr(v, 'I_netzero_ccu_') else 0))
        
        # v.I_[t-1][sector] already INCLUDES the efficiency added in the previous period.
        # Subtracting v.I_efficiency_total[t-1] on top of that would double-count the efficiency
        # for the desal, wwater, and electricity sectors. Subtract efficiency back from each
        # energy sector's value before adding v.I_efficiency_total.
        prev_energy_total = (
            (v.I_[t-1][pc.sector_desal]              - v.I_efficiency_[t-1][pc.sector_desal]) +
            (v.I_[t-1][pc.sector_wwater]             - v.I_efficiency_[t-1][pc.sector_wwater]) +
            (v.I_[t-1][pc.electricity_gas_AC_sector] - v.I_efficiency_[t-1][pc.electricity_gas_AC_sector]) +
            v.I_efficiency_total[t-1] +       # now added once for ALL sectors (including the three above)
            v.I_electrification_total[t-1] +
            prev_netzero_total
        )
        
        I_ALL_except_energy_desal_wwater = v.I_total[t-1] - prev_energy_total  
    I_ALL_except_energy_desal_wwater = np.maximum(0, I_ALL_except_energy_desal_wwater)

    # Create a mask for sectors to include in investment distribution (exclude desal, wwater, and energy sectors)
    sectors_to_exclude = np.zeros(pc.S, dtype=bool)
    sectors_to_exclude[pc.sector_desal] = True
    sectors_to_exclude[pc.sector_wwater] = True
    sectors_to_exclude[pc.electricity_gas_AC_sector] = True

    sectors_to_include = ~sectors_to_exclude    

    
    ##################################################
    # Distribute investment according to sectoral output shares in Y OR X
    ##################################################
    # Y-based distribution ties investment to value added (sectoral Y shares).
    if p.Investment_distribution_Y_based:  
        v.Sectoral_investment_distribution_[t][sectors_to_include] = v.Y_[t-1][sectors_to_include] / np.sum(v.Y_[t-1][sectors_to_include])
    else:
        v.Sectoral_investment_distribution_[t][sectors_to_include] = v.X_[t-1][sectors_to_include] / np.sum(v.X_[t-1][sectors_to_include])
    
    ##################################################
    # Endogenous investment switch    
    ##################################################
    if p.Endogenous_investment:  # Endogenous investment in current formulation is only effective if coupled with Endogenous_A_matrix = True
        if t == 1:
            v.gI_endog_[t] = v.gI_trend_endog[t] * np.ones_like(v.gI_endog_[t])
            # Adjusted to subtract both desalination and wastewater initial investments
            v.I_[t][sectors_to_include] = I_ALL_except_energy_desal_wwater * (1 + v.gI_endog_[t][sectors_to_include]) * v.Sectoral_investment_distribution_[t][sectors_to_include]
        # Endogenous investment includes a scarcity constraint-induced additional investment mechanism, designed to alleviate scarcity constraints
        # Before investment is determined: If investment in a maturing economy exceeds consumption, reduce average investment growth rate by an exogenous factor
        if p.Endogenous_investment_reduction: # With a fixed factor and the amount of excess investment over consumption in percent to consumption
            if np.sum(v.I_[t-1]) > v.C[t-1]:
                if t==1:
                    v.gI_trend_endog[t] = pc.gI_avg
                else:
                    v.gI_trend_endog[t] = v.gI_trend_endog[t-1] * p.investment_reduction_factor
            else:
                v.gI_trend_endog[t] = pc.gI_avg
        else:
            v.gI_trend_endog[t] = pc.gI_avg
        if p.Endogenous_excess_import_reduction and p.Adapt_investment_production_scarcity: # Scarcity in production constraints enters investment growth rate next to import excess
            v.gI_endog_[t] = v.gI_trend_endog[t] * (1 + v.excess_import_[t-1]*p.import_excess_adjustment + v.supply_constraint_IntP_[t-1]*p.investment_scarcity_factor + v.supply_constraint_K_[t-1]*p.investment_scarcity_factor)
        elif p.Endogenous_excess_import_reduction: # Investment growth rate only determined by import excess
            v.gI_endog_[t] = v.gI_trend_endog[t] * (1 + v.excess_import_[t-1]*p.import_excess_adjustment)
        # IMPORTANT part: Here, the endogenous growth rate for investment is applied to each sectoral investment based on the capital stock distribution
        if t > 1: # Only for t > 1, as t=1 is handled above.
            v.I_[t][sectors_to_include] = I_ALL_except_energy_desal_wwater * (1 + v.gI_endog_[t][sectors_to_include]) * v.Sectoral_investment_distribution_[t][sectors_to_include]
    else:
        if t == 1 or not getattr(p, 'Investment_exog_semi_endog_correction', False):
            # Period 1 or correction disabled: use raw calibrated trend rate
            v.gI_trend_endog[t] = pc.gI_avg
        else:
            # ──────────────────────────────────────────────────────────────────
            # Semi-endogenous Keynesian dampening of exogenous investment trend
            # Motivation: even with an exogenous trend rate, large macro
            # imbalances should moderate investment (Keynesian channel):
            #   1. Gov deficit crowding-out: high deficit → higher borrowing
            #      costs and reduced business confidence
            #   2. Profit rate decline: lower retained earnings → less internal
            #      finance available for new investment
            # Both signals are measured at t-1, normalised to GDP, and combined
            # additively into correction_factor ∈ [0,1] scaling pc.gI_avg.
            # ──────────────────────────────────────────────────────────────────
            gdp_t1 = max(np.sum(v.Y_[t-1]), 1.0)   # positive aggregate GDP denominator

            # Signal 1: government deficit / GDP
            gov_deficit       = v.Gov_exp[t-1] - v.Gov_rev[t-1]
            gov_deficit_ratio = gov_deficit / gdp_t1
            gov_threshold     = getattr(p, 'I_correction_gov_deficit_threshold', 0.05)
            gov_sensitivity   = getattr(p, 'I_correction_gov_deficit_sensitivity', 2.0)
            gov_correction    = max(0.0, gov_deficit_ratio - gov_threshold) * gov_sensitivity

            # Signal 2: profit rate fall relative to calibrated initial level
            profit_rate_now   = np.sum(v.P_[t-1]) / gdp_t1
            profit_rate_init  = np.sum(pc.P_) / max(np.sum(pc.Y_), 1.0)
            profit_gap        = max(0.0, profit_rate_init - profit_rate_now)
            profit_sensitivity = getattr(p, 'I_correction_profit_sensitivity', 2.0)
            profit_correction  = profit_gap * profit_sensitivity

            # Combined: scale gI_trend down, bounded to [0, 1]
            correction_factor   = max(0.0, 1.0 - gov_correction - profit_correction)
            v.gI_trend_endog[t] = pc.gI_avg * correction_factor

        v.I_[t][sectors_to_include] = I_ALL_except_energy_desal_wwater * (1 + v.gI_trend_endog[t]) * v.Sectoral_investment_distribution_[t][sectors_to_include]


#########################################################################################
    # 4. Consolidate all energy investments into the Sectoral Investment Vector (v.I_)
    #########################################################################################
    # A. Add the pre-calculated array investments
    v.I_[t] = v.I_[t] + v.I_efficiency_[t] + v.I_netzero_ccs_[t] + v.I_netzero_nature_[t] + v.I_netzero_ccu_[t]
    
    # B. Assign Battery Storage Ownership (Owned by the Utility sector)
    v.I_[t][pc.electricity_gas_AC_sector] += v.I_storage[t]
    
    # C. Assign Electrification Equipment Ownership
    # Distribute the cost of industrial heat pumps/equipment to the sectors making the switch,
    # weighted by their output (v.X_) to simulate larger sectors paying more.
    if p.enable_industrial_electrification and v.I_electrification_total[t] > 0:
        target_indices = pc.electrification_target_sectors
        total_target_X = np.sum(v.X_[t-1][target_indices])
        
        if total_target_X > 0:
            for idx in target_indices:
                share = v.X_[t-1][idx] / total_target_X
                v.I_[t][idx] += v.I_electrification_total[t] * share

    # D. Recalculate Subtotals for Demand Vector usage
    total_ccs_inv = np.sum(v.I_netzero_ccs_[t])
    total_nature_inv = np.sum(v.I_netzero_nature_[t])
    total_ccu_inv = np.sum(v.I_netzero_ccu_[t]) if hasattr(v, 'I_netzero_ccu_') else 0.0


    ########################################################################################################################################################################################################################
    # 5. FDI module: Decompose investment into domestic + FDI components
    ########################################################################################################################################################################################################################
    # Pre-compute I_total[t] here so fdi_module can compute FDI_share_of_investment
    # and apply the 95% aggregate cap correctly. It will be recomputed later (model.py
    # line ~941) after any countercyclical policy boost, so there is no double-counting.
    v.I_total[t] = np.sum(v.I_[t])
    fdi_module(v, p, pc, t)

    ########################################################################################################################################################################################################################
    # 6. Investment demand vectors
    ########################################################################################################################################################################################################################
    mach_idx = pc.sectors.index('Manufacture of machinery and equipment n.e.c.') if 'Manufacture of machinery and equipment n.e.c.' in pc.sectors else 5
    # Create Storage Demand Vector
    storage_demand_vec = np.zeros(pc.S)
    storage_demand_vec[mach_idx] = v.I_storage_local[t] # Only local generates domestic activity
    # 3. Update the Demand Vector
    re_investment_demand_ = v.I_RE[t] * pc.dI_RE_
    
    # --------------------------------------------------------------------------------
    # ---  HYDROGEN INVESTMENT DEMAND (Supply Chain) ---
    # --------------------------------------------------------------------------------
    h2_investment_demand_ = np.zeros(pc.S)
    
    if v.I_hydrogen[t] > 0:
        # Define who builds the hydrogen plants (Electrolyzers + Facilities)
        h2_demand_shares = {
            'Manufacture of machinery and equipment n.e.c.': 0.60, # Electrolyzers/Compressors
            'Civil engineering': 0.40                              # Plant construction/Pipelines
        }
        
        for sector_name, share in h2_demand_shares.items():
            if sector_name in pc.sectors:
                idx = pc.sectors.index(sector_name)
                h2_investment_demand_[idx] += v.I_hydrogen[t] * share
            else:
                fallback_idx = pc.sectors.index('Construction of buildings') if 'Construction of buildings' in pc.sectors else 0
                h2_investment_demand_[fallback_idx] += v.I_hydrogen[t] * share
    # --------------------------------------------------------------------------------

    nz_investment_demand_ = (total_ccs_inv + total_nature_inv + total_ccu_inv) * pc.dI    
    efficiency_investment_demand = v.I_efficiency_total[t] * pc.dI_Effic_
    electrification_investment_demand_ = v.I_electrification_total[t] * pc.dI_Electrification_
    
# Compute "other" as the exact residual so that sum(I_demand_[t]) = sum(I_[t]) - I_storage_import
    # by construction, regardless of any distribution-vector normalisation.
    # ── IDENTITY GUARDS (always run at t=3) ──────────────────────────────────
    if t == 3:
        assert abs(np.sum(pc.dI) - 1.0) < 1e-6,               f"pc.dI does not sum to 1: {np.sum(pc.dI)}"
        assert abs(np.sum(pc.dI_RE_) - 1.0) < 1e-6,           f"pc.dI_RE_ does not sum to 1: {np.sum(pc.dI_RE_)}"
        assert abs(np.sum(pc.dI_Effic_) - 1.0) < 1e-6,        f"pc.dI_Effic_ does not sum to 1: {np.sum(pc.dI_Effic_)}"
        assert abs(np.sum(pc.dI_Electrification_) - 1.0) < 1e-6, f"pc.dI_Electrification_ does not sum to 1: {np.sum(pc.dI_Electrification_)}"
        # ─────────────────────────────────────────────────────────────────────────────

    # Use np.sum(re_investment_demand_) instead of v.I_RE[t]:
    # If dI_RE_ sums to slightly more than 1 (duplicate sector entries in pc.sectors), the old
    # formula subtracted I_RE but added back I_RE * sum(dI_RE_) > I_RE, creating a positive gap.
    # Using the ACTUAL demand sum as the subtractor forces the identity to hold always.
    # All other distribution vectors (dI_Effic_, dI_Electrification_, dI, h2/grid shares) are
    # verified to sum to 1.0, so their scalar vs np.sum() forms are equivalent.
    # ── OTHER INVESTMENT DEMAND — pure residual approach ─────────────────────────
    # Compute the scalar residual first (what is NOT explained by named components)
    _other_scalar = (
        np.sum(v.I_[t])
        - np.sum(re_investment_demand_)          # demand sum == v.I_RE[t] only if dI_RE_ sums to 1
        - np.sum(efficiency_investment_demand)   # = v.I_efficiency_total[t] * dI_Effic_ summed
        - np.sum(electrification_investment_demand_)
        - np.sum(grid_upgrade_demand_)
        - np.sum(h2_investment_demand_)
        - np.sum(nz_investment_demand_)          # = (ccs+nature+ccu) * dI summed
        - v.I_storage[t]                 # full storage (local + import)        
    )
    # Distribute residual by pc.dI (must sum to 1.0 — guarded above)
    other_investment_demand_ = np.maximum(0, _other_scalar) * pc.dI
# ─────────────────────────────────────────────────────────────────────────────


    # --- EFFICIENCY DEMAND NORMALISATION ---
    # v.I_efficiency_[t] is computed from energy intensities (energy_investment_module).
    # v.I_efficiency_total[t] is the scalar used in ownership allocation for v.I_[t].
    # These can diverge due to different sector weights. Force the demand vector
    # to sum to the same total as the ownership scalar so CHECK 2 stays clean.
    if v.I_efficiency_total[t] > 0 and np.sum(v.I_efficiency_[t]) > 0:
        v.I_efficiency_[t] = (
            v.I_efficiency_[t]
            / np.sum(v.I_efficiency_[t])
            * v.I_efficiency_total[t]
        )

    # Final Sum (Added grid_upgrade_demand_)
    v.I_demand_[t] = (other_investment_demand_ + 
                      re_investment_demand_ + 
                      efficiency_investment_demand + 
                      h2_investment_demand_ + 
                      nz_investment_demand_ + 
                      electrification_investment_demand_ + 
                      storage_demand_vec + 
                      grid_upgrade_demand_)

   ########################################################################################################################################################################################################################
    # 7. Private/public investment split
    ########################################################################################################################################################################################################################
    # Total investment (including FDI portion)
    # These are used for GDP accounting, demand, and capital stock evolution
    v.I_private_[t] = (v.I_[t] ) * pc.s_private_
    v.I_public_[t] =  (v.I_[t] ) * pc.s_public_

    # Domestic-only investment: subtract FDI to get the portion that must be
    # financed domestically (through loans, profits, or government budget).
    # I_private_dom_ is used for firm financing (loans, HH wealth).
    # I_public_dom_ is used for government expenditure (GI).
    if p.enable_FDI:
        v.I_public_dom_[t] = np.maximum(v.I_public_[t] - v.FDI_net_flow_[t] * pc.s_public_, 0.0)
        v.I_private_dom_[t] = np.maximum(v.I_private_[t] - v.FDI_net_flow_[t] * pc.s_private_, 0.0)
    else:
        v.I_public_dom_[t] = v.I_public_[t].copy()
        v.I_private_dom_[t] = v.I_private_[t].copy()

    ########################################################################################################################################################################################################################
    # 8. Government countercyclical policy
    ########################################################################################################################################################################################################################
    if p.Countercyclical_gov_policy:
        if v.gY[t-1] < 0 or v.inflation[t-1] < 0:
            if v.gy[t-1] < 0:
                v.Gov_confidence_boost[t] = - v.gy[t-1] 
            elif v.inflation[t-1] < 0:
                v.Gov_confidence_boost[t] = - v.inflation[t-1]
            boost_factor = (1 + v.Gov_confidence_boost[t] * p.gov_conf_boost_factor)
            v.I_public_[t] = v.I_public_[t] * boost_factor
            v.I_public_dom_[t] = v.I_public_dom_[t] * boost_factor
            
            # 1. Re-calculate total investment per sector after the public boost
            v.I_[t] = v.I_private_[t] + v.I_public_[t]
            v.I_total[t] = np.sum(v.I_[t])
            
            # 2. Re-calculate standard demand after boost.
            # Use np.sum(re_investment_demand_) so the accounting identity
            # sum(I_demand_) = sum(I_[t]) - I_storage_import holds by construction
            # even after the public-investment boost.
            # ── OTHER INVESTMENT DEMAND — pure residual approach ─────────────────────────
            # Compute the scalar residual first (what is NOT explained by named components)
            _other_scalar = (
                np.sum(v.I_[t])
                - np.sum(re_investment_demand_)          # demand sum == v.I_RE[t] only if dI_RE_ sums to 1
                - np.sum(efficiency_investment_demand)   # = v.I_efficiency_total[t] * dI_Effic_ summed
                - np.sum(electrification_investment_demand_)
                - np.sum(grid_upgrade_demand_)
                - np.sum(h2_investment_demand_)
                - np.sum(nz_investment_demand_)          # = (ccs+nature+ccu) * dI summed
                - v.I_storage[t]                 # full storage (local + import)        
            )
            # Distribute residual by pc.dI (must sum to 1.0 — guarded above)
            other_investment_demand_ = np.maximum(0, _other_scalar) * pc.dI
# ─────────────────────────────────────────────────────────────────────────────




            # 3. Re-assemble the total demand vector
            # This preserves all the specific supply chains (Grid, H2, NZ, Storage) constructed above.
            v.I_demand_[t] = (other_investment_demand_ + 
                              re_investment_demand_ + 
                              efficiency_investment_demand + 
                              h2_investment_demand_ + 
                              nz_investment_demand_ + 
                              electrification_investment_demand_ + 
                              storage_demand_vec + 
                              grid_upgrade_demand_)
            # --------------------------------------------------------------------------------
            
        else:
            v.Gov_confidence_boost[t] = 0
    else:
        v.Gov_confidence_boost[t] = 0
    

    ########################################################################################################################################################################################################################
    # 9. Government investment expenditure
    ########################################################################################################################################################################################################################
    # Uses domestic-only public investment
    # (FDI-financed portion does not burden the domestic government budget)
    v.GI[t] = np.sum(v.I_public_dom_[t])

    # After the root-cause fix in Section 6 (using np.sum(re_investment_demand_) instead of
    # v.I_RE[t] as the subtractor), the accounting identity holds BY CONSTRUCTION:
    #   sum(I_demand_[t]) = sum(I_[t]) - I_storage_import
    # The only expected gap is the storage-import leakage (battery hardware that routes to IM
    # rather than domestic demand — by design). Everything else should be ~0.
    sum_I = np.sum(v.I_[t])
    sum_Demand = np.sum(v.I_demand_[t])

    gap = sum_Demand - sum_I
    expected_gap = -v.I_storage_import[t]   # storage imports route to IM, not I_demand_
    adjusted_gap = gap - expected_gap        # should be ~0 after Section 6 fix

    ###################################################################################################################################################################################################################################################################################################
