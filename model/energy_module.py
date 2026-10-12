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

import matplotlib.pyplot as plt
from .model_classes import ModelConfig, ModelVariables, ModelResults, ModelParameters
from .calibration import ParametersCalibrated
import numpy as np
import pandas as pd
from dataclasses import fields 
import seaborn as sns 
import warnings

# Water intensities of mitigation (SI Table water_energy_intensities): m3 per GJ, or billion m3 per MtCO2
WI_SOLAR = 0.02
WI_WIND = 0.0
WI_HYDROGEN = 0.15
WI_ELECTRIFICATION = 0.005
WI_CCS_BM3_PER_MT = 0.002
WI_CCU_BM3_PER_MT = 0.0025
WI_SGI_BM3_PER_MT = 0.315


def wastewater_recovery_rate(t: int) -> float:
    """Share of urban water use recovered as treated wastewater: 30% in 2021 (t=1), +2 points per year, at most 80%."""
    return min(0.30 + 0.02 * (t - 1), 0.80)


def wastewater_available_for_nexus(v: ModelVariables, t: int) -> float:
    """
    Reclaimed wastewater available for mitigation in period t (m3): collectable urban wastewater minus
    what the reuse sector already delivers to other users. Based on [t-1], because the water module
    runs after the energy module within a period.
    """
    def urban(i): return v.water_use_hh[i] + v.water_use_ind[i] + v.water_use_ser[i]
    # Both terms are projected one period ahead with their last growth rate
    g_urban = urban(t-1) / urban(t-2) if t > 2 and urban(t-2) > 0 else 1.0
    g_reuse = v.water_use_wwater_tot[t-1] / v.water_use_wwater_tot[t-2] if t > 2 and v.water_use_wwater_tot[t-2] > 0 else 1.0
    return max(0.0, urban(t-1) * g_urban * wastewater_recovery_rate(t) - v.water_use_wwater_tot[t-1] * g_reuse)


def nexus_water_other_m3(v: ModelVariables, pc: ParametersCalibrated, t: int) -> float:
    """Water use of mitigation other than afforestation (m3): solar, wind, hydrogen, electrification, CCS, CCU."""
    elec_load_gj = v.electricity_use_[t][pc.electrification_target_sectors].sum() if hasattr(pc, 'electrification_target_sectors') else 0.0
    return (v.re_gen_solar[t] * WI_SOLAR + v.re_gen_wind[t] * WI_WIND + v.hydrogen_production[t] * WI_HYDROGEN
            + elec_load_gj * WI_ELECTRIFICATION
            + (v.flow_remove_ccs[t] * WI_CCS_BM3_PER_MT + v.flow_reuse_recycle[t] * WI_CCU_BM3_PER_MT) * 1e9)


#########################################################################################################
def energy_module(v: ModelVariables, p: ModelParameters, pc: ParametersCalibrated, t: int):
#################################################################################################################
    
    # ======================================================================
    # --- START ENERGY & EFFICIENCY LOGIC ---
    # ======================================================================

    ########################################################################
    # 1. Update Energy Efficiency Factors (Physical Only)
    ########################################################################
    
    if p.enable_energy_efficiency_gains:
        v.energy_reduction_factor_[t] = v.energy_reduction_factor_[t-1].copy()
        v.energy_reduction_factor_[t][pc.top_20_energy_sectors] += p.efficiency_top_tier_annual_gain
        v.energy_reduction_factor_[t][pc.mid_20_energy_sectors] += p.efficiency_mid_tier_annual_gain
        v.energy_reduction_factor_[t][pc.low_energy_sectors] += p.efficiency_low_tier_annual_gain
        v.energy_reduction_factor_[t] = np.minimum(v.energy_reduction_factor_[t], p.efficiency_max_reduction_pct)
    else:
        v.energy_reduction_factor_[t] = v.energy_reduction_factor_[t-1]
    
    ########################################################################
    # 2. Calculate Energy Demand (Historical Trend Logic)
    ########################################################################
    
    v.gas_use_[t] = v.X_[t] * pc.gas_intensities_ * (1.0 - v.energy_reduction_factor_[t])
    v.electricity_use_[t] = v.X_[t] * pc.electricity_intensities_ * (1.0 - v.energy_reduction_factor_[t])
    v.oil_use_[t] = v.X_[t] * pc.oil_intensities_ * (1.0 - v.energy_reduction_factor_[t])
    
    v.oil_use_final_[t] = v.oil_use_[t].copy()
    v.gas_use_final_[t] = v.gas_use_[t].copy()
    
    elec_idx = pc.electricity_sector_idx
    
    # --- Vision 2030: Targeted Oil-to-Gas Switching ---
    if p.enable_targeted_oil_reduction and p.vision_2030_renewable_target:
        start_year_t = 3 
        target_year_t = 10
        target_reduction_pct = 0.95 
        current_reduction_pct = 0.0
        
        if t > start_year_t: 
            if t >= target_year_t:
                current_reduction_pct = target_reduction_pct
            else:
                years_into_ramp = t - start_year_t 
                total_ramp_duration = target_year_t - start_year_t 
                current_reduction_pct = (years_into_ramp / total_ramp_duration) * target_reduction_pct

        if current_reduction_pct > 0:
            all_target_indices = set(pc.oil_reduction_sectors_idx)
            all_target_indices.discard(elec_idx) 
            target_indices_non_elec = list(all_target_indices)
            
            if target_indices_non_elec: 
                oil_to_reduce_gj = v.oil_use_final_[t][target_indices_non_elec] * current_reduction_pct
                v.oil_use_final_[t][target_indices_non_elec] -= oil_to_reduce_gj
                v.gas_use_final_[t][target_indices_non_elec] += oil_to_reduce_gj
    
    v.gas_use_total[t] = np.sum(v.gas_use_final_[t])
    v.oil_use_total[t] = np.sum(v.oil_use_final_[t])

# ======================================================================
    # --- PART 2: HYDROGEN PRODUCTION & SUBSTITUTION ---
    # ======================================================================
    
    if p.enable_green_hydrogen and t > 1:
        # 1. Determine Production
        v.hydrogen_production[t] = v.k_hydrogen_physical[t] * 0.9 
        
        # 2. Calculate RE Consumption
        v.re_used_for_hydrogen[t] = v.hydrogen_production[t] * p.hydrogen_efficiency
        
        # 3. Cap production logic
        # We assume Renewable Generation (v.renewable_energy_generation) is the TOTAL available.
        available_re = v.renewable_energy_generation[t]
        
        if v.re_used_for_hydrogen[t] > available_re:
            scaling = available_re / v.re_used_for_hydrogen[t]
            v.hydrogen_production[t] *= scaling
            v.re_used_for_hydrogen[t] = available_re
            
        # 4. Split Use
        v.hydrogen_export[t] = v.hydrogen_production[t] * p.hydrogen_export_share
        v.hydrogen_domestic_use[t] = v.hydrogen_production[t] * (1 - p.hydrogen_export_share)

        # 5. Substitution Logic (Replacing Fossil Fuels in Industry)
        if v.hydrogen_domestic_use[t] > 0:
            target_sectors = [
                pc.sectors.index('Manufacture of chemicals and chemical products') if 'Manufacture of chemicals and chemical products' in pc.sectors else -1,
                pc.sectors.index('Manufacture of basic metals') if 'Manufacture of basic metals' in pc.sectors else -1
            ]
            valid_targets = [i for i in target_sectors if i >= 0]
            
            if valid_targets:
                total_target_gas = np.sum(v.gas_use_final_[t][valid_targets])
                if total_target_gas > 0:
                    for i in valid_targets:
                        share = v.gas_use_final_[t][i] / total_target_gas
                        # Calculate substitution (Energy content basis)
                        h2_sub = v.hydrogen_domestic_use[t] * share
                        
                        # Ensure we don't subtract more gas than exists
                        actual_sub = np.minimum(v.gas_use_final_[t][i], h2_sub)
                        v.gas_use_final_[t][i] -= actual_sub
    ########################################################################
    # 3. Determine Electricity Generation Mix
    ########################################################################
    # ======================================================================
    # --- ELECTRIFICATION STRATEGY (Oil/Gas -> Electricity) ---
    # ======================================================================
    if p.enable_industrial_electrification and t > 1:
        target_indices = pc.electrification_target_sectors
        
        # 1. Calculate Cumulative Shift Percentage
        # Logic: Each year we shift 'rate' percent.
        # Share Remaining = (1 - rate)^years
        # Share Shifted = 1 - Share Remaining
        # We use (t-1) because simulation starts at t=1 (Year 0 shift is 0)
        
        rate = p.electrification_annual_rate
        cumulative_shift_pct = 1.0 - (1.0 - rate)**(t - 1)
        
        # 2. Apply Shift to Physical Fuel Use
        if cumulative_shift_pct > 0:
            # A. Calculate amount to shift (from original demand)
            oil_to_shift = v.oil_use_final_[t][target_indices] * cumulative_shift_pct
            gas_to_shift = v.gas_use_final_[t][target_indices] * cumulative_shift_pct
            
            # B. Subtract from Fossil Demand
            v.oil_use_final_[t][target_indices] -= oil_to_shift
            v.gas_use_final_[t][target_indices] -= gas_to_shift
            
            # C. Add to Electricity Demand
            # Assumption: 1 GJ Fossil -> 1 GJ Electricity (Conservative grid load estimate)
            # (Industrial heat pumps often have COP > 2, so this is a conservative upper bound for grid load)
            total_shifted_energy = oil_to_shift + gas_to_shift
            v.electricity_use_[t][target_indices] += total_shifted_energy
            
            # D. Update Totals (so Generation Mix sees new demand)
            v.gas_use_total[t] = np.sum(v.gas_use_final_[t])
            v.oil_use_total[t] = np.sum(v.oil_use_final_[t])
            # Note: v.electricity_use_total will be calculated in the next section based on v.electricity_use_
    if t <= 1: 
        v.oil_use_final_[t][elec_idx] = pc.hist_elec_oil_gj[t-1]
        v.gas_use_final_[t][elec_idx] = pc.hist_elec_gas_gj[t-1]
        v.electricity_use_total[t] = (
            pc.hist_elec_oil_gj[t-1] + pc.hist_elec_gas_gj[t-1] +
            pc.hist_elec_solar_gj[t-1] + pc.hist_elec_wind_gj[t-1] + pc.hist_elec_bio_gj[t-1]
        )
        v.gas_use_total[t] = np.sum(v.gas_use_final_[t])
        v.oil_use_total[t] = np.sum(v.oil_use_final_[t])

    else:
        # --- TREND LOGIC ---
        # Electricity to supply the water for mitigation (part 7) is a load on top of sectoral demand.
        # It is reported in v.electricity_nexus_water_total. With p.nexus_water_electricity_feedback it
        # is also generated for: it is known only at the end of a period, so last period's value is
        # used, and it is kept out of the trend base, where it would accumulate from year to year.
        # Default off: the load is below 0.4% of electricity use, but the perturbation shifts a
        # threshold in the Vision 2030 run around 2040 and moves its 2060 GDP by about 1%.
        nexus_feedback = getattr(p, 'nexus_water_electricity_feedback', False)
        nexus_water_load = v.electricity_nexus_water_total[t-1] if nexus_feedback else 0.0
        sectoral_base = v.electricity_use_total[t-1] - (v.electricity_nexus_water_total[t-2] if nexus_feedback and t > 2 else 0.0)
        target_sectoral_electricity = sectoral_base * (1 + v.Y_growth_rate_forecast[t])
        efficient_demand_total = np.sum(v.electricity_use_[t])
        target_sectoral_electricity = np.minimum(target_sectoral_electricity, efficient_demand_total)
        target_total_electricity_generation = target_sectoral_electricity + nexus_water_load
        v.electricity_use_total[t] = target_total_electricity_generation

        # NET RENEWABLE AVAILABLE FOR ELECTRICITY (Total RE - Used for H2)
        re_gen_net = np.maximum(0, v.renewable_energy_generation[t] - v.re_used_for_hydrogen[t])
        
        re_consumed_by_elec = np.minimum(target_total_electricity_generation, re_gen_net)
        ff_gap_needed = np.maximum(0, target_total_electricity_generation - re_consumed_by_elec)
        
        base_oil_share = pc.ff_oil_share_2023
        base_gas_share = pc.ff_gas_share_2023
        
        if p.net_emission_reduction_green_investments_exports_transformation:
            # We need to transition the FOSSIL GAP MIX.
            
            start_year_t = 3 
            target_year_t = 10 # 2030
            
            if t <= start_year_t:
                current_reduction_pct = 0.0
            elif t <= target_year_t:
                 # Phase 1: Ramp up reduction to 95% by 2030 (Vision 2030)
                target_reduction_pct = 0.95
                years_into_ramp = t - start_year_t
                total_ramp_duration = target_year_t - start_year_t
                ratio = np.maximum(0, years_into_ramp / total_ramp_duration) if total_ramp_duration > 0 else 0
                current_reduction_pct = np.minimum(ratio, 1.0) * target_reduction_pct
            else:
                # Phase 2 (Post-2030): MAINTAIN the 95% reduction at the 2030 level to keep oil near zero.
                current_reduction_pct = 0.95

            # Apply the reduction to the Base Shares (2023 values)
            oil_share_to_reduce = base_oil_share * current_reduction_pct
            new_oil_share = np.maximum(0, base_oil_share - oil_share_to_reduce)
            new_gas_share = base_gas_share + oil_share_to_reduce

        elif p.enable_targeted_oil_reduction and p.vision_2030_renewable_target:
            # Vision 2030 targeted oil-reduction logic
            start_year_t = 3; target_year_t = 10; target_reduction_pct = 0.95; current_reduction_pct = 0.0
            if t > start_year_t:
                if t >= target_year_t: current_reduction_pct = target_reduction_pct
                else: current_reduction_pct = ((t - start_year_t) / (target_year_t - start_year_t)) * target_reduction_pct
            oil_share_to_reduce = base_oil_share * current_reduction_pct
            new_oil_share = base_oil_share - oil_share_to_reduce
            new_gas_share = base_gas_share + oil_share_to_reduce
        else:
            new_oil_share = base_oil_share; new_gas_share = base_gas_share       
        v.oil_use_final_[t][elec_idx] = ff_gap_needed * new_oil_share
        v.gas_use_final_[t][elec_idx] = ff_gap_needed * new_gas_share
        
        v.gas_use_total[t] = np.sum(v.gas_use_final_[t])
        v.oil_use_total[t] = np.sum(v.oil_use_final_[t])
        
        x_based_sectoral_total = np.sum(v.electricity_use_[t])
        if x_based_sectoral_total > 0:
            rescale_factor = target_sectoral_electricity / x_based_sectoral_total
            v.electricity_use_[t] = v.electricity_use_[t] * rescale_factor

    ########################################################################
    # 4. Substitute remaining Fossil Fuels (non-electricity) with remaining RE
    ########################################################################
    
    total_elec_demand_gj = v.electricity_use_total[t]
    # Re-calculate Net RE available (Total - H2 use)
    re_gen_net = np.maximum(0, v.renewable_energy_generation[t] - v.re_used_for_hydrogen[t])
    
    re_consumed_by_elec = np.minimum(total_elec_demand_gj, re_gen_net)
    re_to_substitute = np.maximum(0, re_gen_net - re_consumed_by_elec)

    # Substitute Oil
    oil_demand_non_elec = v.oil_use_total[t] - v.oil_use_final_[t][elec_idx]
    oil_substituted = np.minimum(re_to_substitute, oil_demand_non_elec)
    if oil_demand_non_elec > 0 and oil_substituted > 0:
        oil_reduction_fraction = oil_substituted / oil_demand_non_elec
        for i in range(len(v.oil_use_final_[t])):
            if i != elec_idx: v.oil_use_final_[t][i] *= (1 - oil_reduction_fraction)
    
    re_to_substitute -= oil_substituted

    # Substitute Gas
    gas_demand_non_elec = v.gas_use_total[t] - v.gas_use_final_[t][elec_idx]
    gas_substituted = np.minimum(re_to_substitute, gas_demand_non_elec)
    if gas_demand_non_elec > 0 and gas_substituted > 0:
        gas_reduction_fraction = gas_substituted / gas_demand_non_elec
        for i in range(len(v.gas_use_final_[t])):
            if i != elec_idx: v.gas_use_final_[t][i] *= (1 - gas_reduction_fraction)

    v.gas_use_total[t] = np.sum(v.gas_use_final_[t])
    v.oil_use_total[t] = np.sum(v.oil_use_final_[t])
    v.total_energy_demand[t] = v.gas_use_total[t] + v.electricity_use_total[t] + v.oil_use_total[t]
    v.renewable_energy_share[t] = v.renewable_energy_generation[t] / v.total_energy_demand[t] if v.total_energy_demand[t] > 0 else 0

    ########################################################################
    # 5. Calculate Emissions (Standard Logic)
    ########################################################################
    v.oil_emissions_[t] = v.oil_use_final_[t] * pc.oil_emission_factor
    v.gas_emissions_[t] = v.gas_use_final_[t] * pc.gas_emission_factor

    elec_sector_gas_emissions = v.gas_use_final_[t][elec_idx] * pc.gas_emission_factor
    elec_sector_oil_emissions = v.oil_use_final_[t][elec_idx] * pc.oil_emission_factor
    total_elec_sector_emissions = elec_sector_gas_emissions + elec_sector_oil_emissions
    
    # Updated denominator: Total generation includes (Oil + Gas + RE consumed by elec)
    # Note: re_consumed_by_elec depends on re_gen_net, calculated above
    total_elec_gen_gj = (v.gas_use_final_[t][elec_idx] + v.oil_use_final_[t][elec_idx] + re_consumed_by_elec)
    dynamic_elec_emission_factor = total_elec_sector_emissions / total_elec_gen_gj if total_elec_gen_gj > 0 else 0
    v.electricity_emissions_[t] = v.electricity_use_[t] * dynamic_elec_emission_factor

    v.gas_emissions_total[t] = np.sum(v.gas_emissions_[t])
    v.electricity_emissions_total[t] = np.sum(v.electricity_emissions_[t]) 
    v.oil_emissions_total[t] = np.sum(v.oil_emissions_[t])
    v.total_emissions[t] = v.gas_emissions_total[t] + v.oil_emissions_total[t]
    
# ======================================================================
    # --- NET ZERO LOGIC ---
    # ======================================================================
    v.emissions_gross_total[t] = v.total_emissions[t]

    if p.net_emission_reduction_green_investments_exports_transformation:
            # Net-zero mitigation logic
            # Reuse/Recycle
            ccu_capacity_mt = p.CCU_capacity_baseline * (1 + p.CCU_growth_rate)**(t-1)
            v.flow_reuse_recycle[t] = np.minimum(v.emissions_gross_total[t] * 0.15, ccu_capacity_mt) 

            # CCS
            years_to_target = 14; ccs_annual_add = p.CCS_capacity_target_2035 / years_to_target
            v.flow_remove_ccs[t] = np.minimum(p.CCS_capacity_target_2035, ccs_annual_add * t)
            
            # --------------------------------------------------------------------------------
            # 3. Nature (Saudi Green Initiative) - CAPPED BY WASTEWATER AVAILABILITY
            # --------------------------------------------------------------------------------
            if t <= 40:
                fraction = (t - 1) / 39.0  # 39 years from 2021 to 2060
            else:
                fraction = 1.0
                
            min_sequestration = 0.2 + (20.0 - 0.2) * (fraction ** 2)
            max_sequestration = 2.0 + (100.0 - 2.0) * (fraction ** 2)
            
            intended_sequestration = (min_sequestration + max_sequestration)/2.0
            
            # --- CAP SGI BASED ON AVAILABLE WASTEWATER ---
            # Afforestation gets the reclaimed wastewater left after the reuse sector's deliveries to other
            # users and after the other mitigation uses (solar, wind, H2, electrification, CCS, CCU)
            avail_for_sgi_m3 = max(0.0, wastewater_available_for_nexus(v, t) - nexus_water_other_m3(v, pc, t))
            max_seq_from_water = avail_for_sgi_m3 / (WI_SGI_BM3_PER_MT * 1e9)
            
            # Stop planting if water limit is reached
            v.flow_remove_nature[t] = min(intended_sequestration, max_seq_from_water)
            
            # --------------------------------------------------------------------------------
            # --- TREE PLANTING ACCOUNTING (5-Year Maturity Delay + Min/Max Range) ---
            # --------------------------------------------------------------------------------
            if not hasattr(v, 'sgi_trees_planted_avg'):
                v.sgi_trees_planted_avg = np.zeros(len(v.emissions_gross_total))
                v.sgi_trees_planted_min = np.zeros(len(v.emissions_gross_total))
                v.sgi_trees_planted_max = np.zeros(len(v.emissions_gross_total))
            
            # Function to calculate new trees needed based on previously planted maturing trees
            def get_req_trees(P_array, current_t, target_co2_mt, rate_mt_per_billion):
                existing_co2 = 0.0
                for j in range(0, current_t - 1): # Check all trees planted in prior years
                    age = current_t - j
                    # 5-year growth curve: 20% per year until 100%
                    mult = 0.2 * age if age <= 5 else 1.0
                    existing_co2 += P_array[j] * mult * rate_mt_per_billion
                shortfall = max(0.0, target_co2_mt - existing_co2)
                return shortfall / (0.2 * rate_mt_per_billion) # New trees operate at 20% capacity in year 1

            # Calculate annual plantings required (Avg: 10kg, Min: 12kg, Max: 8kg)
            v.sgi_trees_planted_avg[t-1] = get_req_trees(v.sgi_trees_planted_avg, t, v.flow_remove_nature[t], 10.0)
            v.sgi_trees_planted_min[t-1] = get_req_trees(v.sgi_trees_planted_min, t, v.flow_remove_nature[t], 12.0)
            v.sgi_trees_planted_max[t-1] = get_req_trees(v.sgi_trees_planted_max, t, v.flow_remove_nature[t], 8.0)
            
            # Carry forward the last known planting rate to 't' just for plotting smoothness
            v.sgi_trees_planted_avg[t] = v.sgi_trees_planted_avg[t-1]
            v.sgi_trees_planted_min[t] = v.sgi_trees_planted_min[t-1]
            v.sgi_trees_planted_max[t] = v.sgi_trees_planted_max[t-1]
            # --------------------------------------------------------------------------------
            
            # Net Accounting
            physical_removal = v.flow_reuse_recycle[t] + v.flow_remove_ccs[t] + v.flow_remove_nature[t]
            max_physical_removal = v.emissions_gross_total[t] * 0.95
            if physical_removal > max_physical_removal:
                scaling_factor = max_physical_removal / physical_removal
                v.flow_reuse_recycle[t] *= scaling_factor; v.flow_remove_ccs[t] *= scaling_factor; v.flow_remove_nature[t] *= scaling_factor
                physical_removal = max_physical_removal
            v.emissions_net_total[t] = v.emissions_gross_total[t] - physical_removal
            
            # Carbon Market
            if p.enable_carbon_market:
                demand_potential = v.emissions_net_total[t]
                domestic_supply = v.flow_remove_ccs[t] + v.flow_remove_nature[t]
                start_year_market = 5 
                if t < start_year_market: liquidity_cap = 0.0
                else: liquidity_cap = p.carbon_market_liquidity * (1 + getattr(p, 'carbon_market_growth_rate', 0.113))**(t - start_year_market)
                actual_traded_volume = np.min([demand_potential, liquidity_cap])
                market_value = actual_traded_volume * 1_000_000 * p.carbon_credit_price_floor / 1000
                v.I_carbon_credits_[t] = market_value * pc.dX 
                amount_domestic = np.min([actual_traded_volume, domestic_supply])
                if amount_domestic > 0:
                    value_domestic = amount_domestic * 1_000_000 * p.carbon_credit_price_floor / 1000
                    if domestic_supply > 0:
                        share_ccs = v.flow_remove_ccs[t] / domestic_supply; share_nature = v.flow_remove_nature[t] / domestic_supply
                        oil_idx = pc.sectors.index('Extraction of crude petroleum and natural gas')
                        agri_idx = pc.sectors.index('Agriculture, Forestry & Fishing') if 'Agriculture, Forestry & Fishing' in pc.sectors else pc.agr_sector
                        v.I_carbon_credit_revenue_[t][oil_idx] += value_domestic * share_ccs
                        v.I_carbon_credit_revenue_[t][agri_idx] += value_domestic * share_nature
                v.P_[t] -= v.I_carbon_credits_[t]
                v.P_[t] += v.I_carbon_credit_revenue_[t]

    elif p.vision_2030_target_CCU_SGI_Carbon:
            # --- VISION 2030 SCENARIO ---
            # Strategy: Aggressive deployment of SGI (Nature) and initial CCS/Hydrogen, 
            # but capped to reflect a 2030 milestone rather than full 2060 neutrality.
            
            # 1. Reuse/Recycle (CCU & Hydrogen)
            ccu_growth_moderate = getattr(p, 'CCU_growth_rate', 0.10) * 0.7 
            ccu_capacity_mt = p.CCU_capacity_baseline * (1 + ccu_growth_moderate)**(t-1)
            v.flow_reuse_recycle[t] = np.minimum(v.emissions_gross_total[t] * 0.07, ccu_capacity_mt)
            
            # 2. CCS
            years_to_target = 14
            ccs_annual_add = p.CCS_capacity_target_2035 / years_to_target
            v.flow_remove_ccs[t] = np.minimum(p.CCS_capacity_target_2035 * 0.6, ccs_annual_add * t)
            
            # --------------------------------------------------------------------------------
            # 3. Nature (Saudi Green Initiative) - CAPPED BY WASTEWATER AVAILABILITY
            # --------------------------------------------------------------------------------
            if t <= 40:
                fraction = (t - 1) / 39.0  # 39 years from 2021 to 2060
            else:
                fraction = 1.0
                
            min_sequestration = 0.2 + (1.2 - 0.2) * (fraction ** 2)
            max_sequestration = 2.0 + (6.0 - 2.0) * (fraction ** 2)
            
            intended_sequestration = (min_sequestration + max_sequestration) / 2.0
            
            # --- CAP SGI BASED ON AVAILABLE WASTEWATER ---
            # Afforestation gets the reclaimed wastewater left after the reuse sector's deliveries to other
            # users and after the other mitigation uses (solar, wind, H2, electrification, CCS, CCU)
            avail_for_sgi_m3 = max(0.0, wastewater_available_for_nexus(v, t) - nexus_water_other_m3(v, pc, t))
            max_seq_from_water = avail_for_sgi_m3 / (WI_SGI_BM3_PER_MT * 1e9)
            
            # Stop planting if water limit is reached
            v.flow_remove_nature[t] = min(intended_sequestration, max_seq_from_water)
            
            # --------------------------------------------------------------------------------
            # --- TREE PLANTING ACCOUNTING (5-Year Maturity Delay + Min/Max Range) ---
            # --------------------------------------------------------------------------------
            if not hasattr(v, 'sgi_trees_planted_avg'):
                v.sgi_trees_planted_avg = np.zeros(len(v.emissions_gross_total))
                v.sgi_trees_planted_min = np.zeros(len(v.emissions_gross_total))
                v.sgi_trees_planted_max = np.zeros(len(v.emissions_gross_total))
            
            def get_req_trees(P_array, current_t, target_co2_mt, rate_mt_per_billion):
                existing_co2 = 0.0
                for j in range(0, current_t - 1):
                    age = current_t - j
                    mult = 0.2 * age if age <= 5 else 1.0
                    existing_co2 += P_array[j] * mult * rate_mt_per_billion
                shortfall = max(0.0, target_co2_mt - existing_co2)
                return shortfall / (0.2 * rate_mt_per_billion)

            v.sgi_trees_planted_avg[t-1] = get_req_trees(v.sgi_trees_planted_avg, t, v.flow_remove_nature[t], 10.0)
            v.sgi_trees_planted_min[t-1] = get_req_trees(v.sgi_trees_planted_min, t, v.flow_remove_nature[t], 12.0)
            v.sgi_trees_planted_max[t-1] = get_req_trees(v.sgi_trees_planted_max, t, v.flow_remove_nature[t], 8.0)
            
            v.sgi_trees_planted_avg[t] = v.sgi_trees_planted_avg[t-1]
            v.sgi_trees_planted_min[t] = v.sgi_trees_planted_min[t-1]
            v.sgi_trees_planted_max[t] = v.sgi_trees_planted_max[t-1]
            # --------------------------------------------------------------------------------

            # 4. Net Accounting & Scaling
            # --------------------------------------------------------------------------------
            physical_removal = v.flow_reuse_recycle[t] + v.flow_remove_ccs[t] + v.flow_remove_nature[t]
            
            max_physical_removal = v.emissions_gross_total[t] * 0.95
            if physical_removal > max_physical_removal:
                scaling_factor = max_physical_removal / physical_removal
                v.flow_reuse_recycle[t] *= scaling_factor
                v.flow_remove_ccs[t] *= scaling_factor
                v.flow_remove_nature[t] *= scaling_factor
                physical_removal = max_physical_removal
                
            v.emissions_net_total[t] = v.emissions_gross_total[t] - physical_removal
            
            # Carbon Market (Restricted)
            if p.enable_carbon_market:
                demand_potential = v.emissions_net_total[t]
                domestic_supply = v.flow_remove_ccs[t] + v.flow_remove_nature[t]
                
                start_year_market = 6 # Market matures slightly later/slower
                if t < start_year_market: 
                    liquidity_cap = 0.0
                else: 
                    # Liquidity is 50% tighter than Net Zero scenario
                    liquidity_cap = (p.carbon_market_liquidity * 0.5) * (1 + getattr(p, 'carbon_market_growth_rate', 0.113))**(t - start_year_market)
                
                # HARD LIMIT: Market cannot cover more than 20% of remaining emissions 
                # (Forces real abatement rather than just offsets)
                max_market_coverage = demand_potential * 0.20
                
                actual_traded_volume = np.min([demand_potential, liquidity_cap, max_market_coverage])
                market_value = actual_traded_volume * 1_000_000 * p.carbon_credit_price_floor / 1000
                v.I_carbon_credits_[t] = market_value * pc.dX 
                
                # Revenue Recyling
                amount_domestic = np.min([actual_traded_volume, domestic_supply])
                if amount_domestic > 0:
                    value_domestic = amount_domestic * 1_000_000 * p.carbon_credit_price_floor / 1000
                    if domestic_supply > 0:
                        share_ccs = v.flow_remove_ccs[t] / domestic_supply
                        share_nature = v.flow_remove_nature[t] / domestic_supply
                        oil_idx = pc.sectors.index('Extraction of crude petroleum and natural gas')
                        agri_idx = pc.sectors.index('Agriculture, Forestry & Fishing') if 'Agriculture, Forestry & Fishing' in pc.sectors else pc.agr_sector
                        
                        # Check indices to prevent out of bounds
                        if oil_idx < len(v.I_carbon_credit_revenue_[t]):
                            v.I_carbon_credit_revenue_[t][oil_idx] += value_domestic * share_ccs
                        if agri_idx < len(v.I_carbon_credit_revenue_[t]):
                            v.I_carbon_credit_revenue_[t][agri_idx] += value_domestic * share_nature
                
                v.P_[t] -= v.I_carbon_credits_[t]
                v.P_[t] += v.I_carbon_credit_revenue_[t]

    else:
            # Business as Usual (BAU)
            v.emissions_net_total[t] = v.emissions_gross_total[t]
            
            # --- Prevent carry-over of mitigation flows from a previous run ---
            # Force all advanced mitigation physical flows to 0.0 in Baseline
            v.flow_remove_nature[t] = 0.0
            v.flow_remove_ccs[t] = 0.0
            v.flow_reuse_recycle[t] = 0.0
            
            if not hasattr(v, 'sgi_trees_planted_avg'):
                v.sgi_trees_planted_avg = np.zeros(len(v.emissions_gross_total))
                v.sgi_trees_planted_min = np.zeros(len(v.emissions_gross_total))
                v.sgi_trees_planted_max = np.zeros(len(v.emissions_gross_total))
    # ======================================================================
    # --- PART 6: WATER-ENERGY NEXUS (Water for Energy & Mitigation) ---
    # ======================================================================
    # Quantify the water demand for energy generation, green technologies, 
    # and carbon sequestration (SGI) based on literature intensities.
    
    # Water demand by technology (identical intensities in all scenarios)
    total_nexus_water_m3 = nexus_water_other_m3(v, pc, t) + v.flow_remove_nature[t] * WI_SGI_BM3_PER_MT * 1e9
    v.water_use_nexus[t] = total_nexus_water_m3

    # Source allocation. With mitigation policy, reclaimed wastewater not delivered to other users is
    # used up to its availability; a shortfall is met by desalination (70%) and groundwater (30%).
    is_sust_or_v2030 = (getattr(p, 'vision_2030_target_CCU_SGI_Carbon', False)
                        or getattr(p, 'Transformation_scenario', False))
    if is_sust_or_v2030:
        v.water_energy_wwater[t] = min(total_nexus_water_m3, wastewater_available_for_nexus(v, t))
    else:
        v.water_energy_wwater[t] = 0.0
    shortfall_m3 = total_nexus_water_m3 - v.water_energy_wwater[t]
    v.water_energy_desal[t] = shortfall_m3 * 0.70
    v.water_energy_gw[t] = shortfall_m3 * 0.30
    # The national water balance (water_module.calculate_water_demand, later in the period) adds
    # v.water_use_nexus[t] to total water use; nothing is written to the total here.
# ======================================================================
    # --- PART 7: ENERGY-WATER NEXUS (Energy Demand for Water Supply) ---
    # ======================================================================
    # Quantify the electricity required to treat and pump the water used by 
    # the energy and mitigation sectors (SGI, Green H2, CCS, etc.)
    
    # 1. Define Energy Intensities for Water Supply (Literature-based)
    
    # Desalination (Reverse Osmosis in the GCC): ~3.5 kWh / m3
    # Reference: Siddiqi & Anadon (2011) "The water-energy nexus in Middle East and North Africa"
    # Reference: IRENA (2019) "Water Use in Saudi Arabia's Energy Sector"
    ei_desal_kwh_per_m3 = 3.5 
    
    # Groundwater Abstraction (Deep aquifer pumping in Saudi Arabia): ~1.0 kWh / m3
    # Reference: Qureshi (2014) "Water resources in the GCC countries: An overview"
    ei_gw_kwh_per_m3 = 1.0 
    
    # Wastewater Treatment (0.8 kWh/m3) + Long-Distance Pumping to remote SGI sites (1.5 kWh/m3)
    # Reference: Plappally & Lienhard (2012) for treatment; GCC pipeline transport averages for pumping.
    ei_wwater_kwh_per_m3 = 0.8 + 1.5  # Total: 2.3 kWh / m3    
    # Conversion factor: 1 kWh = 0.0036 GJ
    kwh_to_gj = 0.0036
    
    # 2. Calculate Electricity Demand for Nexus Water (in GJ)
    # Using the actual cubic meters (m3) calculated in Part 6
    if hasattr(v, 'water_energy_desal') and hasattr(v, 'water_energy_gw') and hasattr(v, 'water_energy_wwater'):
        energy_for_nexus_desal_gj = (v.water_energy_desal[t] * ei_desal_kwh_per_m3) * kwh_to_gj
        energy_for_nexus_gw_gj = (v.water_energy_gw[t] * ei_gw_kwh_per_m3) * kwh_to_gj
        energy_for_nexus_wwater_gj = (v.water_energy_wwater[t] * ei_wwater_kwh_per_m3) * kwh_to_gj
        
        # Total added electricity demand to support the green transition's water needs
        total_nexus_electricity_added_gj = energy_for_nexus_desal_gj + energy_for_nexus_gw_gj + energy_for_nexus_wwater_gj
        
        # Track this specific Nexus electricity demand (useful for plotting later)
        if hasattr(v, 'electricity_nexus_water_total'):
            v.electricity_nexus_water_total[t] = total_nexus_electricity_added_gj
            
        # 3. Close the loop (optional, p.nexus_water_electricity_feedback): this load enters national
        # electricity generation, and with it fuel use and emissions, in the next period (section 3)
            
        # Optional: if a specific water sector index is available in the 40-sector array,
        # the demand can be added directly to that sector so the economy pays for it.
    return v.I_[t],v.I_demand_[t],v.gas_use_[t],v.oil_use_final_[t],v.gas_use_final_[t], v.electricity_use_[t], v.oil_use_[t], v.energy_reduction_factor_[t],\
            v.gas_use_total[t],v.oil_use_total[t],v.electricity_use_total[t], v.renewable_energy_generation[t],\
            v.I_efficiency_[t],v.I_netzero_nature_[t],v.emissions_net_total[t],v.emissions_gross_total[t],v.I_carbon_credit_revenue_[t],\
            v.I_carbon_credits_[t], v.flow_reuse_recycle[t],v.flow_remove_ccs[t],v.flow_remove_nature[t],v.marginal_cost_of_efficiency_[t]
