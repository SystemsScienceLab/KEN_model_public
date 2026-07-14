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
Energy Investment Module
========================

Computes all energy-related investments: renewable energy (solar, wind, bio),
industrial electrification, hydrogen, net-zero (CCS, nature-based, CCU),
energy efficiency, and battery storage.

This module was extracted from investment_module.py for code clarity.
It is called at the beginning of investment_module() before desalination,
wastewater, and general sectoral investment distribution.

Variables set by this module (on v, in-place):
  - electrification_added_load[t]
  - k_re_physical[t], renewable_energy_generation[t]
  - I_solar[t], I_wind[t], I_bio[t], I_RE[t]
  - re_gen_solar[t], re_gen_wind[t], re_gen_bioenergy[t]
  - I_hydrogen[t], I_hydrogen_[t]
  - I_netzero_ccs_[t], I_netzero_nature_[t], I_netzero_ccu_[t]
  - I_efficiency_total[t], marginal_cost_of_efficiency_[t]
  - I_electrification_total[t]
  - I_storage[t], I_storage_local[t], I_storage_import[t]
  - k_storage_physical[t], storage_coverage_pct[t], storage_local_share[t]
  - I_[t][electricity_gas_AC_sector]  (RE investment assigned to electricity sector)
"""

import numpy as np
from .model_classes import ModelVariables, ModelParameters
from .calibration import ParametersCalibrated

def energy_investment_module(v: ModelVariables, p: ModelParameters, pc: ParametersCalibrated, t: int):
    """
    Compute energy-related investments for time step t.

    Must be called at the start of investment_module(), before desalination,
    wastewater, and general sectoral investment logic.

    Parameters
    ----------
    v  : ModelVariables        — model state arrays (modified in-place)
    p  : ModelParameters       — scenario parameters
    pc : ParametersCalibrated  — calibrated parameters
    t  : int                   — current time step (1-indexed, t=1 is 2021)
    """

    ########################################################################################################################################################################################################################
    # 1. Industrial electrification load
    ########################################################################################################################################################################################################################
    electrification_load_gj = 0.0
    
    if p.enable_industrial_electrification and t > 1:
        # 1. Identify Target Sectors
        target_indices = pc.electrification_target_sectors
        
        # 2. Estimate Fossil Energy Use (Pool)
        estimated_fossil_use_gj = (
            pc.gas_intensities_[target_indices] + 
            pc.oil_intensities_[target_indices]
        ) * v.X_[t-1][target_indices]
        
        # 3. Calculate Fossil Energy Removed THIS YEAR (Incremental)
        # Logic: Current fossil demand * Annual Rate
        fossil_removed_gj = np.sum(estimated_fossil_use_gj * p.electrification_annual_rate)
        
        # 4. Convert to Electricity Needed (Using COP)
        # 1 GJ Gas -> (1/COP) GJ Electricity
        cop = getattr(p, 'electrification_cop', 2.5) 
        electrification_load_gj = fossil_removed_gj / cop
        
        # Store for use in other modules
        v.electrification_added_load[t] = electrification_load_gj

    ########################################################################################################################################################################################################################
    # 2. Renewable energy shares and investment
    ########################################################################################################################################################################################################################
    current_solar_share = 0.0
    current_wind_share = 0.0
    current_bio_share = 0.0

    if p.enable_dynamic_re_mix and t > 3:
        # Scenario: Dynamic Mix (Ramp from 2023 levels to 2030 Targets)
        target_year_idx = 10 # 2030
        
        if t <= target_year_idx:
            progress = (t - 3) / (target_year_idx - 3)
            current_solar_share = pc.re_solar_share + (p.target_solar_share_2030 - pc.re_solar_share) * progress
            current_wind_share = pc.re_wind_share + (p.target_wind_share_2030 - pc.re_wind_share) * progress
            current_bio_share = pc.re_bio_share + (p.target_bio_share_2030 - pc.re_bio_share) * progress
        else:
            current_solar_share = p.target_solar_share_2030
            current_wind_share = p.target_wind_share_2030
            current_bio_share = p.target_bio_share_2030
            
        # Normalization
        total_share = current_solar_share + current_wind_share + current_bio_share
        if total_share > 0:
            current_solar_share /= total_share
            current_wind_share /= total_share
            current_bio_share /= total_share
    else:
        # Scenario: Fixed Shares
        current_solar_share = pc.re_solar_share
        current_wind_share = pc.re_wind_share
        current_bio_share = pc.re_bio_share

    ########################################################################################################################################################################################################################
    # 2b. Required renewable energy investment (segregated by technology)
    ########################################################################################################################################################################################################################
    
    if t <= 1:
        # --- HISTORICAL OVERRIDE (t=1: 2021) ---
        target_RE_generation = (
            pc.hist_elec_solar_gj[t-1] +  
            pc.hist_elec_wind_gj[t-1] +   
            pc.hist_elec_bio_gj[t-1]
        )
        
        v.k_re_physical[t] = target_RE_generation
        v.renewable_energy_generation[t] = target_RE_generation

        depreciated_capacity = v.k_re_physical[t-1] * (1 - pc.delta_RE)
        new_capacity_needed = np.maximum(0, v.k_re_physical[t] - depreciated_capacity)
        
        # SEGREGATED CALCULATION (Based on historical data split)
        hist_total = target_RE_generation if target_RE_generation > 0 else 1.0
        
        # Calculate specific investment for each tech
        v.I_solar[t] = (new_capacity_needed * (pc.hist_elec_solar_gj[t-1]/hist_total)) * pc.capex_solar_per_gj
        v.I_wind[t]  = (new_capacity_needed * (pc.hist_elec_wind_gj[t-1]/hist_total)) * pc.capex_wind_per_gj
        v.I_bio[t]   = (new_capacity_needed * (pc.hist_elec_bio_gj[t-1]/hist_total)) * pc.capex_bio_per_gj
        
        v.I_RE[t] = v.I_solar[t] + v.I_wind[t] + v.I_bio[t]

    else:
        # --- POST-2023 FORECAST ---
        
        # A. Calculate Base Grid Demand
        base_grid_demand = v.electricity_use_total[t-1]
        
        # B. Calculate Baseline RE Target
        if p.net_emission_reduction_green_investments_exports_transformation:
            # --- NET ZERO LOGIC (Hybrid V2030 + Long Term) ---
            v2030_idx = 10  # Year 2030 corresponds to t=10 (approx)
            nz_target_idx = 40 # Year 2060
            final_re_share = getattr(p, 'target_re_share_2060', 0.85)  # Default 85%; set to 1.0 in Net Zero 2060
            
            # 1. Phase 1: Follow Vision 2030 Ramp exactly first
            ramp_length = len(pc.re_target_share_ramp)
            
            if t <= v2030_idx:
                # Use the calibrated V2030 ramp
                ramp_index = min(t - 1, ramp_length - 1)
                current_target_share = pc.re_target_share_ramp[ramp_index]
                
            # 2. Phase 2: Grow from V2030 level to 85% by 2060
            else:
                # Get the share achieved in 2030
                share_in_2030 = pc.re_target_share_ramp[-1] # Likely 0.50
                
                # Calculate progress between 2030 and 2060
                years_past_2030 = t - v2030_idx
                years_to_go = nz_target_idx - v2030_idx
                fraction = years_past_2030 / years_to_go
                
                # Interpolate
                current_target_share = share_in_2030 + (final_re_share - share_in_2030) * fraction
                
                # Cap at final target
                current_target_share = np.minimum(current_target_share, final_re_share)

            target_RE_generation = base_grid_demand * current_target_share

        elif p.vision_2030_renewable_target:
            # Standard Vision 2030 Logic (stays at 50% after 2030)
            ramp_index = t - 1 
            ramp_length = len(pc.re_target_share_ramp)
            if ramp_index >= ramp_length:
                current_target_share = np.minimum(pc.re_target_share_ramp[-1], 0.95)
            else:
                current_target_share = pc.re_target_share_ramp[ramp_index]
            
            target_RE_generation = base_grid_demand * current_target_share
        else:
            target_RE_generation = v.renewable_energy_generation[t-1] * (1 + p.gRE_trend_rate)
        # C. Add Electrification Load
        if p.enable_industrial_electrification:
            target_RE_generation += v.electrification_added_load[t]

        # D. Calculate Investment
        depreciated_capacity = v.k_re_physical[t-1] * (1 - pc.delta_RE)
        new_capacity_needed = np.maximum(0, target_RE_generation - depreciated_capacity)
        
        v.k_re_physical[t] = depreciated_capacity + new_capacity_needed
        v.renewable_energy_generation[t] = v.k_re_physical[t]    

        cost_factor = 1.0
        
        # Check if enabled AND if the share from LAST YEAR met the threshold
        # (We use t-1 because current year's share isn't calculated yet)
        current_re_share_actual = v.renewable_energy_share[t-1]
        
        if p.enable_re_cost_learning and current_re_share_actual >= p.re_cost_learning_threshold:
            # The "Learning Curve" is unlocked
            # Cost(t) = Cost(base) * (1 - rate)^t
            cost_factor = (1.0 - p.re_cost_learning_rate) ** t

        # Calculate adjusted CapEx per GJ
        adjusted_solar_capex = pc.capex_solar_per_gj * cost_factor
        adjusted_wind_capex  = pc.capex_wind_per_gj * cost_factor
        adjusted_bio_capex   = pc.capex_bio_per_gj * cost_factor

        # SEGREGATED CALCULATION (Based on current shares)
        v.I_solar[t] = (new_capacity_needed * current_solar_share) * adjusted_solar_capex
        v.I_wind[t]  = (new_capacity_needed * current_wind_share) * adjusted_wind_capex
        v.I_bio[t]   = (new_capacity_needed * current_bio_share) * adjusted_bio_capex
        
        v.I_RE[t] = v.I_solar[t] + v.I_wind[t] + v.I_bio[t]

    # --- 3. Apply Shares to Generation Output ---
    v.re_gen_solar[t] = v.renewable_energy_generation[t] * current_solar_share
    v.re_gen_wind[t] = v.renewable_energy_generation[t] * current_wind_share
    v.re_gen_bioenergy[t] = v.renewable_energy_generation[t] * current_bio_share
    
    ########################################################################################################################################################################################################################
    # 3. Hydrogen investment 
    ########################################################################################################################################################################################################################    
    if p.enable_green_hydrogen:
        if t <= 1:
            v.k_hydrogen_physical[t] = 0
            v.I_hydrogen[t] = 0
        else:
            # 1. Define Target Capacity (Ramp up to 2030)
            target_year_idx = 10 
            if t <= target_year_idx:
                fraction = (t - 1) / (target_year_idx - 1)
                target_capacity = p.hydrogen_target_2030_gj * fraction
            else:
                target_capacity = v.k_hydrogen_physical[t-1] * 1.05 
            
            # 2. Calculate Investment (Depreciation + New Capacity)
            delta_h2 = 0.05 
            depreciated_capacity = v.k_hydrogen_physical[t-1] * (1 - delta_h2)
            new_capacity_needed = np.maximum(0, target_capacity - depreciated_capacity)
            
            v.k_hydrogen_physical[t] = depreciated_capacity + new_capacity_needed
            
            # 3. Calculate total scalar cost ('000 SAR)
            v.I_hydrogen[t] = (new_capacity_needed * p.electrolyzer_capex_per_gj) / 1000.0
            
    else:
        v.k_hydrogen_physical[t] = 0
        v.I_hydrogen[t] = 0
    ########################################################################################################################################################################################################################
    # 4. Net zero investments (CCS, Nature-based, CCU)
    ########################################################################################################################################################################################################################
    if p.net_emission_reduction_green_investments_exports_transformation:

        # --- A. CCS INVESTMENT (Hard-to-Abate Industry) ---
        ccs_target_year = 15 # Target year index (e.g., 2035)
        
        if t <= ccs_target_year:
            # Linear ramp up to target
            ccs_capacity_needed = (p.CCS_capacity_target_2035 / ccs_target_year) * t
        else:
            # Maintain capacity (or slight growth)
            ccs_capacity_needed = p.CCS_capacity_target_2035
            
        # Calculate annual addition (Delta)
        ccs_prev = (p.CCS_capacity_target_2035 / ccs_target_year) * (t-1) if t > 1 else 0
        ccs_added = np.maximum(0, ccs_capacity_needed - ccs_prev)
        
        # COST CALCULATION:
        # Capacity Added (Mt) * Cost (SAR/tonne) * 1000 -> Thousand SAR
        # Note: We multiply by 1000 because model units are usually '000 SAR
        cost_ccs = ccs_added * p.CCS_cost_per_tonne * 1000.0
        
        # Assign to Chemicals Sector (Proxy for Heavy Industry)
        chem_idx = pc.sectors.index('Manufacture of chemicals and chemical products') if 'Manufacture of chemicals and chemical products' in pc.sectors else 0
        v.I_netzero_ccs_[t][chem_idx] = cost_ccs

# --- B. NATURE INVESTMENT (SGI - Afforestation & Water Infrastructure) ---
        # While physical sequestration is random, financial investment is based 
        # on the EXPECTED (Mean) planting capacity trajectory.
        
        def get_expected_nature_cap(time_step):
            if time_step <= 1:
                return 1.0 # 1 Mt Mean in 2021
            elif time_step <= 40:
                fraction = (time_step - 1) / 39.0
                # Quadratic expansion from 1.0 Mt (2021) to 60.0 Mt Mean (2060)
                return 1.0 + (60.0 - 1.0) * (fraction ** 2)
            else:
                return 60.0 # Cap maintains after 2060

        nature_cap_needed = get_expected_nature_cap(t)
        nature_prev = get_expected_nature_cap(t - 1) if t > 1 else 0.0
        nature_added = np.maximum(0, nature_cap_needed - nature_prev)
        
        # ----------------------------------------------------------------------
        # 1. Tree Planting Cost (Assigned to Agriculture)
        # ----------------------------------------------------------------------
        nature_cost_factor = getattr(p, 'nature_cost_per_tonne', 50.0)
        cost_nature_trees = nature_added * nature_cost_factor * 1000.0 # '000 SAR
        
        agri_idx = pc.sectors.index('Agriculture, Forestry & Fishing') if 'Agriculture, Forestry & Fishing' in pc.sectors else pc.agr_sector
        v.I_netzero_nature_[t][agri_idx] = cost_nature_trees

        # ----------------------------------------------------------------------
        # 2. SGI Wastewater Infrastructure CapEx (Assigned to Construction)
        # ----------------------------------------------------------------------
        # 1 Mt of CO2 sequestration requires 0.315 Billion m3 of water per year.
        water_capacity_added_m3 = nature_added * 0.315 * 1e9
        
        # CapEx for advanced tertiary treatment + long-distance desert pipeline transport
        # Conservative GCC benchmark: ~18 SAR per m3 of annual capacity
        capex_ww_sgi_per_m3 = getattr(p, 'capex_wwater_sgi_per_m3', 18.0) 
        
        cost_wwater_sgi = (water_capacity_added_m3 * capex_ww_sgi_per_m3) / 1000.0 # '000 SAR
        
        # Assign to Civil Engineering or Construction Sector
        try:
            construction_idx = pc.sectors.index('Civil engineering')
        except ValueError:
            construction_idx = pc.sectors.index('Construction of buildings') if 'Construction of buildings' in pc.sectors else 0
            
        # Use += so that overlapping indices accumulate rather than overwrite
        v.I_netzero_nature_[t][construction_idx] += cost_wwater_sgi

        # --- C. CCU INVESTMENT (Synthetic Fuels / Chemicals) ---

        # 1. Determine Capacity Needed
        # We rely on the physical flow calculated in energy_module (MtCO2e)
        ccu_capacity_needed = v.flow_reuse_recycle[t]
        
        # 2. Determine Capacity Added (Delta)
        # We assume new investment is needed for incremental capacity
        ccu_prev = v.flow_reuse_recycle[t-1] if t > 1 else 0
        ccu_added = np.maximum(0, ccu_capacity_needed - ccu_prev)
        
        # 3. COST CALCULATION
        # CCU is expensive (Conversion energy + equipment).
        # Default Assumption: ~200 USD/tonne (~750 SAR/tonne) if parameter is missing.
        ccu_cost_factor = getattr(p, 'CCU_cost_per_tonne', 750.0)
        cost_ccu = ccu_added * ccu_cost_factor * 1000.0 # Convert to Thousand SAR
        
        # 4. Assign to Sector
        # CCU (Methanol, Synfuels, Urea) fits best in "Manufacture of chemicals"
        chem_idx = pc.sectors.index('Manufacture of chemicals and chemical products') if 'Manufacture of chemicals and chemical products' in pc.sectors else 0
        
        # Store in variable (I_netzero_ccu_ must be defined in ModelVariables)
        if hasattr(v, 'I_netzero_ccu_'):
             v.I_netzero_ccu_[t][chem_idx] = cost_ccu

    ########################################################################################################################################################################################################################
    # 5. Energy efficiency investment
    ########################################################################################################################################################################################################################
    if p.enable_energy_efficiency_gains:
        # 1. Determine the Targeted Efficiency Gain % for this year (Sectors)
        # This matches the logic that WILL happen in energy_module later
        efficiency_gains = np.zeros(pc.S)
        
        # Assign gains based on tiers (defined in calibration)
        efficiency_gains[pc.top_20_energy_sectors] = p.efficiency_top_tier_annual_gain
        efficiency_gains[pc.mid_20_energy_sectors] = p.efficiency_mid_tier_annual_gain
        efficiency_gains[pc.low_energy_sectors] = p.efficiency_low_tier_annual_gain
        
        # 2. Estimate Energy Consumption (GJ) from previous period
        # (We use t-1 because we haven't calculated t yet, standard for investment planning)
        estimated_energy_use_gj = (
            pc.gas_intensities_ + 
            pc.electricity_intensities_ + 
            pc.oil_intensities_
        ) * v.X_[t-1]
        
        # 3. Calculate Energy to be Saved (GJ)
        energy_saved_gj = estimated_energy_use_gj * efficiency_gains
        
        # 4. Calculate Investment Cost (Thousand SAR)
        # Cost = Energy Saved (GJ) * Base Capex Cost (SAR/GJ)
        # Note: pc.base_efficiency_capex_ is calibrated in calibration.py
        v.I_efficiency_[t] =  energy_saved_gj * pc.base_efficiency_capex_
        
        # 5. Sum to Total
        v.I_efficiency_total[t] = np.sum(v. I_efficiency_[t])
        
        # Store the marginal cost of efficiency for later analysis
        if t > 1:
            v.marginal_cost_of_efficiency_[t] = np.mean(pc.base_efficiency_capex_)
            
    else:
        v.I_efficiency_total[t] = 0.0

    ########################################################################################################################################################################################################################
    # 6. Electrification investment (industrial equipment cost)
    ########################################################################################################################################################################################################################
    # We calculated the physical load at the top of the function. 
    # Now we calculate the cost of the industrial equipment (not the grid).
    
    if p.enable_industrial_electrification and t > 1:
        # Re-calculate the GJ switched to apply the cost factor
        # Load (Elec) * COP = Original Fossil Energy Switched
        cop = getattr(p, 'electrification_cop', 2.5)
        energy_switched_gj = v.electrification_added_load[t] * cop
        
        # Calculate Cost
        v.I_electrification_total[t] = (energy_switched_gj * p.electrification_cost_per_gj) / 1000.0
    else:
        v.I_electrification_total[t] = 0.0

    ########################################################################################################################################################################################################################
    # 7. Battery storage investment
    ########################################################################################################################################################################################################################
    # Logic: 14h for Solar, 4h for Wind
    # Timeline: Start 2025 -> 10% Coverage by 2030 -> 100% Coverage by 2060
    # Localization: 100% Import initially -> Local production starts after 2030
    
    if p.net_emission_reduction_green_investments_exports_transformation and t > 1:
        # 1. Calculate Target Capacity (Physical GJ)
        # ----------------------------------------------------
        # Convert Annual Gen (GJ) to Daily Avg (GJ/day)
        daily_solar_gj = v.re_gen_solar[t] / 365.0
        daily_wind_gj = v.re_gen_wind[t] / 365.0
        
        # Requirement: configurable hours per day for Solar and Wind (defaults: 14h solar, 4h wind)
        # Increase storage_hours_solar/wind in Net Zero 2060 to buffer higher RE variability
        _solar_h = getattr(p, 'storage_hours_solar', 14)
        _wind_h  = getattr(p, 'storage_hours_wind', 4)
        req_storage_solar = daily_solar_gj * (_solar_h / 24)
        req_storage_wind  = daily_wind_gj  * (_wind_h  / 24)
        
        total_storage_needed_gj = req_storage_solar + req_storage_wind
        
        # 2. Apply Policy Ramp-up (Coverage %)
        # ----------------------------------------------------
        current_year = 2021 + (t - 1)
        
        if current_year < 2025:
            coverage = 0.0
        elif current_year <= 2030:
            # Ramp 0% -> 10%
            years_done = current_year - 2025
            coverage = 0.0 + (0.10 * (years_done / 5.0))
        elif current_year <= 2060:
            # Ramp 10% -> 100%
            years_done = current_year - 2030
            coverage = 0.10 + (0.90 * (years_done / 30.0))
        else:
            coverage = 1.0
            
        v.storage_coverage_pct[t] = coverage
        target_capacity_gj = total_storage_needed_gj * coverage
        
        # 3. Calculate Investment (Delta Capacity)
        # ----------------------------------------------------
        prev_capacity = v.k_storage_physical[t-1]
        capacity_added = np.maximum(0, target_capacity_gj - prev_capacity)
        v.k_storage_physical[t] = prev_capacity + capacity_added
        
        # Cost calculation
        # 1 GJ = 277.778 kWh
        kwh_per_gj = 277.778
        
        # Pricing: 375 SAR/kWh (2025) -> 37.5 SAR/kWh (2040)
        cost_2025 = 375.0 # ~100 USD
        cost_2040 = 37.5  # ~10 USD
        
        if current_year <= 2025:
            current_cost_kwh = cost_2025
        elif current_year >= 2040:
            current_cost_kwh = cost_2040
        else:
            # Linear Interpolation
            fraction = (current_year - 2025) / (2040 - 2025)
            current_cost_kwh = cost_2025 - (fraction * (cost_2025 - cost_2040))
            
        # Convert to SAR/GJ
        current_cost_gj = current_cost_kwh * kwh_per_gj
        
        # Investment in '000 SAR (standard model unit)
        v.I_storage[t] = (capacity_added * current_cost_gj) / 1000.0
        
        # 4. Localization Strategy (Import vs Local)
        # ----------------------------------------------------
        # "Initial import 100%, after reaching 10% coverage (2030), start producing inhouse until 100%"
        
        if coverage < 0.10:
            local_share = 0.0 # 100% Import
        else:
            # Ramp local share from 0% (2030) to 100% (2050 - aggressive industrialization)
            # Assumption: Takes 20 years to fully localize supply chain
            years_since_start = current_year - 2030
            local_share = np.minimum(1.0, years_since_start / 20.0)
            
        v.storage_local_share[t] = local_share
        
        v.I_storage_local[t] = v.I_storage[t] * local_share
        v.I_storage_import[t] = v.I_storage[t] * (1 - local_share)
        
        # 5. Integration
        # Local -> Machinery Sector (Domestic Demand)
        # Import -> Machinery Sector (But flows to Imports)
        mach_idx = pc.sectors.index('Manufacture of machinery and equipment n.e.c.') if 'Manufacture of machinery and equipment n.e.c.' in pc.sectors else 5
        
        # We add the LOCAL part to the specific demand vector later
    else:
        v.I_storage[t] = 0.0

