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
import numpy.typing as npt
from dataclasses import dataclass
import pandas as pd
from typing import Dict
from .calibration import ParametersCalibrated
NDArray = npt.NDArray[np.float64]
scenario_config = {
    'Baseline_scenario':                True,
    'Vision_2030_scenario':             False,
    'Transformation_scenario':          False,
    'Zero_growth_rates_steady_state':   False,
}

@dataclass
class ModelParameters:
    """Manual input parameters of the model."""
    Warnings_ON: bool                       # Toggle between showing warnings or not
    Y_timeseries_model: bool                # Toggle between Calculating expected growth rates of GDP Output Y from time series or from final demand growth
    Inflation_timeseries_model: bool        # Toggle between Calculating expected inflation from time series or from average past inflation
    Limitational_nominal_production: bool           # Toggle between limitational Nominal or Real production
    Limitational_real_production: bool              # Toggle between limitational Nominal or Real production and not
    Productivity_Coefficients_X_adjustment: bool  # Toggle between adjusting productivity coefficients to X or Y
    Endogenous_A_matrix: bool               # Toggle between endogenous A (Leontief production) matrix and fixed A matrix
    Update_productivity_coefficients: bool  # Toggle between updating the A matrix to changes in the production structure or not
    Endogenous_investment: bool             # Toggle between endogenous investment and investment growing with fixed rate
    Endogenous_investment_reduction: bool    # Toggle between reducing investment in case of investment crosses a certain threshold, or not
    investment_reduction_factor: float      # Factor for reducing investment growth rate if investment exceeds certain threshold
    Investment_exog_semi_endog_correction: bool   # Semi-endogenize exogenous investment trend via Keynesian macro signals (gov deficit + profit rate)
    I_correction_gov_deficit_threshold: float     # Gov deficit-to-GDP ratio above which investment dampening begins (e.g. 0.05 = 5% of GDP)
    I_correction_gov_deficit_sensitivity: float   # Strength of gov-deficit crowding-out signal (e.g. 2.0 → each 1pp excess deficit cuts gI_trend by 2pp)
    I_correction_profit_sensitivity: float        # Strength of profit-rate-decline signal (e.g. 2.0 → each 1pp fall in P/Y below initial level cuts gI_trend by 2pp)
    Investment_distribution_Y_based: bool  # Toggle between distributing investment according to sectoral output Y shares or TOTAL output X  stock shares
    Endogenous_excess_import_reduction: bool # Set to true if you endogenously want to reduce imports mirroring directed industrial policy by the Saudi government
    import_reduction_investment_effectivity_factor: float  # Factor for the effectivity of investment growth due to import reduction
    Vision_2030_activity_increase: bool     # Toggle between including Vision 2030 sectoral growth activities or not
    Endogenous_import_reduction_cluster_growth: bool   # Toggle between endogenous import reduction via cluster growth or not
    cluster_growth_effectivity_factor: float  # Factor for the effectivity of cluster growth mechanism
    Adapt_investment_production_scarcity: bool  # Toggle between adapting investment to production scarcity or not
    investment_scarcity_factor: float       # Factor for adapting investment to production scarcity
    Invest_loan_adaptation_to_HH_wealth: bool  # Toggle between adapting investment financed by loans to household wealth dynamics or not
    decrease_invest_profit: float        # Decrease factor for invest_profit in case of decreasing HH wealth
    Multiple_V2030_nonoil_Exports_growth: bool  # Toggle between multiple Vision 2030 informed non-oil export growth rates or one single rate
    Growth_rates_0_test: bool               # Toggle to test the steady-state solution of the model without any growth rates for model consistency
    Y_GDP_Expenditure_X_harmonization: bool # Toggle between adapting X to the Y_expenditure by adding intermediate sales S, or by taking output X as given, and leaving an inconsistency
    Y_timeseries_model_log: bool            # Toggle between Calculating expected growth rates of GDP directly (false) or via log levels (true)
    GY_timeseries_model: bool               # Toggle between time series estimation and fixed average growth rate.
    Gov_expenditures_oil_exports_cyclicality: bool # Toggle to make government expenditures co-depend on oil exports with a lag
    government_oil_exports_cyclicality_factor: float  # Factor for government expenditure growth rate dependence on oil export shocks  
    Countercyclical_gov_policy: bool        # Toggle between countercyclical government policy and fixed government spending growth rate.
    gov_conf_boost_factor: float            # Strength of government confidence boost. Multiplicative: with a value of 1, the inverse of deflation or negative growth is added to government investment!
    profit_tax_rate_1: float                # Set the profit tax rate exogenously to improve government finances
    wage_tax_rate_1: float                  # Set the wage tax rate exogenously to improve government finances
    profit_tax_rate_timing_1: int           # Period when profit tax is introduced (0 = never, >0 = from that period onwards)
    wage_tax_rate_timing_1: int             # Period when wage tax is introduced (0 = never, >0 = from that period onwards)
    profit_tax_rate_2: float                # Set the profit tax rate exogenously to improve government finances
    wage_tax_rate_2: float                  # Set the wage tax rate exogenously to improve government finances
    profit_tax_rate_timing_2: int           # Period when profit tax is introduced (0 = never, >0 = from that period onwards)
    wage_tax_rate_timing_2: int             # Period when wage tax is introduced (0 = never, >0 = from that period onwards)
    tax_phase_in_years: int                 # Number of years over which a new tax rate is phased in linearly (default 5; set to 1 for instant step-change)
    gov_bank_share: float                  # Government ownership share of banking sector (asset-weighted; PIF + GOSI + PPA)
    Oil_export_timeseries_model: bool       # Toggle between time series estimation and fixed average growth rate.
    Oil_exports_random_fluctuations: bool   # Introduce a random fluctuation in oil exports, based on time series data
    oil_exports_uncertainty_change: float    # This parameter changes the variability of oil fluctuations as a multiple of past variance
    Non_oil_export_timeseries_model: bool   # Toggle between time series estimation and fixed average growth rate. 
    Import_pure_adjustment: bool            # Toggle between PURE imports adjusting if demand exceeds supply, or not
    Import_timeseries_model: bool           # Toggle between time series estimation and fixed average growth rate.
    Import_residual_adjustment: bool        # Toggle between imports adjusting if demand exceeds supply, or not
    Demand_reduction_adjustment: bool       # Toggle between imports adjusting if demand exceeds supply, or not
    Demand_desired_adjustment: bool         # Toggle setting demand to zero if it exceeds supply after adjustment, or not
    Sectoral_wage_growth: bool              # Toggle between sectoral wage growth and average economy-wide wage growth
    Inflation_based_wage_growth: bool       # Wage growth according to inflation of past period in the model
    Wage_growth_limit: bool                 # Limit for wage growth in the model
    Y_GDP_sectoral_zero_max: bool           # Toggle between setting sectoral production to zero if it is negative, or not
    ρ: float                                # Loan repayment rate
    ρ_ext: float                            # External sovereign bond repayment rate (analogous to ρ for domestic bonds)
    α0: float                               # fixed consumption
    α1: float                               # MPC out of income
    α3: float                               # MPC out of wealth
    eK: float                               # Capital productivity
    KX: float                               # Initial capital stock to output ratio
    Begin_timeseries_estimation_Y: int      # Year from which the time series estimation starts, -20 means 20 years before the start of the model
    Begin_timeseries_estimation_GY: int     # Year from which the time series estimation starts, -20 means 20 years before the start of the model
    Begin_timeseries_estimation_EX_oil: int # Year from which the time series estimation starts, -20 means 20 years before the start of the model
    Begin_timeseries_estimation_EX_nonoil: int      # Year from which the time series estimation starts, -20 means 20 years before the start of the model
    Begin_timeseries_estimation_deflator: int      # Year from which the time series estimation starts, -20 means 20 years before the start of the model
    gamma_i: float                          # # Investment adjustment rate to match target capital stock taken from Naqvi and Stockhammer (2018) PK directed change Ecol Econ model, set to this value
    price_floor: float                      # Price floor set by government to avoid deflation
    price_ceiling: float                    # Price ceiling set by government to avoid inflation
    price_smooth_factor: float              # EMA weight on current cost share in normal-cost pricing (0=full smooth, 1=no smooth)
    price_decline_limit: float              # Max price decline per period (downward price stickiness, Lavoie administered price theory)
    Fix_finance_sector_price: bool          # Pin financial-sector price to 1 each period (analogous to desal/wastewater); True = stabilise, False = let price float freely
    oil_growth_adjustment: float            # Adjustment factor for oil growth to meet growth rates of final demand
    import_excess_adjustment: float         # Intensity of adjustment of production to mitigate excessive imports
    flex_intermediate_inputs: float         # Flexibility of intermediate inputs
    flex_capital: float                     # Flexibility of capital stock in production
    wage_population_growth_adjustment: float           # Adjustment factor for wage costs to meet growth rates of final demand
    No_limitational_production_supply_led: bool  # If Limitational_production is False, set True for a fully supply-side-led economy (Q_s)
    No_limitational_production_demand_led: bool # If Limitational_production is False, set True for a fully demand-led economy
    vision_2030_timing: float               # The time for which Saudi Vision 2030 activities are relevant
    extended_vision_2030_timing: float      # Extended time for which Saudi Vision 2030 activities are relevant
    non_oil_export_growth_adjustment: float # Adjustment factor for non-oil export growth for scenarios of foreign trade changes for KSA
    V2030_non_oil_export_scaling_factor: float  # Scaling factor for non-oil export growth rates in Vision 2030 and Sustainability scenarios
    construction_sector_V2030_target: float  # Target growth rate of construction sector in Vision 2030
    tourism_sector_V2030_target: float       # Target growth rate of tourism sector in Vision 2030
    AI_hightech_sector_V2030_target: float  # Target growth rate of AI and high tech sectors in Vision 2030
    plastic_sector_V2030_target: float      # Target growth rate of plastic sector in Vision 2030
    water_capacity_constraint_agr: float    # Put a capacity constraint on water usage of agricultural sector in intermediate inputs
    capacity_constraint_nonagr: float       # Put a generalized capacity constraint on all (non-agr) sectors in intermediate inputs
    constraint_period_agr: float            # Period for which the capacity constraint is relevant for agricultural sector
    end_constraint_period_agr: float            # Period for which the capacity constraint is relevant for agricultural sector
    constraint_period_nonagr: float         # Period for which the capacity constraint is relevant for all other sectors
    desalination_high_growth: float         # High growth rate of desalination sector
    water_use_efficiency_change: float      # Change in water use efficiency
    import_growth_adjustment: float         # Adjustment factor for import growth for scenarios of foreign trade changes for KSA
    renewable_energy_scenario: bool         # Enables the renewable energy transition
    vision_2030_scenario: bool              # False = Current Trend, True = Vision 2030
    vision_2030_renewable_target: bool      # Logical switch for Vision 2030 (50% renewable energy by 2030)
    gRE_trend_rate: float                   # Annual growth rate for renewables in the trend scenario
    net_emission_reduction_green_investments_exports_transformation: bool  # Master switch: net-zero-2060 emission reduction, green energy investment, and green-export pathway
    energy_sector_public_share: float      # Public sector share in the energy sector
    enable_targeted_oil_reduction: bool      # Toggle to enable/disable the 95% oil reduction policy
    oil_reduction_target_pct: float          # The target reduction (e.g., 0.95 for 95%)
    oil_reduction_end_t: int                 # The time step to reach the target (e.g., 9 for 2030)
    enable_energy_efficiency_gains: bool    # Toggle to enable/disable energy efficiency improvements
    efficiency_max_reduction_pct: float     # Max reduction (e.g., 0.9 = 90% reduction)
    efficiency_top_tier_annual_gain: float  # Annual efficiency gain for Top 20 sectors (e.g., 0.05 = 5%)
    efficiency_mid_tier_annual_gain: float  # Annual efficiency gain for Middle 20 sectors (e.g., 0.03 = 3%)
    efficiency_low_tier_annual_gain: float  # Annual efficiency gain for all other sectors (e.g., 0.02 = 2%)
    efficiency_cost_as_percent_of_bill: float # The CapEx for 1st year's gain, as % of that sector's energy bill
    efficiency_cost_exponent: float         # Exponent to make costs rise as efficiency increases (e.g., 2.0)
    efficiency_payback_period: float
    efficiency_investment_shares: Dict[str, float] # The sectoral breakdown of efficiency investment
    enable_industrial_electrification: bool
    electrification_annual_rate: float
    electrification_cost_per_gj: float
    vision_2030_target_CCU_SGI_Carbon: bool

        # NET ZERO PARAMETERS ---
    # 1. Reduce
    renewable_energy_target_2030: float     # e.g. 0.50
    hydrogen_penetration_rate: float        # e.g. 0.05
    LCOH_green_2030: float                  # Cost of Green Hydrogen (SAR/kg)
    # 2. Reuse/Recycle (CCU & EOR)
    CCU_capacity_baseline: float            # Initial CCU capacity (Mt)
    CCU_growth_rate: float                  # Annual growth of CCU capacity
    # 3. Remove (CCS & Nature)
    CCS_capacity_target_2035: float         # Target CCS capacity (Mt)
    CCS_cost_per_tonne: float               # Cost per tonne of CCS (SAR)
    nature_based_removal_target: float      # Target removal from Nature (Mt)
    # 4. Carbon Market
    enable_carbon_market: bool              # Toggle market logic
    carbon_credit_price_floor: float        # Floor price (SAR/tonne)
    carbon_market_liquidity: float          # Max credits available (Mt)
# --- HYDROGEN PARAMETERS ---
    enable_green_hydrogen: bool             # Toggle to enable H2 logic
    hydrogen_target_2030_gj: float          # Production target by 2030 in GJ
    electrolyzer_capex_per_gj: float        # Investment cost per GJ of capacity (derived from $ per kW)
    hydrogen_efficiency: float              # GJ of Electricity needed to produce 1 GJ of H2 (e.g. 1.3 - 1.5)
    
    # --- RENEWABLE MIX CONTROL ---
    enable_dynamic_re_mix: bool      # Switch: False = Keep 2023 shares; True = Use targets below
    target_solar_share_2030: float   # Target share for Solar by 2030 (e.g., 0.75)
    target_wind_share_2030: float    # Target share for Wind by 2030 (e.g., 0.20)
    target_bio_share_2030: float     # Target share for Bioenergy by 2030 (e.g., 0.05)
    hydrogen_export_share: float            # % of production destined for export (vs domestic use)
    
    enable_re_cost_learning: bool    # Toggle to enable annual cost reduction for renewables
    re_cost_learning_rate: float     # Annual % reduction in CapEx (e.g., 0.02 for 2%)
    re_cost_learning_threshold: float # Share required to trigger reduction (e.g., 0.50 for 50%)

    # --- FOREIGN DIRECT INVESTMENT (FDI) ---
    enable_FDI: bool                 # Switch: True = decompose investment into domestic + FDI; False = all investment is domestic
    FDI_target: float                # Total cumulative FDI growth target from 2021 base year (e.g. 0.30 = 30% total growth over vision_2030_timing period)
    FDI_follow_domestic_growth: bool # If True, FDI grows with domestic investment (gI_trend_endog) + FDI_target on top; if False, FDI grows only at fixed FDI_target rate from base year

    # --- REMITTANCES & BALANCE OF PAYMENTS ---
    enable_remittances: bool             # Switch: True = project remittances forward; False = remittances stay at zero
    remittances_follow_wage_growth: bool # If True, remittances grow with nominal wage bill growth; if False, grow at calibrated average rate
    remittances_growth_adjustment: float # Manual adjustment to remittances growth rate (additive, e.g. 0.01 = +1pp)
    r_FDI_repatriation: float            # Rate of return on FDI stock repatriated abroad (primary income outflow, Phase 2)
    r_gov_ext_assets: float              # Rate of return on government gross external assets (primary income inflow, Phase 2)
    max_FDI_income_share: float          # Cap: FDI income repatriation cannot exceed this share of private profits (prevents instability)
    Gov_ext_assets_extraction: float     # Share of Gov_ext_assets_income extracted to Gov_rev; (1-this) reinvested into asset stock (0=full reinvestment, 1=full extraction)
    remittances_saudization_rate: float  # Annual decline in foreign-worker share of remittances (Saudization). 0=no effect, 0.015=1.5%/yr compounding
    gov_surplus_pif_rate: float          # Share of government fiscal surplus invested in PIF (Gov_ext_assets). 0=no reinvestment, 1=100% of surplus into PIF

    @classmethod
    def default_values(cls):
        return cls(
            ####################################################################################################################################################################
            # KEY scenario switches and parameters that have large effects in model and determine main scenario characteristics
            ####################################################################################################################################################################
            # Scenario switches: the main parameters for each scenario are set together here.
            #######################################################
            # Expectation mechanism for supply choice, whether it depends on time series estimation or average growth rates (GDP and Inflation)
            #######################################################
            Y_timeseries_model =                    True,       # Standard: TRUE, expectations based on time series estimation. Toggle between Calculating expected growth rates of GDP Output Y from time series or from average final demand growth in the past.
            Begin_timeseries_estimation_Y =         -30,        # Year from which the time series estimation starts, standard:-30 (means 30 years before the start of the model).
            Inflation_timeseries_model =            True,       # Standard: TRUE. Expectations based on average past inflation. Toggle between Calculating expected inflation from time series or from average past inflation
            Begin_timeseries_estimation_deflator =  -30,        # Year from which the time series estimation starts, standard:-30 (means 30 years before the start of the model).
            #######################################################
            # Endogenize A matrix and investment dynamics
            #######################################################
            Investment_distribution_Y_based =       True,       # Standard: True in V2030/sust (inv. adapts more to value added!), False in baseline (no structural change, i.e. X-bound). Investment according to sectoral output Y shares or TOTAL output X  stock shares
            Endogenous_investment =                 False,       # Toggle between endogenous investment and investment growing with fixed rate
            Endogenous_investment_reduction =       True,      # Toggle between reducing investment in case of investment crosses a certain threshold, or not
            investment_reduction_factor =           0.25,        # Standard value: 0.9, i.e. reduce average investment growth rate by (1 - parameter.value) if investment exceeds certain threshold
            # Semi-endogenous Keynesian dampening when Endogenous_investment = False
            Investment_exog_semi_endog_correction = True,        # True = apply Keynesian macro dampening to exogenous gI_trend in baseline
            I_correction_gov_deficit_threshold    = 0.037,        # 4% of GDP deficit is the trigger threshold
            I_correction_gov_deficit_sensitivity  = 2.3,         # Each 1pp excess deficit above threshold reduces gI_trend by 2pp
            I_correction_profit_sensitivity       = 2.3,         # Each 1pp fall in profit rate (P/Y) below initial level reduces gI_trend by 2pp
            Endogenous_A_matrix =                   False,       # Toggle between an endogenous A matrix (modified by import excesses and Vision 2030 goals) and a fixed A matrix (Leontief production); used for import control.
            Adapt_investment_production_scarcity =  False,       # Add reaction to production scarcity for intermediate inputs and capital (NOT labour) to modify the endogenous investment growth
            investment_scarcity_factor =            0.7,        # Factor for adapting investment to production scarcity, higher values lead to stronger reaction of investment to production scarcity
            Vision_2030_activity_increase =         False,       # Toggle for including Vision 2030 sectoral growth activities
            Endogenous_import_reduction_cluster_growth =    False,   # disabled in baseline scenario; enabled with non-zero effectivity factor in transformation scenario
            cluster_growth_effectivity_factor =             0.5,      # Factor for the effectivity of cluster growth mechanism; higher values lead to stronger reaction of domestic production to excess imports via cluster growth
            Invest_loan_adaptation_to_HH_wealth =           False,      # Standard: True in V2030. Toggle between adapting investment financed by loans to household wealth dynamics or not
            decrease_invest_profit =                        0.9,        # Standard: 0.9 in V2030, i.e. 10% decrease invest out of profit per year. Decrease factor for invest_profit in case of decreasing HH wealth, e.g. 0.5 means invest_profit is halved if HH wealth decreases
            #######################################################
            # Demand and import adjustment mechanism parameters in case demand exceeds limited supply (in case of supply constraints)
            #######################################################
            Endogenous_excess_import_reduction = False,          # Requires Endogenous_investment and Endogenous_A_Matrix= True. Set to true if you endogenously want to reduce imports mirroring directed industrial policy by the Saudi government
            import_excess_adjustment =      0,                # Standard: 0.5 Sust, 0.2 V2030, False baseline.  Regulates amount of imports above sectoral production that is added to sectoral production and deducted from imports of goods by that sector
            import_reduction_investment_effectivity_factor =  0.5, # Standard value: 0.5 in V2030/Sust. Higher values mean more investment is needed to achieve the import reduction. Factor for the effectivity of investment growth due to import reduction, higher values lead to stronger reaction of investment to import reduction
            Import_pure_adjustment =        True,               # Standard: TRUE. Pure import adjustment increases imports when demand exceeds supply, without limiting demand; this is the default and most stable option.
            Demand_reduction_adjustment =   False,              # Standard: False. Alternative to import pure adjustment: Instead of increasing imports, demand is reduced to domestic production and supply possibilities with fixed imports. (E.g. if imports cannot be increased at will)
            Import_residual_adjustment =    False,              # Standard: False.  After demand has been reduced, the residual gap between demand and supply is adjusted via imports.
            Demand_desired_adjustment =     False,              # Standard: False.  After demand has been reduced, residual desired demand is set to zero if it exceeds supply.
            #######################################################
            # Flexibility parameters for production function
            #######################################################
            flex_intermediate_inputs =              0,         # STANDARD baseline: 0 (0%, limit growth due to lack of industrial policy in baseline). This term is additive, because more flex in int inp: v.beta_[t]* v.IntP_[t-1] * pc.non_agr_ * (1 + p.flex_intermediate_inputs  + v.gI_endog_[t-1]) 
            flex_capital =                          1,         # STANDARD baseline: 1 (multiplicatively neutral). This term is multiplicative: v.kappa_[t]  * v.K_[t-1] * (1 + v.gk_[t-1]) * p.flex_capital), so a value of 1 is multiplicatively neutral
            Update_productivity_coefficients =      True,      # Productivity parameters alpha, beta, kappa are updated to changes in the production structure if this is TRUE.
            #######################################################
            # Government policy and growth rate time series parameters government expenditures
            #######################################################
            GY_timeseries_model =                       True,           # Toggle between time series estimation and fixed average growth rate. 
            Gov_expenditures_oil_exports_cyclicality =  False,          # Standard: False. Optionally makes government expenditure pro-cyclical with the oil-export shock.
            government_oil_exports_cyclicality_factor = 0.2,           # Government expenditure growth rate changes multiplicatively with this factor.
            Begin_timeseries_estimation_GY  =           -30,            # Year from which the time series estimation starts, standard: -30 (means 30 years before the start of the model)
            Countercyclical_gov_policy =                True,           # Toggle between countercyclical government policy and fixed government spending growth rate.
            gov_conf_boost_factor =                     0.5,            # Strength of government confidence boost. Multiplicative: with a value of 1, the inverse of deflation or negative growth is added to government investment!            
            profit_tax_rate_1 =                         0,              # Standard value if not 0: 0.12 
            wage_tax_rate_1 =                           0,              # Standard value if not 0: 0.05
            profit_tax_rate_timing_1 =                  0,              # Set to period number to activate, 9 = 2030
            wage_tax_rate_timing_1 =                    0,              # Set to period number to activate, 9 = 2030
            profit_tax_rate_2 =                         0,              # Standard value if not 0: 0.2
            wage_tax_rate_2 =                           0,              # Standard value if not 0: 0.05 or 0.07
            profit_tax_rate_timing_2 =                  0,              # Set to period number to activate, 29 = 2050
            wage_tax_rate_timing_2 =                    0,              # Set to period number to activate, 29 = 2050
            tax_phase_in_years =                        5,              # Phase-in duration in years for new taxes (linear ramp; 1 = instant step-change)
            gov_bank_share =                        0.22,       # ~22% of Saudi banking assets owned by government entities (PIF ~37% of SNB, GOSI stakes, PPA stakes; asset-weighted 2021)
            #######################################################
            # OIL export growth mechanism and adjustment of oil growth to analyze oil dependency of Saudi economy
            #######################################################
            Oil_export_timeseries_model =           False,      # Toggle between time series estimation and fixed average growth rate. 
            Begin_timeseries_estimation_EX_oil =    - 30,       # Year from which the time series estimation starts, standard: -30 (means 30 years before the start of the model).
            oil_growth_adjustment =                 0.025,          #  Standard: 0. Or 0.025, which about replicates past GDP growth Adjustment factor for oil Export growth to replicate long term growth rates of final demand
            Oil_exports_random_fluctuations =       False,
            oil_exports_uncertainty_change =        1,         # Multiplier on oil export variability relative to past variance; 1 = no change, 1.5 = 50% increase in variability
            #######################################################
            # Vision 2030 related timing and non-oil export growth adjustment
            #######################################################
            vision_2030_timing =                    9,          # Standard: 9 until 2030. from 2021, it is 9 years to 2030 for Saudi vision 2030!
            extended_vision_2030_timing =           14,         # Standard: 14 until 2035 in V2030, 19 until 2040 Sust. Extended time horizon for Vision 2030 activities, e.g. for desalination sector growth that goes beyond 2030
            Multiple_V2030_nonoil_Exports_growth =  False,       # Toggle between multiple Vision 2030-informed non-oil export growth rates or a single rate
            non_oil_export_growth_adjustment =      0.025,      # Standard for baseline/BAU: 0.025 (i.e. 2.5%), Standard for Vision 2030 Proxy: 0.05. 0 means no change, non-oil export growth be changed in subtractive/additive way in the model.
            V2030_non_oil_export_scaling_factor =   2,          # Standard for Vision 2030 Proxy: 2.0. DOES NOT expire after end Vision 2030, this is designed as a long-term strategy! Scaling factor for non-oil export growth rates in Vision 2030 and Sustainability scenarios
            construction_sector_V2030_target =      0.30,       # Standard: 0.15 V2030, 0.3 Sust.
            tourism_sector_V2030_target =           0.30,       # Standard: 0.15 V2030, 0.3 Sust.
            AI_hightech_sector_V2030_target =       0.30,       # Standard: 0.15 V2030, 0.3 Sust.
            plastic_sector_V2030_target =           0.30,       # Standard: 0.25 V2030, 0.25 Sust.
                # 0.025 approximates average growth rate of the average economy in the last 10 years
                # 0.05 (5% growth rate) for non oil exports (ca. double average Y growth rate last 10 years) at the moment is taken to proxy the Saudi Vision 2030 strategy
            #######################################################
            # Wage growth adjustment parameter
            #######################################################
            Sectoral_wage_growth =                  True,       # When TRUE, wages move in line with sectoral growth rates, with a lag. 
            Inflation_based_wage_growth =           True,       # When TRUE, wages are adjusted by past inflation (the default choice).
            wage_population_growth_adjustment      = 0.0113,                    # Standard: 0.0113, to reflect population growth.
            # Population growth data:
            # Average annual population growth rate in KSA (2020-2060):
            # SSP1: 1.13% ; 0.0113
            # SSP2: 1.39% ; 0.0139
            # SSP3: 1.64% ; 0.0164
            #######################################################
            # Production function type switches: If limitational production is true, supply constraints apply, otherwise choose between demand-led or supply-choice-led economy
            #######################################################
            Limitational_real_production =              True,       # Standard: TRUE, once tested and verified. An endogenous and adaptive supply constraints to REAL output
            Limitational_nominal_production =           False,      # Standard: FALSE if REAL limitational production works., i.e. an endogenous and adaptive supply constraints to NOMINAL output
            No_limitational_production_demand_led =     False,      # IF Limitational_production is false, set this to TRUE if you want a completely demand-led economy (i.e. supply equals demand)
            No_limitational_production_supply_led =     False,      # IF Limitational_production is false, set this to TRUE if you want a completely supply-choice (Q_s) led economy
            price_floor =                               0.5,
            price_ceiling =                             1,
            price_smooth_factor =                       0.5,        # EMA weight on current cost share: 0.3 = 30% actual cost, 70% smoothed (normal-cost pricing, Kalecki/Lavoie)
            price_decline_limit =                       0.1,       # Max 10% price decline per period — downward stickiness (administered price theory, Lavoie)
            Fix_finance_sector_price =                  True,       # Standard: True. Pin financial-sector price to 1 (prevents large drop seen in unconstrained runs). Set False to let the price float freely.
            ####################################################################################################################################################################
            # Energy sector switches and growth rates
            ####################################################################################################################################################################            
            renewable_energy_scenario =             False,      # Enables the renewable energy transition
            vision_2030_scenario =                  False,      # False = Current Trend, True = Vision 2030
            net_emission_reduction_green_investments_exports_transformation = False,  # Set to True to run the Net Zero 2060 pathway
            vision_2030_target_CCU_SGI_Carbon =     True,     # Toggle for the specific Vision 2030 target implementation.
            vision_2030_renewable_target =          False,      # Set to True for the Vision 2030 scenario, False for the trend scenario
            gRE_trend_rate =                        0.154,          # Annual renewable growth rate for the trend scenario.
            energy_sector_public_share =            0.8,        # 0.8 (80% public ownership of the energy sector in KSA); set to 0.5 for a more socialized energy sector
            enable_targeted_oil_reduction =         False,      # Oil reduction according to V2030, 95% until 2030 for three sectors (industries, power plant, desalination)
            oil_reduction_target_pct =              0.95,
            oil_reduction_end_t =                   9,          # t=9 corresponds to 2030 (since t=0 is 2021)
            enable_energy_efficiency_gains =        False,       # Set to True to activate the new logic
            efficiency_max_reduction_pct =          0.50,       # Maximum reduction is 50%
            efficiency_top_tier_annual_gain =       0.01,       # 1% gain per year for the top-tier sectors
            efficiency_mid_tier_annual_gain =       0.005,      # 0.5% gain per year for the mid-tier sectors
            efficiency_low_tier_annual_gain =       0.0025,     # 0.25% gain per year for the remaining sectors            
            efficiency_cost_as_percent_of_bill =    0.10,       # 10% of annual energy bill to get 1st year's gain
            efficiency_cost_exponent = 2.0,                     # Cost = Base * (1 / (1-EffShare))^2
            efficiency_payback_period = 3.0,
                                # --- ELECTRIFICATION STRATEGY (Continuous) ---
            enable_industrial_electrification = False,
            # Annual Shift Rate: 0.015 means 1.5% of the sector's fossil fuel use 
            # is converted to electricity every year.
            electrification_annual_rate = 0.045,    
            # Cost: Approx 300 SAR per GJ of capacity (Industrial Heat Pumps/Furnaces)
            electrification_cost_per_gj = 300.0,
            efficiency_investment_shares = {
                'Manufacture of machinery and equipment n.e.c.': 40,
                'Construction of buildings': 30,
                'Architectural and engineering activities; technical testing and analysis': 20,
                'Specialized construction activities': 10
            },
 
            ####################################################################################################################################################################
            # Test the model without growth rates: set almost every switch to False (except Import_pure_adjustment and Limitational_production are TRUE) and adjust all growth factors to zero
            ####################################################################################################################################################################
            Growth_rates_0_test = False,                        # Toggle to test the steady-state solution of the model without any growth rates for model consistency
            # Note: to activate this test, all time series models must be set to False, and all endogenous mechanisms that might affect external growth rates must be disabled.
            ####################################################################################################################################################################
            # Water capacity constraints and desalination investment dynamics parameters
            ####################################################################################################################################################################
            # it enters the production function with v.beta_[t]* v.IntP_[t-1] * pc.s_agr_ * (1 - p.water_capacity_constraint_agr) * (1 + p.flex_intermediate_inputs + v.gI_endog_[t-1]),
            water_capacity_constraint_agr = 1,                  # Standard: 1, means NO constraint. Set to number between 0 and 1 to implement capacity constraint on the intermediate input of water in the agricultural sector.
            # water_capacity_constraint_agr = 0.5, based on Odnoletkova and Patzek, p. 4,s loosely models that non-renewable ground water for the AGR sector runs out, and can only be partly substituted
            capacity_constraint_nonagr = 0,                     # Standard: 0, means NO constraint. Set to number between 0 and 1 to implement capacity constraint on the intermediate input of all sectors except agriculture
            constraint_period_agr = 10,                         # Period for which the capacity constraint is relevant for AGR sector
            end_constraint_period_agr = 42,                     # END constraint is relevant for AGR sector
            constraint_period_nonagr = 10,                      # Period for which the capacity constraint is relevant for Non-AGR sectors
            desalination_high_growth = 1,                       # Set to 1 for a high growth rate of the desalination sector with past average desalination growth rates from the desalination database
            water_use_efficiency_change = 0,                    # Exponential  changes in water use efficiency every year, positive values give an increase in water use efficiency
            # For water_use_efficiency_change = 0.015, is the standard scenario value for the sustainability scenario, with high desalination growth
            ####################################################################################################################################################################
            # Scenario switches and parameters varied less often
            ####################################################################################################################################################################
            Warnings_ON = True,                                 # Toggle between showing warnings or not
            Y_timeseries_model_log = False,                     # Toggle between expected growth rates of GDP directly (False) or via log levels (True); log-level model tends to underperform; use with caution.
            Non_oil_export_timeseries_model = False,            # No non-oil export time series available for KSA; switch kept False.
            Begin_timeseries_estimation_EX_nonoil = -30,        # Year from which the time series estimation starts, -20 means 20 years before the start of the model
            Import_timeseries_model = False,                    # Toggle between time series estimation and fixed average growth rate. 
            Wage_growth_limit = False,                          # Toggle between wage growth limit in the model or not
            Y_GDP_sectoral_zero_max = False,                    # Toggle between setting sectoral production to zero if it is negative, or not.
            Y_GDP_Expenditure_X_harmonization = False,          # Toggle between adapting X to the Y_expenditure by adding intermediate sales S, or by taking output X as given, and leaving an inconsistency
            ρ=0.05,                                             # Loan repayment rate, these 5% per year are standard (20 years credit), is the same as in Poledna, Miess et al. (2023) ABM paper
            ρ_ext=0.05,                                         # External sovereign bond repayment rate; 0.05 = 20-year maturity (consistent with domestic bond treatment)
            α0=0,                                               # fixed consumption
            α1=1,                                               # Marginal Propensity to Consume (MPC) out of wage income, set to 1 according to IOT data, where cons exceeds labour income (thus actually calibrated)
            α3=0,                                               # MPC out of wealth, is set to zero for now
            eK=1,                                               # Capital productivity; set to 1 by convention (see calibration). Can be differentiated into target vs. actual capital stock productivity.
            KX=3,                                               # Initial capital stock to output ratio, calibrated according to structural CGE literature (model input)
            gamma_i=0.4,                                        # Investment adjustment rate to match target capital stock taken from Naqvi and Stockhammer (2018) PK directed change Ecol Econ model, set to this value
            import_growth_adjustment = 0,                       # 0 means no change, import growth will be changed in subtractive/additive way in the model. Small changes in import growth have considerable impact on the economy
            ####################################################################################################################################################################
            # Carbon capture, removal & market parameters
            ####################################################################################################################################################################
            renewable_energy_target_2030 = 0.50,
            hydrogen_penetration_rate = 0.02,
            LCOH_green_2030 = 5.55,                 # ~1.48 USD * 3.75

            CCU_capacity_baseline = 0.5,            # 0.5 Mt initial
            CCU_growth_rate = 0.02,                 # 2% growth

            CCS_capacity_target_2035 = 44.0,        # 44 Mt target
            CCS_cost_per_tonne = 350.0,             # Approx (69+25)*3.75
            nature_based_removal_target = 278.0,    # 278 Mt (SGI target)

            enable_carbon_market = True,            # Active in baseline
            carbon_credit_price_floor = 37.5,       # 2024 Auction price
            carbon_market_liquidity = 10.0,         # 10 Mt limit
            # --- HYDROGEN PARAMETERS (Calibrated to NEOM Phase 1) ---
            enable_green_hydrogen = False,          # Disabled in baseline
            hydrogen_target_2030_gj = 30_000_000,  # Target: ~0.25 Mt (approx NEOM capacity)
            electrolyzer_capex_per_gj = 475.0,     # ~40% of total project cost (Electrolyzers + Balance of Plant)
            hydrogen_efficiency = 1.4,             # Standard PEM/Alkaline efficiency
            hydrogen_export_share = 0.8,           # Share of H2 production exported

            enable_re_cost_learning = False,        # Disabled in baseline
            re_cost_learning_rate = 0.02,           # 2% reduction per year in renewable cost
            re_cost_learning_threshold = 0.20,      # Renewable share at which cost reduction starts
            # --- RENEWABLE MIX DEFAULTS ---
            enable_dynamic_re_mix = False,          # False: keep 2023 shares unchanged
            target_solar_share_2030 = 0.75,         # Target solar share by 2030 (used when switch is True)
            target_wind_share_2030 = 0.20,
            target_bio_share_2030 = 0.05,

            # --- FDI PARAMETERS ---
            enable_FDI = True,              # FDI enabled in baseline (with zero target → constant share)
            FDI_target = 0.0,               # No FDI growth target in baseline
            FDI_follow_domestic_growth = True,   # FDI tracks domestic investment growth → constant FDI share (since FDI_target=0)

            # --- REMITTANCES & BALANCE OF PAYMENTS ---
            enable_remittances = True,              # Remittances projection enabled
            remittances_follow_wage_growth = True,  # Remittances grow with nominal wage bill
            remittances_growth_adjustment = 0.0,    # No additional adjustment
            r_FDI_repatriation = 0.05,              # Rate of return on FDI stock repatriated abroad, set to 5%
            r_gov_ext_assets = 0.05,                # Return on government gross external assets (PIF/SAMA reserves), set to 5%
            max_FDI_income_share = 0.50,            # Cap FDI income payments at 50% of private profits to prevent model instability
            Gov_ext_assets_extraction = 0.5,        # 50% of Gov_ext_assets_income extracted to Gov_rev; 50% reinvested into asset stock
            remittances_saudization_rate = 0.0,     # Baseline: no Saudization effect
            gov_surplus_pif_rate = 0.0,              # Baseline: no PIF reinvestment

            ####################################################################################################################################################################
            # Not yet implemented — the following switch is defined but has no effect on model computation
            ####################################################################################################################################################################
            Productivity_Coefficients_X_adjustment = False,  # Adjusting productivity coefficients to output X vs. Y; not yet implemented
        )

