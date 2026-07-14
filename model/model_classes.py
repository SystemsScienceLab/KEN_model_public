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
from .parameters_baseline import ModelParameters
from .calibration import ParametersCalibrated

NDArray = npt.NDArray[np.float64]


class ModelConfig:
    """Configuration of the model."""

    def __init__(self, p: ModelParameters, pc: ParametersCalibrated, T: int):
        self.T = T
        self.p = p
        self.pc = pc


@dataclass
class ModelVariables:
    """Dynamic variables within a model simulation."""

    #################################################################################################################################################
    # Dynamic IO table
    #################################################################################################################################################
    A__: NDArray                                # Coefficients matrix A (input-output table normalized by 1/X)
    Z__: NDArray                                # Matrix of intermediate input flows (nominal)
    A_proxy__: NDArray                          # Dynamic input-output table (proxy for calculations)
    gA__: NDArray                               # Growth rate of the input-output table (to endogenize)
    desal_share: NDArray
    #################################################################################################################################################
    # Adapting productivity and other endogenized coefficients
    #################################################################################################################################################
    alpha_: NDArray                             # Productivity coefficient labor
    beta_: NDArray                              # Productivity coefficient intermediate inputs
    kappa_: NDArray                             # Productivity coefficient capital
    invest_profit: NDArray                      # Share of investment financed by profits (not loans), adaptable to scenario
    
    #################################################################################################################################################
    # Sectoral Macro variables (denoted with _)
    #################################################################################################################################################
    Y_: NDArray                                 # Final demand (sales, nominal)
    y_: NDArray                                # Final demand (sales, real, corrected for overall inflation)
    Y_s_: NDArray                               # Final demand (sales, nominal) SUPPLY constrained after Production Function by sector s
    y_s_: NDArray                               # REAL Final demand (sales, real) SUPPLY constrained after Production Function by sector s
    Y_non_oil_: NDArray                         # Non-oil GDP proxy (national accounts definition, scaled to GASTAT ratio)
    Y_oil_: NDArray                             # Oil GDP proxy (national accounts definition, scaled to GASTAT ratio)
    y_non_oil_: NDArray                        # Non-oil GDP proxy real
    y_oil_: NDArray                            # Oil GDP proxy real
    Y_gov_: NDArray                             # Government activities GDP proxy (national accounts definition, scaled to GASTAT ratio)
    y_gov_: NDArray                             # Real government activities GDP proxy
    Y_net_taxes: NDArray                        # Net taxes on products aggregate (fixed ratio of total GDP)
    Y_other_gdp: NDArray                        # Other GDP = Government activities + Net taxes on products (national accounts)
    Y_diff_: NDArray                            # Difference (should be 0) between final demand after supply constraints Y_s_ and Y_ final demand after unmet demand correction
    Y_test_alpha_: NDArray                      # Y component from labour productivity term (alpha)
    Y_test_beta_: NDArray                       # Y component from intermediate input productivity term (beta)
    Y_test_kappa_: NDArray                      # Y component from capital productivity term (kappa)
    X_diff_: NDArray                            # Difference (should be 0) between X_proxy in input-output endogenization and X_ calculated with new Leontief matrix
    X_inv_diag__: NDArray                       # Inverse diagonal matrix of X_ (to calculate shares of intermediate inputs for each sector)
    Q_s_: NDArray                               # Supply choice nominal by firms determined by their expectations function.
    q_s_: NDArray                               # Supply choice real by firms determined by their expectations function.
    Q_d_: NDArray                               # Aggregate desired NOMINAL demand (sum of all desired demand components) before production takes place
    q_d_: NDArray                               # Aggregate desired REAL demand (sum of all desired demand components) before production takes place
    Q_: NDArray                                 # Aggregate REALIZED Demand (after aggregate demand has been reduced due to supply constraints)
    X_: NDArray                                 # Total output (intermediate + final demand, nominal)
    x_: NDArray                                 # Total output (intermediate + final demand, real)
    dX_: NDArray                                # Sectoral distribution of total nominal output (share of each sector in total output)
    C_: NDArray                                 # Sectoral household consumption (nominal)
    c_: NDArray                                 # Sectoral household real consumption (corrected for sectoral inflation)
    GY_: NDArray                                # Sectoral government consumption for IOTs
    K_: NDArray                                 # Capital stock (nominal)
    I_: NDArray                                 # Investment in capital stock (nominal)
    I_energy_eff_: NDArray                      # Investment in energy efficiency (nominal)
    I_private_: NDArray                         # Private investment in capital stock (nominal) — TOTAL including FDI-financed portion, used for GDP
    I_public_: NDArray                          # Public (gov) investment in capital stock (nominal) — TOTAL including FDI-financed portion, used for GDP
    I_private_dom_: NDArray                     # Private investment financed domestically (= I_private_ minus FDI private share) — used for firm financing
    I_public_dom_: NDArray                      # Public investment financed domestically (= I_public_ minus FDI public share) — used for gov expenditure
    Sectoral_investment_distribution_: NDArray  # Sectoral distribution of investment (share of each sector in total investment DONE)
    Sectoral_en_eff_investment_distribution_: NDArray  # Sectoral distribution of investments in energy efficiency!
    L_: NDArray                                 # Loans (nominal)
    D_HH: NDArray                               # HH deposits = loan proceeds created by endogenous money, attributed to households (D_HH = sum(L_))
    IntS_: NDArray                              # Intermediate sales nominal
    intS_: NDArray                              # Intermediate sales real (corrected by sectoral inflation)
    IntP_: NDArray                              # Intermediate purchases nominal
    intP_: NDArray                              # Intermediate purchases real (corrected by sectoral inflation)
    IntP__: NDArray                             # Intermediate purchases matrix (to calculate shares of intermediate inputs for each sector)
    IntP_shares__: NDArray                      # SHARES MATRIX of intermediate inputs by other sectors in total intermediate inputs for each sector
    IntP_shares_v2030__: NDArray                # SHARES MATRIX for Vision 2030 sectors of intermediate inputs by other sectors in total intermediate inputs for each sector
    IntP_V2030_proxy__: NDArray                 # Intermediate purchases increased by a PROXY of how much intermediate inputs will increase to reduce imports accordingly
    IntP_proxy_: NDArray                        # Intermediate purchases proxy for endogenization of A matrix
    IntP_proxy__: NDArray                       # Intermediate purchases proxy (sum for each sector) for endogenization of A matrix
    IntP_shares_proxy__: NDArray                # PROXY SHARES MATRIX that give you how to get back from aggregate sum of intermediate inputs by sector to individual intermediate inputs
    Import_IntP_ratio_: NDArray                 # Intermediate purchases to import ratio
    Import_IntS_ratio_: NDArray                 # Intermediate Sales to import ratio
    P_: NDArray                                 # Firm profits nominal
    pi_real_: NDArray                           # Firm profits real (corrected by sectoral price levels)
    P_tot_: NDArray                             # Total profits including financial profits and public profits
    unit_cost_smooth_: NDArray                  # EMA-smoothed unit cost share (X-P_tot)/X — normal-cost pricing (Kalecki/Lavoie)
    pi_tot_: NDArray                            # Total profits real (corrected by sectoral price levels)
    EX_: NDArray                                # Exports nominal
    EX_oil_: NDArray                            # Oil Exports nominal
    EX_non_oil_: NDArray                        # Non-Oil Exports nominal
    ex_: NDArray                                # Exports real
    ex_oil_: NDArray                            # Oil Exports real
    ex_non_oil_: NDArray                        # Non-Oil Exports real
    IM_: NDArray                                # Imports nominal
    im_: NDArray                                # Imports real
    W_: NDArray                                 # Nominal Wages vector (compensation of employees) from IOTs
    w_: NDArray                                 # REAL wages vector (deflated by aggregate price level; sectoral CPI deflation not implemented)
    DEPR_: NDArray                              # Depreciation of capital stock from IOTs
    p_: NDArray                                 # Prices
    Min1_: NDArray                              # Minimum 1 in production function
    Min2_: NDArray                              # Minimum 2 in production function
    supply_constraint_IntP_: NDArray            # Supply constraint metric INT INPUT and Capital K (0 = no supply constraint, 1 = full supply constraint)
    supply_constraint_K_: NDArray               # Supply constraint metric INT INPUT and Capital K (0 = no supply constraint, 1 = full supply constraint)
    supply_constraint_W_: NDArray               # Supply constraint metric WAGE (0 = no supply constraint, 1 = full supply constraint)
    binding_intermediate_input_gap_: NDArray    # Binding intermediate input gap (0 = no binding gap, 1 = full binding gap)
    binding_capital_stock_gap_: NDArray         # Binding capital stock gap (0 = no binding gap, 1 = full binding gap)
    binding_wage_gap_: NDArray                  # Binding wage gap (0 = no binding gap, 1 = full binding gap)
    excess_import_: NDArray                     # Excess import measure (0 = no excess import, 1 or >1 = full excess import equal more than Y)
    excess_scarcity_: NDArray                   # Excess scarcity measure (0 = no excess scarcity, 1 = full excess scarcity)
    Unmet_demand_: NDArray                      # Unmet demand
    unmet_demand_total: NDArray                 # Total unmet demand
    I_demand_: NDArray                          # Investment demand (the investment components produces by sector s, not to be confused with investment BY sector s)
    Y_production_: NDArray                      # Final demand (Y) by sector s calculated by production approach
    Y_distribution_: NDArray                    # Final demand (Y) by sector s calculated by distribution approach
    X_Y_ratio_: NDArray                         # Ratio of total output (X) to final demand (Y) by sector s
    X_proxy_: NDArray                           # Total output (X) by sector s calculated via simple ratio to Y_
    Domestic_increase_: NDArray                 # Domestic increase in production (X) by sector s to improve import dependencies and implement industrial transformation in KSA
    
    ################################################
    # Government sectoral variables
    ################################################
    Tax_products_net_: NDArray                  # Net tax on products
    Tax_production_: NDArray                    # Tax on production
    Sub_production_: NDArray                    # Subsidies on production
    Tax_investments_: NDArray                   # Tax on investments


 
    #################################################################################################################################################
    # MACRO VARIABLES - no underscore_
    #################################################################################################################################################

    # Macro aggregates
    I_total: NDArray                            # Investment in capital stock (nominal)
    I_total_desired: NDArray                    # Investment in capital stock by sectors (nominal, desired according to investment function before supply constraints, used for diagnostics)
    I_total_demand: NDArray                     # Investment in capital stock demand (nominal, from demand side calculation)
    i_total: NDArray                            # Investment in capital stock (real)
    I_desal: NDArray                            # Desalination investment
    inflation: NDArray                          # Inflation rate BIP deflator, weighted with dX
    deflator_gdp: NDArray                       # Price index GDP deflator: Cumulated (summed up) inflation over total model horizon
    inflation_consumers: NDArray                # Inflation rate consumer prices, weighted with dC
    price_index_cons: NDArray                   # Consumer Price Index (CPI): Cumulated (summed up) inflation over total model horizon
    PB: NDArray                                 # Bank profits

    # Disposable (net) income from wage (labour inc related VAT deducted)
    YD_wage: NDArray                            # Nominal disposable wage income, TH deducted
    yd_wage: NDArray                            # Real disposable wage income, corrected by consumption price index
    
    # Disposable (net) income from profits (TH + profit inc related VAT deducted)
    P_distributed: NDArray                      # Private profit net of firm debt service: sum(s_private*P) - (rl+rho)*L
    YD_profit: NDArray
    C: NDArray                                  # Consumption nominal
    c: NDArray                                  # Consumption real
    V: NDArray                                  # Household financial wealth (portfolio claims on banks)
    HH_total_wealth: NDArray                    # Household net worth = V + D_HH + E_firm_HH + E_bank_HH + OFA_HH (balance-sheet consistent)
    OFA_HH: NDArray                             # HH Other Financial Assets stock (cumulative TFM residual; ΔOFA_HH = ΔD_HH − ΔE_firm_HH)
    
    # Wages (total sum household income), i.e. compensation of employees
    W: NDArray                                  # Nominal wages total
    w: NDArray                                  # Real wages total

    # Government and banks
    GY: NDArray                                 # Government consumption for IOTs
    Gov_exp: NDArray                            # Government expenditures total (IOTs + gov. data) nominal
    gov_exp: NDArray                            # Government expenditures total (IOTs + gov. data) real (corrected by gdp deflator)
    Gov_rev: NDArray                            # Government revenues total (IOTs + gov. data)
    # Household tax (captures residual tax income by government not in IOTs)
    TH: NDArray
    VAT: NDArray                                # Value added tax
    
    # Government
    GI: NDArray                                 # Government investment (public firms)
    GV: NDArray                                 # Government savings (initialized as net international investment position)
    PIF: NDArray                                # Public Investment Fund (PIF) asset stock (Saudi accumulated stock of government wealth from past oil income)
    GP: NDArray                                 # Government profits
    Bond_domestic: NDArray                      # Government bonds held domestically (60% of new issuance; initial stock from SAMA 2021 data)
    Gov_net_wealth: NDArray                     # Government liquid net worth (BSM-consistent) = Gov_ext_assets − Bond_dom − Bond_ext
    Gov_net_wealth_FULL: NDArray                # Government net worth (BSM-consistent) = Gov_ext_assets − Bond_dom − Bond_ext + E_firm_gov + E_bank_gov + OFA_Gov
    Gov_net_wealth_FULL_plus_aramco: NDArray         # Government net worth plus Aramco equity valuation
    OFA_Gov: NDArray                            # Gov Other Financial Assets stock (cumulative TFM residual; ΔOFA_Gov = Gov_surplus − ΔGov_liquid_NW − ΔE_firm_gov − ΔE_bank_gov)
    gov_fiscal_balance: NDArray                 # Government fiscal balance = Gov_rev − Gov_exp (positive = surplus, negative = deficit)
    Tax_profits: NDArray                        # Tax on profits
    Tax_wages: NDArray                          # Tax on wages
    Gov_confidence_boost: NDArray               # Government confidence boost in times of economic crisis ("state-led economy")







    #################################################################################################################################################
    # Growth rates
    #################################################################################################################################################
    gY: NDArray                                 # Growth rate of nominal final demand
    gy: NDArray                                 # Growth rate of REAL final demand
    gY_e: NDArray                               # Expected growth rate of nominal final demand
    gY_non_oil: NDArray                         # Growth rate of Nominal non-oil GDP proxy
    gY_oil: NDArray                             # Growth rate of Nominal oil GDP proxy
    gy_non_oil: NDArray                         # Growth rate of REAL non-oil GDP proxy
    gy_oil: NDArray                             # Growth rate of REAL oil GDP proxy
    gY_gov: NDArray                             # Growth rate of Nominal government activities GDP proxy
    gy_gov: NDArray                             # Growth rate of REAL government activities GDP proxy
    gK: NDArray                                 # Growth rate of NOMINAL capital stock
    gk: NDArray                                 # Growth rate of REAL capital stock
    gY_: NDArray                                # Sectoral growth rate of Nominal final demand
    gy_: NDArray                                # Sectoral growth rate of REAL final demand
    gY_e_: NDArray                              # Sectoral expected growth rate of final demand
    gX_: NDArray                                # Sectoral growth rate of NOMINAL TOTAL OUTPUT X
    gx_: NDArray                                # Sectoral growth rate of REAL TOTAL OUTPUT X
    gK_: NDArray                                # Sectoral growth rate of NOMINAL capital stock
    gk_: NDArray                                # Sectoral growth rate of REAL capital stock
    gI_endog_: NDArray                          # Endogenous growth rate of investment
    gI_trend_endog: NDArray                     # Endogenous trend growth rate of investment in case of excess investment
    gW: NDArray                                 # Growth rate of wages
    gIM_: NDArray                               # Growth rate of imports
    gV: NDArray                                 # Growth rate household wealth
    Y_growth_rate_forecast: NDArray              # MEAN Forecasted growth rate of GDP Output Y
    Inflation_forecast: NDArray                  # MEAN Forecasted growth Inflation based on non-oil GDP deflator (including OIL inflation explodes it all)
    Oil_growth_rate_forecast:NDArray            # Forecasted growth rate of oil exports (as proxied by Oil Activities (4) from ts_national)
    Exports_nonoil_growth_rate_forecast: NDArray # Forecasted growth rate of oil exports (as proxied by Oil Activities (4) from ts_national)
    GY_growth_rate_forecast: NDArray             # Forecasted growth rate of government consumption
    IM_growth_rate_forecast: NDArray             # Forecasted growth rate of imports
    domestic_increase_investment_growth_: NDArray  # Growth rate of investment to ENABLE domestic increase in production (X) by sector
    oil_exports_shock: NDArray                  # Oil exports shock as mean over sectoral oil exports
    alpha_2_increase_factor: NDArray            # Factor by which output GDP Y increased in period one, used to scale alpha_2 consumption toward the steady-state value

    #################################################################################################################################################
    # REAL VARIABLES - Physical quantities / real values (lowercase)
    #################################################################################################################################################
    x_: NDArray                                 # Real output
    u_: NDArray                                 # Real capacity utilization
    k_: NDArray                                 # Real capital stock
    water_use_: NDArray                         # Water use
    water_use_hh: NDArray                       # Household water use
    water_use_agr: NDArray                      # Agricultural water use
    water_use_ind: NDArray                      # Industrial water use
    water_use_ser: NDArray                      # Service water use
    water_use_total_national: NDArray                          # Total water use (scalar)
    water_use_intensities_: NDArray               # Water use intensity endogenous from crop mode (m3 per 1000 SAR of output)

    #################################################################################################################################################
    # Water Supply (Desalination & Wastewater)
    #################################################################################################################################################
    # Desalination
    K_w_no_desal: NDArray                       # Water capital stock (without desalination)
    K_w_desal: NDArray                          # Desalination capital stock
    k_desal: NDArray                            # Desalination capital stock
    k_desal_physical_yearly:NDArray             # Physical capital stock of installed desalination capacity in m3 installed capacity
    water_use_desal_tot:NDArray                 # Physical aggregate water use of desalinated water derived from capacity
    pot_desal: NDArray                          # Potential desalination supply based on capacity
    share_desal: NDArray                        # Share of desalination in total water supply (dynamic)

    # Wastewater
    K_w_no_wwater: NDArray                      # Water capital stock (without wastewater)
    K_w_wwater: NDArray                         # Wastewater reuse capital stock
    k_wwater: NDArray                           # Wastewater reuse capital stock
    k_wwater_physical_yearly:NDArray            # Physical capital stock of installed wastewater reuse capacity in m3 installed capacity
    water_use_wwater_tot:NDArray                # Physical aggregate water use of wastewater reuse derived from capacity
    pot_wwater: NDArray                         # Potential wastewater reuse supply based on capacity
    share_wwater: NDArray                       # Share of wastewater reuse in total water supply (dynamic)
    
    # Groundwater
    water_use_gw_tot:NDArray                    # Physical aggregate water use of groundwater (residual)
    water_use_scale_factor: NDArray             # Scale factor for water use when potential supply exceeds demand
    
    # Sectoral water supply mix
    water_use_desal_: NDArray                   # Sectoral desalinated water use
    water_use_wwater_: NDArray                  # Sectoral reused wastewater use
    water_use_gw_: NDArray                      # Sectoral groundwater use
    
    # Household water supply mix
    water_use_desal_hh: NDArray                 # Household desalinated water use
    water_use_wwater_hh: NDArray                # Household reused wastewater use
    water_use_gw_hh: NDArray                    # Household groundwater use

    #################################################################################################################################################
    # WATER USE
    #################################################################################################################################################
    water_use_: NDArray                         # Sectoral water use (aggregated by sector)
    water_use_sectoral_: NDArray                 # Detailed sectoral water use calculated from intensities
    water_use_hh: NDArray                       # Household water use

    #################################################################################################################################################
    # Energy Use Variables
    #################################################################################################################################################
    gas_use_: NDArray                           # Sectoral gas use in GJ
    electricity_use_: NDArray                   # Sectoral electricity use in GJ
    oil_use_: NDArray                           # Sectoral oil use in GJ
    gas_use_total: NDArray                      # Total gas use in GJ
    electricity_use_total: NDArray              # Total electricity use in GJ
    oil_use_total: NDArray                      # Total oil use in GJ
    
    #################################################################################################################################################
    # RENEWABLE ENERGY & FINAL ENERGY
    #################################################################################################################################################
    renewable_energy_generation: NDArray        # Total renewable energy generation in GJ
    re_gen_solar: NDArray                       # Solar generation in GJ
    re_gen_wind: NDArray                        # Wind generation in GJ
    re_gen_bioenergy: NDArray                   # Bioenergy generation in GJ
    renewable_energy_share: NDArray             # Share of renewables in total energy demand (%)
    total_energy_demand: NDArray                # Total energy demand (gas+oil+elec) in GJ
    energy_reduction_factor_: NDArray           # Sectoral energy reduction factor (0.0 to 0.9)    # Final energy use after substitution by renewables
    electrification_added_load: NDArray         # Electricity (GJ) added due to fossil fuel switching

    #################################################################################################################################################
    #  Energy efficiency
    #################################################################################################################################################
    I_efficiency_: NDArray                      # Sectoral investment in energy efficiency (Thousand SAR)
    I_efficiency_total: NDArray                 # Total investment in energy efficiency (Thousand SAR)
    I_electrification_total: NDArray            # Total investment in Industrial Electrification (Switching Oil/Gas to Elec)
    marginal_cost_of_efficiency_: NDArray       # The current cost to save 1 more GJ/yr
 
    re_generation_counterfactual_base: NDArray  # Stores the RE generation base from the target year, as if no efficiency gains occurred
    oil_use_final_: NDArray                     # Final sectoral oil use after RE substitution (GJ)
    gas_use_final_: NDArray                     # Final sectoral gas use after RE substitution (GJ)
    oil_reduction_factor_: NDArray              # Sectoral oil reduction factor (0.0 to 0.95)
  
    # RENEWABLE INVESTMENT & COST
    I_RE: NDArray                               # Total investment in new renewable capacity (Thousand SAR)
    k_re_physical: NDArray                      # Physical capital stock of renewables (GJ/year capacity)
    # --- Segregated Renewable Investments ---
    I_solar: NDArray                            # Investment in Solar PV (Thousand SAR)
    I_wind: NDArray                             # Investment in Wind (Thousand SAR)
    I_bio: NDArray                              # Investment in Bioenergy (Thousand SAR)
    # Cost of Transition Analysis
    cost_of_ff_substituted_: NDArray            # Sectoral cost of fossil fuels that were replaced
    cost_of_re_used_: NDArray                   # Sectoral cost of the renewable energy used as a substitute
    net_cost_of_transition_: NDArray            # Net cost difference for each sector

    gas_emissions_: NDArray                     # Sectoral gas emissions in MtCO2
    electricity_emissions_: NDArray             # Sectoral electricity emissions in MtCO2
    oil_emissions_: NDArray                     # Sectoral oil emissions in MtCO2
    total_emissions: NDArray                    # Aggregate total emissions (MtCO2)

    gas_emissions_total: NDArray                # Total gas emissions in MtCO2
    electricity_emissions_total: NDArray        # Total electricity emissions in MtCO2
    oil_emissions_total: NDArray                # Total oil emissions in MtCO2

    # NET ZERO VARIABLES
    emissions_gross_total: NDArray              # Total emissions BEFORE removal
    emissions_net_total: NDArray                # Total emissions AFTER removal
    
    # Circular Carbon Economy Flows (MtCO2) - MACRO VARIABLES (No Underscore)
    flow_reduce: NDArray                        # Abated via efficiency/renewables
    flow_reuse_recycle: NDArray                 # Captured via CCU/EOR
    flow_remove_ccs: NDArray                    # Captured via CCS
    flow_remove_nature: NDArray                 # Sequestered via Nature
    
    # Investments (Thousand SAR) - SECTORAL VARIABLES (With Underscore)
    I_netzero_ccs_: NDArray                     # Investment in CCS infrastructure
    I_netzero_nature_: NDArray                  # Investment in Nature/SGI
    I_netzero_ccu_: NDArray                     # Investment in CCU/Recycling
    I_carbon_credits_: NDArray                  # Spending on Carbon Credits
    
     # Carbon Market
    carbon_price: NDArray                       # Dynamic carbon price
    I_carbon_credit_revenue_: NDArray           # Income from selling credits (Nature/CCS projects)
# --- HYDROGEN VARIABLES ---
    k_hydrogen_physical: NDArray                # Physical Capacity of Electrolyzers (GJ output/year)
    I_hydrogen: NDArray                         # Total Investment in Hydrogen (Thousand SAR)
    I_hydrogen_: NDArray                        # Sectoral demand for H2 investment (who builds it?)
    
    hydrogen_production: NDArray                # Total H2 Produced (GJ)
    hydrogen_domestic_use: NDArray              # H2 used in KSA industry (GJ)
    hydrogen_export: NDArray                    # H2 exported (GJ)
    re_used_for_hydrogen: NDArray               # Renewable Electricity consumed by Electrolyzers (GJ)
# For printing energy results
    # Energy Demand
    total_energy_demand: NDArray  # Total Energy (GJ)
    gas_use_total: NDArray        # Total Gas (GJ)
    oil_use_total: NDArray        # Total Oil (GJ)
    electricity_use_total: NDArray # Total Electricity (GJ)
    renewable_energy_share: NDArray # % Share of RE
    
    # Renewable Generation Breakdown
    renewable_energy_generation: NDArray
    re_gen_solar: NDArray
    re_gen_wind: NDArray
    re_gen_bioenergy: NDArray
    k_re_physical: NDArray        # Physical Capacity
    
    # Hydrogen
    k_hydrogen_physical: NDArray
    hydrogen_production: NDArray
    hydrogen_domestic_use: NDArray
    hydrogen_export: NDArray
    re_used_for_hydrogen: NDArray
    
    # Emissions
    total_emissions: NDArray      # Gross Emissions
    emissions_net_total: NDArray  # Net Emissions
    emissions_gross_total: NDArray
    oil_emissions_total: NDArray
    gas_emissions_total: NDArray
    electricity_emissions_total: NDArray
    
    # Net Zero Flows
    flow_reuse_recycle: NDArray
    flow_remove_ccs: NDArray
    flow_remove_nature: NDArray
    
    # Investments (Financial)
    I_RE: NDArray                 # Renewable CapEx
    I_hydrogen: NDArray           # Hydrogen CapEx
    I_efficiency_total: NDArray   # Energy Efficiency CapEx
    I_electrification_total: NDArray # Electrification CapEx
    
    # Sectoral Arrays (Vector size S)
    # Note: These are arrays of vectors (T x S), ensure initialization handles (T, S)
    gas_use_: NDArray
    oil_use_: NDArray
    electricity_use_: NDArray
    oil_use_final_: NDArray
    gas_use_final_: NDArray
    energy_reduction_factor_: NDArray
    
    I_hydrogen_: NDArray
    I_netzero_ccs_: NDArray
    I_netzero_nature_: NDArray
    
    oil_emissions_: NDArray
    gas_emissions_: NDArray
    electricity_emissions_: NDArray
    
    I_carbon_credits_: NDArray
    I_carbon_credit_revenue_: NDArray
    
    electrification_added_load: NDArray
    marginal_cost_of_efficiency_: NDArray
    I_solar: NDArray
    I_wind: NDArray
    I_bio: NDArray

    # --- Additional model variables ---
    I_netzero_ccu_: NDArray
    green_export_total: NDArray
    EX_non_oil_total: NDArray

    k_storage_physical: NDArray       # Capacity in GJ
    I_storage: NDArray                # Investment in Billion/Thousand SAR
    I_storage_local: NDArray          # Local portion
    I_storage_import: NDArray         # Imported portion
    storage_coverage_pct: NDArray     # Policy target (0 to 100%)
    storage_local_share: NDArray      # Localization %

    #################################################################################################################################################
    # Foreign Direct Investment (FDI) variables
    #################################################################################################################################################
    # Sectoral FDI variables (T × S arrays, suffix "_")
    FDI_net_flow_: NDArray              # Sectoral net FDI inflows (Thousand SAR) — the portion of I_ financed by foreign capital
    FDI_gross_flow_: NDArray            # Sectoral gross FDI inflows (Thousand SAR)
    FDI_stock_: NDArray                 # Sectoral cumulative FDI stock (Thousand SAR)
    I_domestic_: NDArray                # Sectoral domestically-financed investment = I_ - FDI_net_flow_ (Thousand SAR)

    # Macro (aggregate) FDI variables (T arrays, no suffix)
    FDI_net_total: NDArray              # Total net FDI inflows = sum(FDI_net_flow_) (Thousand SAR)
    FDI_gross_total: NDArray            # Total gross FDI inflows = sum(FDI_gross_flow_) (Thousand SAR)
    FDI_stock_total: NDArray            # Total gross FDI stock = sum(FDI_stock_) (Thousand SAR)
    FDI_net_stock_: NDArray             # Sectoral cumulative net FDI stock (net inflows only, no gross amplification)
    FDI_net_stock_total: NDArray        # Total net FDI stock = sum(FDI_net_stock_) (Thousand SAR) — used for income repatriation
    FDI_share_of_investment: NDArray    # Share of total investment financed by FDI = FDI_net_total / I_total

    # Remittances (T arrays, no suffix)
    remittances: NDArray                # Personal remittances paid (outflow from Saudi Arabia, 1000 SAR)
    remittances_growth: NDArray         # Applied growth rate of remittances at time t

    #################################################################################################################################################
    # Balance of Payments variables (BPM6 structure)
    #################################################################################################################################################
    # Current account components
    trade_balance: NDArray              # Trade balance = sum(EX_oil_ + EX_non_oil_) - sum(IM_) (1000 SAR)
    FDI_income_payments: NDArray        # FDI profit income repatriated abroad (outflow); capped net-stock based (1000 SAR)
    primary_income_balance: NDArray     # Primary income balance = reserve_income - FDI_income_payments (1000 SAR) — Phase 2
    secondary_income_balance: NDArray   # Secondary income balance = -remittances (1000 SAR)
    current_account: NDArray            # Current account = trade_balance + primary_income + secondary_income (1000 SAR)
    # Capital account (BPM6 definition: debt forgiveness, investment grants — negligible for Saudi)
    capital_account: NDArray            # Capital account ≈ 0 for Saudi Arabia (1000 SAR)
    # Financial account components
    financial_account: NDArray          # Financial account = -current_account (since capital_account ≈ 0) (1000 SAR)
    # Government Gross External Asset Stock (clearing variable for BoP)
    # Represents SAMA reserves + PIF foreign holdings (gross; liabilities tracked separately in Bond_domestic, Bond_external)
    # All forex flows (trade surplus, FDI inflows, remittance outflows) accumulate here
    Gov_ext_assets: NDArray             # Government gross external asset stock (1000 SAR): SAMA reserves + PIF foreign holdings
    Gov_ext_assets_change: NDArray      # Period change in Gov_ext_assets (1000 SAR)
    Gov_ext_assets_income: NDArray      # Income (+) or cost (−) on Gov_ext_assets stock = r_gov_ext_assets × Gov_ext_assets[t−1] (1000 SAR)
    PIF_income: NDArray                 # PIF return income: r_reserve_income × Gov_net_wealth_plus_aramco[t−1] when > 0, else 0 (1000 SAR)
    # BoP identity check (should equal zero)
    BoP_check: NDArray                  # CAB + capital_account - financial_account, should ≈ 0
    Gov_pif_investment: NDArray         # Government fiscal surplus channelled into PIF (NFA_gov): gov_surplus_pif_rate * max(Gov_rev - Gov_exp, 0) (1000 SAR)
    # Two-phase sovereign financing (activated only when Gov_ext_assets would cross zero)
    # Phase 1 (normal): external deficit financed by drawing down SAMA reserves / PIF (Gov_ext_assets ≥ 0)
    # Phase 2 (cutoff): once reserves exhausted, government issues sovereign bonds to RoW at interest_gov_bonds (4%)
    # Empirical basis: Saudi Arabia began issuing international sovereign bonds in 2016 precisely as a low-oil-price hedge.
    # In Godley-Lavoie SFC terms this is a switch from 'reduction of foreign assets' to 'incurrence of foreign liabilities'.
    # Bond_external is inert (=0) whenever Gov_ext_assets stays positive — no impact on baseline results.
    Bond_external: NDArray              # Stock of sovereign bonds issued to rest-of-world (1000 SAR); zero until NFA_gov floor is hit
    Bond_external_interest_cost: NDArray  # Period interest payments on Bond_external = interest_gov_bonds × Bond_external[t-1] (1000 SAR)
    Bond_external_repayment: NDArray    # Period principal repayment on Bond_external = ρ_ext × Bond_external[t-1] (1000 SAR); enters Gov_exp as a cash outflow
    Bond_external_routine_issuance: NDArray  # Routine external bond issuance = (1 − domestic_bond_share) × raw fiscal deficit (1000 SAR); follows the fixed 60/40 domestic/external split

    def __init__(self, T: int, pc: ParametersCalibrated):

        T = T + 1  # Add one timestep for initial values

        # Initialize all variables to zero or starting value
        for var in self.__dataclass_fields__:
            if var.endswith("__"):
                setattr(self, var, np.zeros([T, pc.S, pc.S]))
            elif var.endswith("_"):
                setattr(self, var, np.zeros([T, pc.S]))
            else:
                setattr(self, var, np.zeros(T))
        for k, v in pc.get_starting_values().items():
                    # Assign value 'v' to the first time step (index 0) of variable 'k'
                    getattr(self, k)[0] = v


@dataclass
class ModelResults:
    """Results of a model simulation run."""

    config: ModelConfig
    macro: pd.DataFrame
    sectoral: Dict[str, pd.DataFrame]
    iomatrix: Dict[str, Dict[int, pd.DataFrame]]  # Dictionary of IO matrices by time step

    def save_stable_outcome_vars(self):
        """Save key variables of simulation."""

        # Loan level corresponding to investment and loan repayment
        np.save('temp_data/L.npy',
                self.sectoral['L'].iloc[-1].to_numpy())

        # Income level corresponding to demand factors
        np.save('temp_data/YD_profit.npy', self.macro['YD_profit'].iloc[-1])
