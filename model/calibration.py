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

import pandas as pd
from io import StringIO
import numpy as np
import math


from .utils import libreoffice_int_to_datetime, to_float_or_nan
from .ar_model_estimation import estimate_ar_model, estimate_ar_model_log, estimate_ar_model_growthrates, forecast, determine_optimal_lag_order, determine_optimal_lag_order_log, determine_optimal_lag_order_growthrates, identify_structural_break
from .water_module import get_agr_data

class ParametersCalibrated:
    """Calibrated input parameters of the model."""



    def __init__(self, parameters, verbose=False, scenario_config: dict = None):




        # Define the time frame of the calibration
        startyear = 2013
        endyear = 2023
        years = endyear - startyear

        ###############################################################################################################################
        ###############################################################################################################################
        ###############################################################################################################################
        # I. Data loading and IO table construction
        ###############################################################################################################################
        ###############################################################################################################################
        ###############################################################################################################################

        ###############################################################################################################################
        # 1.1 Load and preprocess data
        ###############################################################################################################################

        # Desalination share calibrated from water flow data (share of desalination in total water use)
        self.desal_share_of_water_use = 0.1823
        # Wastewater share used only if Add_new_wastewater_sector is True (inactive in standard runs)
        self.wwater_share_of_water_use = 0.1

        # The Sewerage sector from the IO table is used directly as the wastewater sector.
        # This avoids modifying the IO table structure and is the preferred approach.
        self.Add_new_wastewater_sector = False

        # Input-output table
        # Columns and rows are sectors
        io_table = self.io_table = self.prepare_io_table()
        self.sectors = list(io_table.columns)
        self.sector_other_personal_service = self.sectors.index(
            'Other personal service activities')
        self.sectors = list(io_table.columns)

        # Capital productivity
        self.eK_ = np.ones(len(self.sectors)) * parameters.eK

        # Macro vars 1
        # Columns are variables, rows are sectors
        df = self.read_csv(
            'input_data/io_macro_vars.csv')
        self.macro_vars_orig = df.copy()

        # Drop Activities of households as employers of domestic personnel after adding it to Sector 83 "Other personal service activities"
        df.loc[83] += df.loc[84]
        df.drop(84, inplace=True)

        # Split water into water & desalination
        self.sector_desal = self.sectors.index('Desalination')
        self.sector_water_name = "Water collection, treatment and supply"
        self.sector_water = self.sectors.index(self.sector_water_name)
        df = self.add_row(df)
        water_row = df.loc[self.sector_water].copy()
        df.loc[self.sector_desal] = np.zeros(len(water_row))
        x_desal = (
            df.at[self.sector_water, "Total Output"] *
            self.desal_share_of_water_use
        )
        df.loc[self.sector_desal, "Total Output"] = x_desal
        df.loc[self.sector_desal, "Total Intermediate Consumption"] = x_desal
        df.loc[self.sector_desal, "Compensation of employees"] = \
            df.loc[self.sector_water, "Compensation of employees"]
        ########################################################################
        # 1.2 Wastewater sector addition
        ########################################################################
        if self.Add_new_wastewater_sector:
            self.sector_wwater = self.sectors.index('Wwater')
            df = self.add_row(df)
            water_row = df.loc[self.sector_water].copy()
            df.loc[self.sector_wwater] = np.zeros(len(water_row))
            x_wwater = (
                df.at[self.sector_water, "Total Output"] *
                self.wwater_share_of_water_use
            )
            df.loc[self.sector_wwater, "Total Output"] = x_wwater
            df.loc[self.sector_wwater,
                   "Total Intermediate Consumption"] = x_wwater
            df.loc[self.sector_wwater, "Compensation of employees"] = \
                df.loc[self.sector_water, "Compensation of employees"]

        # Standard case: use the existing Sewerage sector as the wastewater sector.
        else:
            self.sector_wwater = self.sectors.index('Sewerage')

        # Financial sector index (used for optional price stabilisation, see Fix_finance_sector_price switch)
        self.sector_finance = self.sectors.index(
            'Financial service activities, except insurance and pension funding')

        desal_data = self.macro_vars = df
        wwater_data = self.macro_vars = df
        # Macro vars 2
        # Columns and rows are variables
        df = pd.read_csv('input_data/io_macro_vars_2.csv',
                         delimiter=";", index_col=0)
        df = df.map(to_float_or_nan)
        self.macro_vars_2 = df
        self.macro_vars_names = self.macro_vars.columns.tolist()

        # Read in time series data and other calibration data,
        # multiply by 1000 to have same units as IO table
        self.ts_energy = pd.read_csv(
            "input_data/ts_energy.csv", index_col=0) * 1000
        df = pd.read_csv(
            "input_data/ts_investment.csv", index_col=0) * 1000
        df.index = df.index.str.replace(' p', '').astype(int)
        self.ts_investment = df
        self.ts_government = pd.read_csv(
            "input_data/ts_government.csv", index_col=0) * 1000
        self.ts_monetary = pd.read_csv(
            "input_data/ts_monetary.csv", index_col=0) * 1000
        self.ts_national = pd.read_csv(
            "input_data/ts_national.csv", index_col=0) * 1000
        df = pd.read_csv("input_data/ts_bank_credits.csv",
                         delimiter=";", index_col=0)
        df.index = df.index.map(libreoffice_int_to_datetime)
        df = df.map(to_float_or_nan)
        self.ts_bank_credits = df * 1000
        df = pd.read_csv("input_data/ts_company_credits.csv", delimiter=";")
        df = df.set_index(["End of Period", "Year"])
        df = df.map(to_float_or_nan)
        self.ts_company_credits = df * 1000
        self.ts_exports = pd.read_csv(
            "input_data/ts_exports.csv", index_col=0) * 1000

        ###############################################################################################################################
        # 1.3 Extract initial data from IOTs and calculate shares
        ###############################################################################################################################
        self.X_ = desal_data['Total Output'].to_numpy()
        self.Y_finaldemand_ = desal_data['Final Demand'].to_numpy()
        self.AX_ = self.IntS_ = desal_data['Total Intermediate Consumption'].to_numpy(
        )

        self.gross_capital_formation = desal_data['Gross capital formation'].to_numpy(
        )
        self.I_total = np.sum(self.gross_capital_formation)
        self.DEPR_ = desal_data['Consumption of fixed capital formation'].to_numpy(
        )
        self.EX_ = desal_data['Total Export'].to_numpy()
        self.IM_ = desal_data['Total imports'].to_numpy()
        self.C_ = desal_data["Households final consumption expenditures"] + \
            desal_data["Non profit institutions serving household final consumption expenditures"]
        self.C_residents = self.macro_vars_2.loc["Direct purchases abroad by residents",
                                                 "Households final consumption expenditures"]

        self.C_non_residents = self.macro_vars_2.loc["Direct purchases in domestic markets by non-residents ",
                                                     "Households final consumption expenditures"]

        self.G_ = desal_data["Government final consumption expenditures"]
        self.P_ = desal_data["Gross operating surplus"].to_numpy()
        self.IntP_data_sum_ = desal_data["Primary inputs at basic prices"].to_numpy(
        )

        ########################################################################
        # 1.4 Sectoral shares of final demand (C, G, I; X and K later)
        ########################################################################

        # Households, Investment, Government
        self.dC = self.C_ / self.C_.sum()
        self.dG = self.G_ / self.G_.sum()
        self.dI = self.gross_capital_formation / self.I_total



        # Production approach to calculate GDP, preferred because no negative values for some sectors.
        # IO Equation: X = AX + Y
        # Final Demand = Y = Final Consumption (C + G) + Capital Formation (I) + Export - Import
        # Expenditure approach yields negative values for some sectors (due to import structure), so not used.
        self.Y_expenditure_ = self.C_ + self.G_ + \
            self.dI * self.I_total + self.EX_ - self.IM_

        if parameters.Y_GDP_Expenditure_X_harmonization:
            self.Y_ = self.X_ - self.IntS_
            self.IM_ = self.Y_ - (self.C_ + self.G_ +
                                  self.dI * self.I_total + self.EX_)
        else:
            self.Y_ = self.X_ - self.IntS_
        self.X_Y_ratio_ = np.where(
            np.isinf(self.X_ / self.Y_), 0, self.X_ / self.Y_)

        # Calculate sectoral share of X in total here after correction
        self.dX = self.X_ / self.X_.sum()

        # define total profits before financial sector profits etc. are deducted
        self.P_tot_ = self.P_.copy()
        self.W_ = desal_data["Compensation of employees"].to_numpy()

        self.wage_share_in_X_ = self.W_ / self.X_

        # Ensure positivy entry in IntP_ for all sectors to avoid
        self.tax_products_net_ = desal_data["Net tax on products"].to_numpy()
        self.tax_production_ = desal_data["Other taxes on production"].to_numpy(
        )
        self.sub_production_ = desal_data["Other subsidies on production"].to_numpy(
        )
        self.tax_investments = self.macro_vars_2.loc["Net tax on products",
                                                     "Gross capital formation"]
        # Calculate sectoral tax rates
        self.tau_sub_production_ = self.sub_production_ / self.X_
        self.tau_tax_production_ = self.tax_production_ / self.X_
        self.tau_tax_products_net_ = self.tax_products_net_ / self.X_
        # Tax investments is homogenous across all sectors, thus only one rate

        self.tau_tax_investments = self.tax_investments / self.I_total
        self.tax_products_net_total = 197854551

        self.GVA_ = self.P_ + self.W_ - self.sub_production_ + self.tax_production_

        # Define capital stock for capital share (using the X to K ratio captured by KX parameter, with setting KX=3 according to the standard value in the literature).
        self.K_ = self.X_ * parameters.KX
        # Calculate sectoral share of K in total here after correction of X
        self.dK = self.K_ / np.sum(self.K_)

        # Test that identity holds: X = AX + Y
        # Total Output = X = Internal Consumption (AX) + Final Demand (Y)
        assert np.allclose(self.X_, self.AX_ + self.Y_), "Invalid IO data"

        # Mathematical helpers
        self.S = self.X_.shape[0]  # Number of sectors
        self.I = np.eye(self.S)  # Identity matrix

        ###############################################################################################################################
        ###############################################################################################################################
        ###############################################################################################################################
        # II. Water and desalination calibration
        ###############################################################################################################################
        ###############################################################################################################################
        ###############################################################################################################################

        ########################################################################
        # 2.1 Water use and desalination
        ########################################################################

        # Desalination
        # Append desalination sector
        # Intensities are in m3/tSAR
        self.water_use_intensities_ = \
            pd.read_csv("input_data/water_use/WU_intensity_85_TO.csv",
                        index_col=0, header=0).to_numpy()[0] * 1000
        self.water_use_intensities_ = \
            self.water_use_intensities_[:-1]  # remove hh sector
        self.water_use_intensities_ = \
            np.append(self.water_use_intensities_, 0)  # add desal sector
        if self.Add_new_wastewater_sector:
            self.water_use_intensities_ = \
                np.append(self.water_use_intensities_,
                          0)  # add wwater sector but ONLY if new sector is created
        # Add household consumption of water
        self.water_use_hh_2021 = 1.825 * 1e9
        self.total_water_use_initial = np.sum(
            # Add water consumption of households manually
            self.X_ * self.water_use_intensities_) + self.water_use_hh_2021

        # Calculate initial water price in tSAR/m3
        water_output_tsar = self.X_[self.sector_water]
        water_output_sar = water_output_tsar
        water_output_m3 = self.total_water_use_initial
        self.water_price = water_output_sar / water_output_m3

        desal_op_ex_shares = {
            "Electricity, gas, steam and air conditioning supply": 0.40,
            "Manufacture of chemicals and chemical products": 0.15,
            "Manufacture of machinery and equipment n.e.c.": 0.15,
            "Repair of computers and personal and household goods": 0.05,
            "Services to buildings and landscape activities": 0.10,
            "Activities of head offices; management consultancy activities": 0.05,
            "Civil engineering": 0.10
        }

        # Reduce inputs of water sector by desal share
        io_table[self.sector_water_name] = (
            io_table[self.sector_water_name] *
            (1-self.desal_share_of_water_use)
        )

        # Add desalinated water as input to the water sector
        io_table.at[self.sector_desal, self.sector_water_name] = \
            self.desal_share_of_water_use * self.X_[self.sector_water]

        # Desalination data
        # expenditures (self.cap_ex, self.op_ex) given in thousandSAR/year
        # capacities (capacity, capacity_cum) given in m3/year

        desal_cap_ex_shares = {
            # Sektor No. 62
            "Architectural and engineering activities; technical testing and analysis": 15,
            # Sektor No. 37
            "Construction of buildings": 35,
            # Sektor No. 38
            "Civil engineering": 0,
            # Sektor No. 26
            "Manufacture of machinery and equipment n.e.c.": 35,
            # Sektor No. 25
            "Manufacture of electrical equipment": 5,
            # Sektor No. 23
            "Manufacture of fabricated metal products, except machinery and equipment": 5,
            # Sektor No. 39
            "Specialized construction activities": 5
        }
        self.dI_desal_ = np.array(
            [desal_cap_ex_shares[sector] if sector in desal_cap_ex_shares else 0 for sector in self.sectors])

        # as desalinated water is used as intermediary input for water sector
        self.desal_use = {
            "industry": 0.1796806632,
            "agriculture": 0.00527973427,
            "military": 0.01082972419,
            "HH": 0.8006797181,
            "other": 0.00353016018
        }

        ts_desal = pd.read_csv(
            'input_data/water_use/ts_desal_2.csv', index_col=0, delimiter=";").map(to_float_or_nan)
        desal_data = ts_desal.loc[2021]

        # Average investment tSAR / year
        self.I_desal = ts_desal["CapEx (M SAR)"].loc[2013:2023].sum(
        ) / 10 * 1e3

        # Operational costs tSAR / year
        op_ex = desal_data["OpEx (M SAR/year)"] * 1e3

        # Desalination capital capacity m3/year
        K_desal_m3 = desal_data["Cumulative Capacity (Mm3/d)"] * 365 * 1e6

        # Desalination water flow data for 2021 in m3/year
        X_desal_m3 = 2.6 * 1e9  # Empirical data (2021)
        self.u_desal = X_desal_m3 / K_desal_m3

        # Price of desalinated water tSAR / m3
        p_desal_m3 = self.p_desal_m3 = op_ex / X_desal_m3

        # Investment tSAR / (m3/y)
        I_per_X_m3 = desal_data["CapEx (M SAR)"] / \
            (desal_data["Capacity (Mm3/d)"] * 365) / 1e3
        X_m3_per_I = 1 / I_per_X_m3

        # Capital productivity tSAR water output / tSAR capital investment
        eK_desal = self.eK_desal = X_m3_per_I * p_desal_m3

        # Water output tSAR / year
        X_desal = X_desal_m3 * p_desal_m3
        K_desal = self.K_desal = K_desal_m3 / eK_desal * p_desal_m3

        # Operational costs tSAR / water output tSAR == 1
        opex_intensity = op_ex / X_desal

        # Average lifetime of desalination plants in years
        desalination_plants_lifetime = 25

        # Depreciation for desalination capital stock according to average plant lifetime
        self.δ_desal = 1 / desalination_plants_lifetime

        # Use empirical value for desalination capital
        self.K_[self.sector_desal] = K_desal
        self.eK_[self.sector_desal] = eK_desal

        # Calculate average desalination cumulative capacity installed growth rate
        desal_cum_cap_ts = ts_desal["Cumulative Capacity (Mm3/d)"].ffill()
        value_2013 = desal_cum_cap_ts[startyear]
        value_2023 = desal_cum_cap_ts[endyear]
        self.gDesalCap_avg = (value_2023 / value_2013) ** (1 / years) - 1

        # Scenario-specific adjustments for Desalination growth
        if scenario_config:
            if scenario_config.get('Transformation_scenario'):
                self.gDesalCap_avg = 0.10  # Higher growth for Transformation
            elif scenario_config.get('Vision_2030_scenario'):
                self.gDesalCap_avg = 0.08  # High growth for Vision 2030/Net Zero
            # Baseline remains at historical average (gDesalCap_avg is 6.64% from 2013 to 2023, is 6.83% from 2018 to 2023)

        # Add operational expenditure for desalination to the IO table
        # as input column of desalination sector
        self.a_desal = np.array(
            [opex_intensity * desal_op_ex_shares[sector]
             if sector in desal_op_ex_shares else 0 for sector in self.sectors]
        )
        io_table["Desalination"] = self.a_desal


        ########################################################################
        # 2.2 Wastewater sector
        ########################################################################

        ts_wwater = pd.read_csv(
            'input_data/water_use/KEN_Wastewater_data_3.csv', sep=";", decimal=",")
        ts_wwater = ts_wwater.set_index("year")
        wwater_data = ts_wwater.loc[2021]

        # Average investment tSAR / year
        self.I_wwater = ts_wwater["cap_ex"].loc[2021:2023].sum() / 10 * 1e3

        # Operational costs tSAR / year
        op_ex = wwater_data["op_ex"] * 1e3

        # Wastewater capital capacity m3/year
        K_wwater_m3 = wwater_data["WW treatment Capacity (Mm3/day)"] * \
            365 * 1e6

        # Wastewater water flow data for 2021 in m3/year
        X_wwater_m3 = 0.42 * 1e9  # Empirical data (2021)
        self.u_wwater = X_wwater_m3 / K_wwater_m3

       # Price of wastewater tSAR / m3
        p_wwater_m3 = self.p_wwater_m3 = op_ex / X_wwater_m3

        # Investment tSAR / (m3/y)
        I_per_X_m3 = wwater_data["cap_ex"] / \
            (wwater_data["WW treatment Capacity (Mm3/day)"] * 365) / 1e3
        X_m3_per_I = 1 / I_per_X_m3

        # Capital productivity tSAR water output / tSAR capital investment
        eK_wwater = self.eK_wwater = X_m3_per_I * p_wwater_m3

        # Water output tSAR / year
        X_wwater = X_wwater_m3 * p_wwater_m3
        K_wwater = self.K_wwater = K_wwater_m3 / eK_wwater * p_wwater_m3

        # Operational costs tSAR / water output tSAR == 1
        opex_intensity = op_ex / X_wwater

        # Average lifetime of wastewater plants in years
        wwater_plants_lifetime = 25

        # Depreciation for wastewater capital stock according to average plant lifetime
        self.δ_wwater = 1 / wwater_plants_lifetime

        # Use empirical value for wastewater capital
        self.K_[self.sector_wwater] = K_wwater
        self.eK_[self.sector_wwater] = eK_wwater

        # Calculate average wastewater cumulative capacity installed growth rate
        wwater_cum_cap_ts = ts_wwater["WW treatment Capacity (Mm3/day)"].ffill(
        )
        wwater_startyear = 2021
        value_2013 = wwater_cum_cap_ts[startyear]
        value_2021 = wwater_cum_cap_ts[wwater_startyear]
        value_2023 = wwater_cum_cap_ts[endyear]
        years_wwater_use = endyear - wwater_startyear
        # Use 2021 as starting year for a more realistic wastewater capacity growth rate
        self.gWWaterCap_avg = (
            value_2023 / value_2021) ** (1 / years_wwater_use) - 1

        # Scenario-specific adjustments for Wastewater growth
        if scenario_config:
            if scenario_config.get('Transformation_scenario'):
                self.gWWaterCap_avg = 0.08  # High growth for Transformation
            elif scenario_config.get('Vision_2030_scenario'):
                self.gWWaterCap_avg = 0.04  # Moderate growth for Vision 2030
            # Baseline remains at historical average (gWWaterCap_avg is 2.06% from 2021 to 2023)

        # Add operational expenditure for wastewater to the IO table
        # as input column of wastewater sector
        wwater_op_ex_shares = {
            "Electricity, gas, steam and air conditioning supply": 0.40,
            "Manufacture of chemicals and chemical products": 0.20,
            "Water collection, treatment and supply": 0.10,
            "Repair and installation of machinery and equipment": 0.10,
            "Waste collection, treatment and disposal activities; materials recovery": 0.05,
            "Human health activities": 0.15
        }
        self.a_wwater = np.array(
            [opex_intensity * wwater_op_ex_shares[sector]
             if sector in wwater_op_ex_shares else 0 for sector in self.sectors]
        )

        if self.Add_new_wastewater_sector:
            io_table["Wwater"] = self.a_wwater

            # Reduce inputs of water sector by wastewater share
            io_table[self.sector_water_name] = (
                io_table[self.sector_water_name] *
                (1 - self.wwater_share_of_water_use)
            )

            # Add wastewater as input to the water sector
            io_table.at[self.sector_wwater, self.sector_water_name] = \
                self.wwater_share_of_water_use * self.X_[self.sector_water]

        # Capital expenditure shares for wastewater
        wwater_cap_ex_shares = {
            "Manufacture of other transport equipment": 5,
            "Construction of buildings": 30,
            "Civil engineering": 5,
            "Manufacture of machinery and equipment n.e.c.": 20,
            "Mining support service activities": 5,
            "Manufacture of fabricated metal products, except machinery and equipment": 10,
            "Manufacture of chemicals and chemical products": 15,
            "Manufacture of electrical equipment": 5,
            "Specialized construction activities": 5
        }
        self.dI_wwater_ = np.array(
            [wwater_cap_ex_shares[sector] if sector in wwater_cap_ex_shares else 0 for sector in self.sectors])

        ###############################################################################################################################
        ###############################################################################################################################
        ###############################################################################################################################
        # III. IO coefficients, production, and price calibration
        ###############################################################################################################################
        ###############################################################################################################################
        ###############################################################################################################################

        ########################################################################
        # 3.1 IO technical coefficients (A matrix)
        ########################################################################

        # Technical coefficient matrix A (input-output)
        self.A = io_table.values * (1/self.X_)
        self.A_sum = np.sum(self.A, axis=0)
        self.IntP_ = self.A_sum * self.X_

        ########################################################################
        # 3.2 Productivity coefficients (limitational production function)
        ########################################################################

        # Set zero productivity coefficients very high so that they do not limit production
        self.alpha_ = np.maximum(self.Y_ / self.W_, 0)
        self.beta_ = np.maximum(self.Y_ / self.IntP_, 0)
        self.kappa_ = np.maximum(self.Y_ / self.K_, 0)

        ########################################################################
        # 3.3 Markup θ
        ########################################################################
        # Calibrate theta directly to share of P in (X - profits) to reflect ALL costs of firm
        # It also makes sense behaviorally, as markup will follow profit expectations, now set to P_[t-1]
        # In the first period nominal X equals real x, since prices are calibrated to 1.
        self.θ_ = self.P_ / (self.X_ - self.P_)
        self.θ_[np.isinf(self.θ_)] = 0

        self.p_ = np.ones(self.S)  # Initialize prices to 1

        ########################################################################
        # 3.4 Oil exports and sectoral definitions
        ########################################################################
        # Define oil sectors, and define an oil export vs. non-oil export sector
        # the exports of crude and refined oil sectors.

        self.crude_oil_sector = self.sectors.index(
            'Extraction of crude petroleum and natural gas')
        self.refined_oil_sector = self.sectors.index(
            'Manufacture of coke and refined petroleum products')
        self.sectors_oil_ = np.zeros(self.S)
        oil_sectors = [self.crude_oil_sector, self.refined_oil_sector]
        for i in oil_sectors:
            self.sectors_oil_[i] = 1
        self.non_oil_sectors_ = 1 - self.sectors_oil_
        self.EX_oil_ = self.EX_ * self.sectors_oil_
        self.EX_non_oil_ = self.EX_ * self.non_oil_sectors_

        ########################################################################
        # 3.4b National accounts GDP activity breakdown (GASTAT definition)
        # Ratios calibrated from ts_national 2021 base year to match official
        # "Gross Domestic Product by Main Activity At Current Prices":
        #   Non-Oil Activities = 48.28%, Oil Activities = 28.06%,
        #   Government Activities = 17.62%, Net Taxes on Products = 6.04%
        ########################################################################
        _ts_nat_2021 = self.ts_national.loc[2021]
        _gdp_total_nat = _ts_nat_2021['Gross Domestic Product (4)']
        self.ratio_non_oil_gdp    = _ts_nat_2021['Non-Oil Activities (4)']   / _gdp_total_nat  # ~0.4828
        self.ratio_oil_gdp        = _ts_nat_2021['Oil Activities (4)']        / _gdp_total_nat  # ~0.2806
        self.ratio_gov_gdp        = _ts_nat_2021['Government Activities (4)'] / _gdp_total_nat  # ~0.1762
        self.ratio_net_taxes_gdp  = _ts_nat_2021['Net Taxes on Products (4)'] / _gdp_total_nat  # ~0.0604

        # Government sector mask: NACE O/P/Q sectors (public admin, education, health/social)
        _gov_sector_names = [
            'Public administration and defence; compulsory social security',
            'Education',
            'Human health activities',
            'Residential care activities',
            'Social work activities without accommodation',
        ]
        self.gov_sectors_ = np.zeros(self.S)
        for _name in _gov_sector_names:
            if _name in self.sectors:
                self.gov_sectors_[self.sectors.index(_name)] = 1

        # Non-oil-private mask: everything that is neither oil nor government
        # NOTE: non_oil_sectors_ (= 1 − sectors_oil_) is kept unchanged for EX decomposition
        self.non_oil_private_sectors_ = 1 - self.sectors_oil_ - self.gov_sectors_

        # Base-year scaling factors so that the sums of the proxy sector variables
        # equal the GASTAT ratios in 2021. Future dynamics track the proxy sectors.
        _Y_sum = np.sum(self.Y_)
        _proxy_non_oil = np.sum(self.Y_ * self.non_oil_private_sectors_)
        _proxy_oil     = np.sum(self.Y_ * self.sectors_oil_)
        _proxy_gov     = np.sum(self.Y_ * self.gov_sectors_)
        self.Y_non_oil_scale = (self.ratio_non_oil_gdp * _Y_sum / _proxy_non_oil) if _proxy_non_oil > 0 else 1.0
        self.Y_oil_scale     = (self.ratio_oil_gdp     * _Y_sum / _proxy_oil)     if _proxy_oil     > 0 else 1.0
        self.Y_gov_scale     = (self.ratio_gov_gdp     * _Y_sum / _proxy_gov)     if _proxy_gov     > 0 else 1.0
        # Net taxes: no sectoral proxy — grows in fixed proportion to total GDP
        self.Y_net_taxes_base = self.ratio_net_taxes_gdp * _Y_sum

        # Initialise calibration-period values with the scaled sectoral proxies
        self.Y_non_oil_ = self.Y_ * self.non_oil_private_sectors_ * self.Y_non_oil_scale
        self.Y_oil_     = self.Y_ * self.sectors_oil_              * self.Y_oil_scale
        self.Y_gov_     = self.Y_ * self.gov_sectors_              * self.Y_gov_scale
        self.Y_net_taxes = self.Y_net_taxes_base

        ########################################################################
        # 3.5 Public sector share (oil, desalination, electricity)
        ########################################################################
        self.electricity_gas_AC_sector = [self.sectors.index('Electricity, gas, steam and air conditioning supply')]

        self.s_public_ = np.zeros(self.S)
        oil_revenues = self.ts_government.loc[2021, 'Oil Revenues (5)']
        oil_share_public = self.oil_share_public = oil_revenues / np.sum(self.P_[oil_sectors])
        for i in oil_sectors:
            self.s_public_[i] = oil_share_public
        self.s_public_[self.sector_desal] = 1
        self.s_public_[self.electricity_gas_AC_sector] = parameters.energy_sector_public_share
        self.s_private_ = 1 - self.s_public_
        # Private investment without investment in desalination
        self.I_private_ = self.dK * (self.I_total - self.I_desal) * self.s_private_
        # Add desalination investment later into public investment
        self.I_public_ = self.dK*(self.I_total - self.I_desal) * self.s_public_
        # Define calibrated desalination investment as part of publich investment
        self.I_public_[self.sector_desal] = self.I_desal
        self.I_ = self.I_private_ + self.I_public_

        ########################################################################
        # 3.6 Financial sector calibration
        ########################################################################

        # Financial sector profits PB
        finance_sector = self.sectors.index(
            'Financial service activities, except insurance and pension funding')
        # Government ownership share of the banking sector (asset-weighted 2021):
        # ~44% of SNB (PIF ~37% + GOSI ~7%), 51% of Riyad Bank, 30% of Alinma (PIF+GOSI+PPA)
        # weighted by total assets → ~22% system-wide.
        self.gov_bank_share = parameters.gov_bank_share
        self.s_public_[finance_sector] = self.gov_bank_share
        self.s_private_[finance_sector] = 1 - self.gov_bank_share
        self.s_bank_ = np.zeros(self.S)
        self.s_bank_[finance_sector] = 1
        self.non_bank_sectors_ = 1 - self.s_bank_
        self.PB = self.P_[finance_sector]
        # Profits are corrected to become non-financial profits
        self.P_ = self.P_ * self.non_bank_sectors_

        # Initialize household wealth with the the total financial liabilities of the financial sector
        # minus government deposits
        # rm = 0.14681853300077513 %
        self.V = (
            self.ts_monetary.at[2021, 'Total Liabilities (9)'] -
            self.ts_monetary.at[2021, 'Government Deposits ** (9)']
        )

        # Initialize total loans from bank credit data
        self.L = self.ts_bank_credits.at['2021-12-31', 'Total']
        self.L_next = self.ts_bank_credits.at['2022-12-31', 'Total']

        # Calibration rl = rm + mu, and markup mu is like 0.0293 (ca. 3 pp), like in Poledna, Miess et al. (2023) ABM paper, p. 11
        self.mu = 0.0293
        # rm is the variable that is adjusted to equilibrate wealth and bank data, yields: rm = 0.14681853300077513 %
        # Alternatively, loans or the stock of wealth could be calibrated to match the interest rate.
        self.rm = (self.PB - self.mu*self.L) / (self.L + self.V)
        self.rl = self.rm + self.mu

        # Calibrate invest_profit to match bank credit data (two data points 2021 and 2022), household wealth, and investment IOT
        self.invest_profit = 1 - (
            ((self.L_next - self.L) * (1 - parameters.ρ)) /
            np.sum(self.I_private_)
        )

        ########################################################################
        # 3.7 Scarcity parameters (AGR sector)
        ########################################################################
        # AGR sector index to implement water scarcity factor in intermediate inputs
        self.agr_sector = self.sectors.index(
            'Crop and animal production, hunting and related service activities')
        self.s_agr_ = np.zeros(self.S)
        self.s_agr_[self.agr_sector] = 1
        self.non_agr_ = 1 - self.s_agr_

        ###############################################################################################################################
        ###############################################################################################################################
        ###############################################################################################################################
        # IV. Macroeconomic time series — growth rates and averages
        ###############################################################################################################################
        ###############################################################################################################################
        ###############################################################################################################################

        ###########################################################################################################################
        # 4.1 GDP growth rates
        ###########################################################################################################################

        # For GDP, main validation growth rate
        self.Y_timeseries = self.ts_national['Gross Domestic Product (4)']
        value_2013 = self.Y_timeseries.loc[startyear]
        value_2019 = self.Y_timeseries.loc[endyear-4]
        value_2020 = self.Y_timeseries.loc[endyear-3]
        value_2021 = self.Y_timeseries.loc[endyear-2]
        value_2022 = self.Y_timeseries.loc[endyear-1]
        value_2023 = self.Y_timeseries.loc[endyear]

        self.gY_2020 = (value_2020 / value_2019) - 1
        self.gY_2021 = (value_2021 / value_2020) - 1
        self.gY_2022 = (value_2022 / value_2021) - 1
        self.gY_2023 = (value_2023 / value_2022) - 1
        self.gY_avg = (value_2023 / value_2013) ** (1 / years) - 1

        ########################################################################
        # 4.2 Inflation measures (Non-oil GDP deflator)
        ########################################################################
        self.GDP_deflator_ts = self.ts_national['GDP (5)']
        self.Oil_sector_deflator_ts = self.ts_national['Oil Sector (5)']
        self.Non_oil_sector_deflator_ts = self.ts_national['Non-oil Sector (5)']

        # Rebase indices from 2018=100 to 2021=1
        self.GDP_deflator = self.ts_national['GDP (5)'] / \
            self.ts_national.loc[2021, 'GDP (5)']
        self.Oil_sector_deflator = self.ts_national['Oil Sector (5)'] / \
            self.ts_national.loc[2021, 'Oil Sector (5)']
        self.Non_oil_sector_deflator = self.ts_national['Non-oil Sector (5)'] / \
            self.ts_national.loc[2021, 'Non-oil Sector (5)']

        value_2013 = self.Non_oil_sector_deflator[startyear]
        value_2019 = self.Non_oil_sector_deflator[endyear-4]
        value_2020 = self.Non_oil_sector_deflator[endyear-3]
        value_2021 = self.Non_oil_sector_deflator[endyear-2]
        value_2022 = self.Non_oil_sector_deflator[endyear-1]
        value_2023 = self.Non_oil_sector_deflator[endyear]

        self.inflation_2020 = (value_2020 / value_2019) - 1
        self.inflation_2021 = (value_2021 / value_2020) - 1
        self.inflation_2022 = (value_2022 / value_2021) - 1
        self.inflation_2023 = (value_2023 / value_2022) - 1
        self.inflation_avg = (value_2023 / value_2013) ** (1 / years) - 1

        ########################################################################
        # 4.3 Investment
        ########################################################################
        # Gross fixed capital formation (excludes large inventory changes in Saudi data)
        self.I_timeseries = self.ts_national['Gross Fixed Capital Formation (4)'].ffill(
        )
        value_2013 = self.I_timeseries.loc[startyear]
        value_2023 = self.I_timeseries.loc[endyear]
        self.gI_avg = (value_2023 / value_2013) ** (1 / years) - 1

        ########################################################################
        # 4.4 Exports, oil and non-oil
        ########################################################################
        self.EXP_timeseries = self.ts_national['Exports of Goods & Services (4)'].ffill(
        )
        value_2013 = self.EXP_timeseries.loc[startyear]
        value_2023 = self.EXP_timeseries.loc[endyear]
        self.gEXP_avg = (value_2023 / value_2013) ** (1 / years) - 1

        # Approximate growth rate OIL EXPORTS by oil activities from 2013-2023
        self.oil_activities = self.ts_national['Oil Activities (4)'].ffill()
        value_2013 = self.oil_activities.loc[startyear]
        value_2023 = self.oil_activities.loc[endyear]
        self.gEXP_oil_avg = (value_2023 / value_2013) ** (1 / years) - 1

        # Non-oil export growth rate approximated from IOT data
        value_2018 = 312494781
        value_2021 = 316257172
        self.gEXP_non_oil_avg = (
            value_2021 / value_2018) ** (1 / (2021-2018)) - 1

        # Yearly growth rates, for later reference
        self.gOil_activities = self.oil_activities.pct_change()

        ########################################################################
        # 4.5 Imports
        ########################################################################
        self.IM_timeseries = self.ts_national['Imports of Goods & Services (4)'].ffill(
        )
        value_2013 = self.IM_timeseries.loc[startyear]
        value_2023 = self.IM_timeseries.loc[endyear]
        self.gIM_avg = (value_2023 / value_2013) ** (1 / years) - 1

        # Yearly growth rates, for later reference
        self.gIM_ = self.IM_timeseries.pct_change()

        ########################################################################
        # 4.6 Government expenditures
        ########################################################################
        self.GovExp_timeseries = self.ts_government['Total Expenditures (5)'].ffill(
        )
        self.gov_expenditure = self.ts_government.loc[2021,
                                                      'Total Expenditures (5)']
        value_2013 = self.GovExp_timeseries.loc[startyear]
        value_2023 = self.GovExp_timeseries.loc[endyear]
        self.gGY_avg = (value_2023 / value_2013) ** (1 / years) - 1

        # Government bonds interest rate
        self.interest_gov_bonds = 0.04  # Assume 4% interest rate on government bonds
        # Assume the same refinancing rate as for corporate loans, equal to the yield
        self.gov_bonds_refinancing_rate = parameters.ρ
        # Assumption of 5 % interest rate on government bonds:
        # See e.g. SAMA bills 2022-2025 https://www.sama.gov.sa/en-US/GovtSecurity/Pages/SAMABills.aspx?&&p_SAMABillYear=8_2024&p_SAMABillPeriod=20241001%2021%3a00%3a00&&PageFirstRow=1&View=c153645f-44b6-423e-bcb9-f51f723f04c9
        # Or government bonds information here, retrieved Jan. 2026: https://www.bondsupermart.com/bsm/bond-factsheet/XS2159975882



        ########################################################################
        # 4.7 Wages
        ########################################################################
        # Wage growth calibrated from IOT data (2018–2021)
        value_2018 = 811481884
        value_2021 = 931608681
        self.gW = (value_2021 / value_2018) ** (1 / (2021-2018)) - 1

        ###############################################################################################################################
        ###############################################################################################################################
        ###############################################################################################################################
        # V. Time series estimation (AR models) and energy calibration
        ###############################################################################################################################
        ###############################################################################################################################
        ###############################################################################################################################

        ###########################################################################################################################
        # 5.1 GDP Y — optimal lag order
        ###########################################################################################################################
        # Shorten the time series so it starts in 2022; the last growth rate then refers to 2021, as intended.
        self.Y_timeseries = self.Y_timeseries[:-2]
        # The end-year values (endyear and endyear-1) are used as exogenous inputs to the model, for Y and the variables below.

        self.begin_time_series_estimation_Y = parameters.Begin_timeseries_estimation_Y

        max_lags = min(
            len(self.Y_timeseries[self.begin_time_series_estimation_Y:-1]) // 5, 10)
        if parameters.Y_timeseries_model_log:
            self.Y_optimal_lag_aic, self.Y_optimal_lag_bic = determine_optimal_lag_order_log(
                self.Y_timeseries, max_lags, self.begin_time_series_estimation_Y)
        else:
            self.Y_optimal_lag_aic, self.Y_optimal_lag_bic = determine_optimal_lag_order_growthrates(
                self.Y_timeseries, max_lags, self.begin_time_series_estimation_Y)
        # The AR model is estimated on growth rates & the time series is restricted by begin_time_series_estimation

        ###########################################################################################################################
        # 5.2 Inflation rates — time series estimation
        ###########################################################################################################################

        self.Deflator_non_oil_GDP_timeseries = self.Non_oil_sector_deflator[:-2]
        # The end-year values (endyear and endyear-1) are used as exogenous inputs to the model, for Y and the variables below.

        self.begin_time_series_estimation_deflator = parameters.Begin_timeseries_estimation_deflator

        max_lags = min(len(
            self.Deflator_non_oil_GDP_timeseries[self.begin_time_series_estimation_deflator:-1]) // 5, 10)
        self.Deflator_optimal_lag_aic, self.Deflator_optimal_lag_bic = determine_optimal_lag_order_growthrates(
            self.Deflator_non_oil_GDP_timeseries, max_lags, self.begin_time_series_estimation_deflator)
        # The AR model is estimated on growth rates & the time series is restricted by begin_time_series_estimation

        ########################################################################
        # 5.3 Energy intensities
        ########################################################################
        try:
            # Step 1: Load the CSV, assuming a standard comma delimiter.
            # Use the 'sector' column as the index for robust alignment.
            df_energy = pd.read_csv(
                "input_data/energy_intensities_for_model_only_2.csv",
                delimiter=';',
                index_col='sector'
            )

            # Step 2: Create a template DataFrame with the model's official sector list.
            # This ensures the data aligns with the model's official sector structure.
            sectors_df = pd.DataFrame(index=self.sectors)

            # Step 3: Reindex the loaded data to match the model's sector order.
            # This will sort the data correctly and fill any missing sectors with 0.
            df_energy_aligned = df_energy.reindex(sectors_df.index).fillna(0)

            # Step 4: Convert the final, aligned columns into NumPy arrays.
            self.gas_intensities_ = df_energy_aligned['gas_intensities_(GJ/Thousand_SAR)'].to_numpy(
            )
            self.electricity_intensities_ = df_energy_aligned[
                'electricity_intensities_(GJ/Thousand_SAR)'].to_numpy()
            self.oil_intensities_ = df_energy_aligned['oil_intensities_(GJ/Thousand_SAR)'].to_numpy(
            )

        except Exception as e:
            print(f"CRITICAL ERROR loading energy_intensities.csv: {e}")
            print("Please ensure the CSV file is clean, uses commas as separators, has the correct headers (sector, gas_intensities, etc.), and is in the 'input_data' folder.")
            raise

        ########################################################################
        # 5.4 Emission intensities
        ########################################################################
        # Legacy: these CSV-derived *_emission_intensities_ are not used downstream, and the raw
        # CSV header's oil factor (66.3 MtCO2/EJ) is outdated. Emissions use the IPCC factors in 5.5.
        try:
            # Load emission intensity data, assuming standard comma delimiter.
            # The sector names in the first column will be used to align the data.
            df_emission = pd.read_csv(
                "input_data/emission_Factors_and_intensities_for_model_use.csv",
                delimiter=';',
                index_col='sector'
            )
            # Create a template DataFrame with the model's official sector list to ensure alignment
            sectors_df = pd.DataFrame(index=self.sectors)

            # Reindex the loaded data to match the model's sector order, filling missing with 0
            df_emission_aligned = df_emission.reindex(
                sectors_df.index).fillna(0)

            # Convert intensity columns to numpy arrays
            self.gas_emission_intensities_ = df_emission_aligned[
                'gas_Emission_intensities (MtCO2/Thousand SAR)'].to_numpy()
            self.electricity_emission_intensities_ = df_emission_aligned[
                'electricity_Emission_intensities (MtCO2/Thousand SAR)'].to_numpy()
            self.oil_emission_intensities_ = df_emission_aligned[
                'oil_Emission_intensities (MtCO2/Thousand SAR)'].to_numpy()

        except Exception as e:
            print(f"CRITICAL ERROR loading emission_intensities.csv: {e}")
            print("Please ensure the CSV file is clean, has the correct headers (sector, gas_emission_intensities, etc.), and is in the 'input_data' folder.")
            raise

        ########################################################################
        # 5.5 Renewable energy & emission factor calibration
        ########################################################################

        # 1. Calibrate initial energy use for 2021 (base year) from existing intensities
        # This is the total initial energy demand in GJ.
        # initial_gas_use = np.sum(self.X_ * self.gas_intensities_)
        # initial_oil_use = np.sum(self.X_ * self.oil_intensities_)
        # initial_electricity_use = np.sum(self.X_ * self.electricity_intensities_)
        # self.initial_total_energy_demand = initial_gas_use + initial_oil_use + initial_electricity_use

        # 2. Set initial renewable energy generation for 2021.
        # Based on 2021 data, renewable capacity was low. We assume it contributed ~1% of total energy.
        # self.initial_RE_share = 0.01  # 1% initial share
        # self.initial_RE_generation = self.initial_total_energy_demand * self.initial_RE_share # in GJ

        # 3. Define growth rates for renewable energy scenarios
        # Scenario 1 (Trend): Based on strong recent growth from a low base.
        # self.gRE_avg_trend = 0.25  # 25% annual growth for the trend scenario

        # Scenario 2 (Vision 2030): The growth path will be calculated dynamically in the model
        # to reach 50% of total energy use by t=10 (year 2030).

        # 4. Define Emission Factors (replacing the old intensity-based method)
        # Using standard factors. Units: MtCO2 / GJ
        # Source: IPCC 2006 Guidelines for National Greenhouse Gas Inventories, Vol. 2, Table 1.4
        # (unchanged in the 2019 Refinement, Vol. 2, Ch. 1, Sec. 1.4: "No refinement").
        # 95% CI: crude oil 71,100-75,500 kg/TJ; natural gas 54,300-58,300 kg/TJ.
        kg_per_tonne = 1000
        TJ_per_GJ = 1/1000
        Mt_per_kt = 1/1000

        # --- EMISSION FACTORS (Corrected to MtCO2/GJ) ---
        # 1. Base Factor in Tonnes/GJ
        # 56,100 kg/TJ = 56.1 Tonnes/TJ = 0.0561 Tonnes/GJ
        factor_gas_tonnes_gj = 0.0561

        # 73,300 kg/TJ = 73.3 Tonnes/TJ = 0.0733 Tonnes/GJ
        factor_oil_tonnes_gj = 0.0733

        # 2. Convert Tonnes to Million Tonnes (Mt)
        # 1 Mt = 1,000,000 Tonnes
        self.gas_emission_factor = factor_gas_tonnes_gj / 1_000_000  # MtCO2/GJ
        self.oil_emission_factor = factor_oil_tonnes_gj / 1_000_000  # MtCO2/GJ
        # We assume renewable energy has zero operational emissions.
        self.re_emission_factor = 0.0

        # The emissions from electricity will now be calculated dynamically based on the
        # decarbonization of its input fuels (handled in model.py)
        # We need the index of the electricity sector for this.
        self.electricity_sector_idx = self.sectors.index(
            'Electricity, gas, steam and air conditioning supply')

        ########################################################################
        # 5.6 Sectors for targeted oil reduction
        ########################################################################
        # User specified: "Power Plants & Desalination", "Industry and Agriculture"

        sectors_to_reduce = [
            # 1. Power Plants & Desalination
            'Electricity, gas, steam and air conditioning supply',
            'Desalination',

            # 2. Agriculture
            'Crop and animal production, hunting and related service activities'
        ]

        # 3. Industry (All Manufacturing, Mining, and Construction)
        for sector_name in self.sectors:
            if (
                'Manufacture of' in sector_name or
                'Mining' in sector_name or
                'Quarrying' in sector_name or
                'Construction' in sector_name or
                'Civil engineering' in sector_name
            ):
                sectors_to_reduce.append(sector_name)

        # Add other related industrial sectors if they exist
        other_industrial = [
            'Extraction of crude petroleum and natural gas',
            'Sewerage',
            'Waste collection, treatment and disposal activities; materials recovery',
            'Repair and installation of machinery and equipment'
        ]
        for sector_name in other_industrial:
            if sector_name in self.sectors:
                sectors_to_reduce.append(sector_name)

        # Get unique sector indices
        indices = [self.sectors.index(s) for s in set(
            sectors_to_reduce) if s in self.sectors]
        self.oil_reduction_sectors_idx = np.array(indices)


        ########################################################################
        # 5.7 Energy efficiency tiers calibration
        ########################################################################
        # Calculate total initial energy intensity for ranking
        total_energy_intensity = self.gas_intensities_ + \
            self.electricity_intensities_ + self.oil_intensities_

        # Get the indices that would sort the array (from lowest to highest)
        sorted_indices = np.argsort(total_energy_intensity)

        # Get the indices for each tier
        # Top 20 most intensive sectors (last 20 in the sorted list)
        self.top_20_energy_sectors = sorted_indices[-20:]

        # Middle 20 intensive sectors (the 20 before the top 20)
        self.mid_20_energy_sectors = sorted_indices[-40:-20]

        # All other sectors (the rest)
        self.low_energy_sectors = sorted_indices[:-40]

        self.energy_reduction_factor_ = np.zeros(self.S)

        ########################################################################
        # 5.8 Energy efficiency cost calibration (payback period logic)
        ########################################################################

        # 1. Define indices
        oil_idx_crude = self.sectors.index(
            'Extraction of crude petroleum and natural gas')
        oil_idx_refined = self.sectors.index(
            'Manufacture of coke and refined petroleum products')
        elec_idx = self.electricity_sector_idx

        # 2. Calculate Total Energy Bill (Thousand SAR)
        sectoral_oil_bill_ = (
            self.A[oil_idx_crude, :] + self.A[oil_idx_refined, :]) * self.X_
        sectoral_gas_bill_ = self.A[oil_idx_crude, :] * self.X_
        sectoral_elec_bill_ = self.A[elec_idx, :] * self.X_
        total_energy_bill_ = sectoral_oil_bill_ + \
            sectoral_gas_bill_ + sectoral_elec_bill_

        # 3. Calculate Total Energy Consumption (GJ)
        total_energy_gj_ = (self.oil_intensities_ + self.gas_intensities_ +
                            self.electricity_intensities_) * self.X_

        # 4. Calculate Average Energy Price (Thousand SAR per GJ)
        #    Avoid division by zero
        self.avg_energy_price_ = np.divide(
            total_energy_bill_,
            total_energy_gj_,
            out=np.zeros_like(total_energy_bill_),
            where=total_energy_gj_ > 0.001
        )

        # 5. Calculate Base CapEx based on Payback Period
        #    Logic: If energy costs 50 SAR/GJ, and payback is 3 years,
        #    We are willing to invest 150 SAR to save 1 GJ/year.
        #    Base CapEx = Price * Payback Years

        # The payback period is configured via the 'efficiency_payback_period' parameter.
        target_payback = parameters.efficiency_payback_period

        self.base_efficiency_capex_ = self.avg_energy_price_ * target_payback

        # 6. Define Investment Shares (Who builds the tech)
        effic_shares = parameters.efficiency_investment_shares
        self.dI_Effic_ = np.array(
            [effic_shares.get(sector, 0) for sector in self.sectors]
        )
        if self.dI_Effic_.sum() > 0:
            self.dI_Effic_ = self.dI_Effic_ / self.dI_Effic_.sum()

        ###########################################################################################################################
        # 5.9 Government expenditures — AR model
        ###########################################################################################################################
        # We use growth rates, since the total level of Gov Exp surpasses GY, but we use the growth rates
        # from GovExp to proxy growth rates in GY, since we do not have a good time series for GY
        self.GY_timeseries = self.GovExp_timeseries

        # Begin time series estimation around 2005, to exclude some major structural breaks in Saudi economy
        # Use X years before the end of the time series
        self.begin_time_series_estimation_gov = parameters.Begin_timeseries_estimation_GY

        max_lags = min(
            len(self.GY_timeseries[self.begin_time_series_estimation_gov:-1]) // 5, 10)
        self.GY_optimal_lag_aic, self.GY_optimal_lag_bic = determine_optimal_lag_order_growthrates(
            self.GY_timeseries, max_lags, self.begin_time_series_estimation_gov)
        # The AR model is estimated on growth rates & the time series is restricted by begin_time_series_estimation

        ###########################################################################################################################
        # 5.10 Oil exports — AR model
        ###########################################################################################################################
        # Approximate OIL EXPORTS by oil activities

        # Begin time series estimation around 2005, to exclude some major structural breaks in Saudi economy
        # Use X years before the end of the time series
        self.begin_time_series_estimation_oil = parameters.Begin_timeseries_estimation_EX_oil

        max_lags = min(
            len(self.oil_activities[self.begin_time_series_estimation_oil:-1]) // 5, 10)
        self.Oil_optimal_lag_aic, self.Oil_optimal_lag_bic = determine_optimal_lag_order_growthrates(
            self.oil_activities, max_lags, self.begin_time_series_estimation_oil)

        ###########################################################################################################################
        # 5.11 Non-oil exports — AR model
        ###########################################################################################################################
        # Take from ts_exports data
        self.non_oil_export_timeseries = self.ts_exports['Grand Total'] - \
            self.ts_exports['Mineral fuels lubricants and related materials']

        # Begin time series estimation around 2005, to exclude some major structural breaks in Saudi economy
        # Use X years before the end of the time series
        self.begin_time_series_estimation_nonoil_exp = parameters.Begin_timeseries_estimation_EX_nonoil

        max_lags = min(len(
            self.non_oil_export_timeseries[self.begin_time_series_estimation_nonoil_exp:-1]) // 5, 10)
        self.non_oil_optimal_lag_aic, self.non_oil_optimal_lag_bic = determine_optimal_lag_order_growthrates(
            self.non_oil_export_timeseries, max_lags, self.begin_time_series_estimation_nonoil_exp)

        ###########################################################################################################################
        # 5.12 Imports — AR model
        ###########################################################################################################################
        self.IM_timeseries = self.IM_timeseries[:-2]
        # Begin time series estimation around 2005, to exclude some major structural breaks in Saudi economy
        # Use X years before the end of the time series
        self.begin_time_series_estimation_im = -20

        max_lags = min(
            len(self.IM_timeseries[self.begin_time_series_estimation_im:-1]) // 5, 10)
        self.IM_optimal_lag_aic, self.IM_optimal_lag_bic = determine_optimal_lag_order_growthrates(
            self.IM_timeseries, max_lags, self.begin_time_series_estimation_im)
        # The AR model is estimated on growth rates & the time series is restricted by begin_time_series_estimation

        ###########################################################################################################################
        # 5.13 Sector lists
        ###########################################################################################################################

        # List of sector names for input of GDP data
        gdp_data_sector_names = [
            "Agriculture, Forestry & Fishing (4)",
            "Mining & Quarrying (4)",
            "a) Crude Petroleum & Natural Gas (4)",
            "b) Other Mining & Quarrying (4)",
            "Manufacturing (4)",
            "a) Petroleum Refining (4)",
            "b) Manufacturing excluding petroleum refining  (4)",
            "Electricity, Gas and Water (4)",
            "Construction (4)",
            "Wholesale & Retail Trade, Restaurants & hotels (4)",
            "Transport, Storage & Communication (4)",
            "Finance, Insurance, Real Estate & Business Services (4)",
            "a) Real Estate (4)",
            "b) Finance , Insurance and Business sevices (4)",
            "Community, Social & Personal Services (4)",
            "Government Services (4)",
            "Gross Value Added  (4)",
            "Net Taxes on Products (4)",
            "Gross Domestic Product (4)"
        ]

        # Function to sanitize and simplify the sector names
        def sanitize_sector_name(sector_name):
            return sector_name.strip().replace(' ', '_').replace('&', 'and').replace(',', '').replace('(', '').replace(')', '').replace('-', '_').replace('a)', 'a').replace('b)', 'b')

        # Dictionary to hold the time series for each sector
        self.gdp_nominal = {}

        # Iterate over the sector names and create time series for each sector
        for sector_name in gdp_data_sector_names:
            sanitized_name = sanitize_sector_name(sector_name)
            self.gdp_nominal[sanitized_name] = self.ts_national[sector_name]

        ###############################################################################################################################
        ###############################################################################################################################
        ###############################################################################################################################
        # VI. Government, household, and macro accounting
        ###############################################################################################################################
        ###############################################################################################################################
        ###############################################################################################################################

        ########################################################################
        # 6.1 Government revenue and fiscal ratios
        ########################################################################

        # Calculate some important government and oil sector revenue to activity ratios
        self.gov_oil_revenues_timeseries = self.ts_government['Oil Revenues (5)']
        self.gov_oil_revenue_ratio = self.gov_oil_revenues_timeseries / self.oil_activities
        self.gov_non_oil_revenue_ratio = self.ts_government[
            'Non-Oil Revenues (5)'] / self.ts_national['Non-Oil Sector (4)']
        self.gov_non_oil_revenue_ratio_avg = self.gov_non_oil_revenue_ratio.tail(
            10).mean()
        self.gov_oil_revenues_to_oil_activity_ratio = self.gov_oil_revenues_timeseries / \
            self.oil_activities

        # Calculate VAT rate, gross and net consumption
        self.vat = self.macro_vars_2.at['Net tax on products',
                                        'Households final consumption expenditures']
        self.C_HH_gross = self.macro_vars_2.at["Primary inputs at purchase prices",
                                               "Households final consumption expenditures"]
        self.C_NPISH_gross = self.macro_vars_2.at["Primary inputs at purchase prices",
                                                  "Non profit institutions serving household final consumption expenditures"]
        self.C_gross = self.C_HH_gross + self.C_NPISH_gross

        self.C_net = self.C_gross - self.vat
        self.tau_vat = self.vat / self.C_net

        # Non-VAT gov tax TH is resudual beteween gov non-oil revenues and other taxes (Net taxes on products incl. VAT, taxes on production + investment, etc.)
        self.TH = (
            self.ts_government.at[2021, 'Non-Oil Revenues (5)']
            - self.vat
            - np.sum(self.tax_products_net_)
            - np.sum(self.tax_production_)
            - self.tax_investments
        )

        # Government oil revenues ratio in relation to crude oil exports
        self.gov_oil_revenues_2021 = self.gov_oil_revenues_timeseries.loc[2021]
        self.gov_oil_revenue_oil_exports_ratio = self.gov_oil_revenues_2021 / \
            np.sum(self.EX_oil_)

        # Government deficit/surplus (net lending net borrowing - nlnb)
        self.gov_nlnb = self.ts_government['Deficit_Surplus (Actual) (5)']

        # Calculate average growth rate of oil sector, to test the model
        self.gOil = (self.gdp_nominal['a_Crude_Petroleum_and_Natural_Gas_4'] +
                     self.gdp_nominal['a_Petroleum_Refining_4']).pct_change()
        self.gOil_avg = self.gOil.tail(10).mean()

        ########################################################################
        # 6.2 Household calibration
        ########################################################################

        # we assume that all wage income is consumed, see IOTs
        self.C_wage_inc_net = np.sum(self.W_)
        self.vat_wage_cons = parameters.α1 * self.tau_vat * self.C_wage_inc_net
        self.YD_wage = np.sum(self.W_)  # - self.vat_wage_cons

        # Calculate disposable income of households and net consumption out of each income type (profits vs. wages)
        self.C_profit_inc_gross = self.C_gross - self.YD_wage
        self.C_profit_inc_net = self.C_profit_inc_gross / (1 + self.tau_vat)
        self.vat_profit = self.tau_vat * self.C_profit_inc_net
        self.vat_profit_cons = self.tau_vat * self.C_profit_inc_net

        # Tax rate of non gov VAT
        self.τ = self.TH / ((np.sum(self.s_private_ * self.P_)) + self.PB)

        # Include all costs in profit income, including the initial payment for investment, because in the model this is also deducted from profits
        self.YD_profit = np.sum(self.s_private_ * self.P_) + self.PB - self.TH

        # Calibrate MPC on GROSS consumption out of PROFIT income
        # α2 is initialized with non-wage consumption (i.e. total consumption - wages, as the data show MPC out of wages = 1)
        # α2 is calibrated high enough that deducting remittances from wage income in the model does not reduce consumption excessively
        self.α2 = self.C_profit_inc_gross / self.YD_profit

        ########################################################################
        # 6.3 GDP calculation
        ########################################################################
        self.GDP = self.Y_.sum() + self.C_residents + self.C_non_residents + self.vat

        ########################################################################
        # 6.4 Initial values for dynamic variables
        ########################################################################
        self.GY = self.G_.sum()    # Total government consumption from IOTs
        self.gov_exp_resid = self.gov_expenditure - self.GY - \
            np.sum(self.I_public_) - \
            np.sum(self.sub_production_)  # residual to total expenditures
        # Net International Investment Position (NIIP):
        # The NIIP reflects the total net investment position of Saudi Arabia (incl. private
        # assets), not only government assets. It is used here as a proxy for government net
        # wealth (ARAMCO is not included in the data).
        # PIF assets as of January 2025: 925 billion USD; conversion rate 1 USD = 3.75 SAR.
        self.PIF = 9.25e8 * 3.75  # PIF assets in SAR (reference value)
        # GV is initialized from the 2021 net international investment position.
        self.GV = self.ts_investment.loc[2021]["Net International Investment Position"]
        # ── Government bond stock split (SAMA Annual Debt Statistics, 2021) ──────────────────────
        # Initial domestic/external bond stocks are calibrated directly from ts_government.csv
        # (column section "Outstanding Public Debt at year end").
        # 2021 outstanding: Internal (domestic) ≈ SAR 559 bn; External ≈ SAR 379 bn; Total ≈ SAR 938 bn.
        # Source: SAMA Annual Statistics 2022, Table 4 — Public Debt by Resident/Non-resident.
        _bond_domestic_2021 = self.ts_government.loc[2021, 'Internal (1).2']   # SAR ~559 bn domestic
        _bond_external_2021 = self.ts_government.loc[2021, 'External (1).2']   # SAR ~379 bn external
        _bond_total_2021    = self.ts_government.loc[2021, 'Total Public Debt (1)']  # SAR ~938 bn
        # Consistency check: domestic + external must equal total (tolerance 1 SAR million = 1,000 tSAR)
        assert abs(_bond_domestic_2021 + _bond_external_2021 - _bond_total_2021) < 1000, (
            f"Bond split inconsistency: {_bond_domestic_2021:.0f} + {_bond_external_2021:.0f} "
            f"!= {_bond_total_2021:.0f}"
        )
        self.Bond_domestic = _bond_domestic_2021      # Initial stock of domestic sovereign bonds
        self.Bond_external_init = _bond_external_2021  # Initial stock of external sovereign bonds
        # New-issuance split: 60 % domestic / 40 % external (rounded from empirical 59.6 % / 40.4 %).
        # Applied every period to the raw fiscal deficit; see model.py bond issuance block.
        self.domestic_bond_share = 0.60
        # 6.72 trillion SAR Market Capitalization Jan 29 2025, source: https://g.co/kgs/f3pL4HL
        self.aramco_value = 6.72e9
        self.Gov_net_wealth = self.GV - self.Bond_domestic - self.Bond_external_init
        self.K_water = self.K_[self.sector_water] - self.K_desal
        self.δ = self.DEPR_.sum() / self.K_.sum()
        self.δ_ = self.DEPR_ / self.K_
        self.δ_[self.sector_desal] = self.δ_desal
        self.δ_[self.sector_wwater] = self.δ_wwater
        self.C = self.C_gross - self.vat

        # Calculate initial prices
        # In the first period nominal equals real, since prices are calibrated to 1.
        self.x_ = self.X_
        self.k_ = self.K_
        self.IntP_ = np.sum(self.A, axis=0) * self.x_

        # Initial stock of loans - initialize according to data, calculate shares from the data above
        self.L_ = self.dX * self.L

        # Initialize inflation with the long-term average 2013 - 2023
        self.inflation = self.inflation_avg
        self.u_ = self.X_ / (self.K_ * self.eK_)

        ########################################################################
        # 6.5 Productivity coefficient for target capital stock
        ########################################################################
        # Calibrate the sectoral productivity coefficient eta_K_ for the target capital stock,
        # following Naqvi and Stockhammer (2018). K is approximated as KX × X (see parameters.KX).
        # The Y/X ratio from 2021 is used as a proxy for scaling.
        self.YX_ratio = np.sum(self.Y_) / np.sum(self.X_)
        self.X_timeseries = self.Y_timeseries/self.YX_ratio
        self.X_timeseries_shifted = self.X_timeseries.shift(1)
        self.eta_K = parameters.gamma_i * self.Y_timeseries / \
            (self.I_timeseries - self.X_timeseries_shifted *
             parameters.KX * (self.δ - parameters.gamma_i))
        self.eta_K_avg = np.mean(self.eta_K)
        self.eta_K_X_avg = self.eta_K_avg / self.YX_ratio

        ###############################################################################################################################
        ###############################################################################################################################
        ###############################################################################################################################
        # VII. Vision 2030 targets and FDI
        ###############################################################################################################################
        ###############################################################################################################################
        ###############################################################################################################################

        ########################################################################
        # 7.1 Sectoral growth targets — Vision 2030
        ########################################################################
        self.sectoral_yearly_growth_rates_V2030_ = np.zeros(self.S)
        self.sectoral_total_targets_V2030_ = np.zeros(self.S)
        self.sectoral_yearly_target_growth_rates_V2030_ = np.zeros(self.S)
        # Vision 2030 timing can also be longer than 9 years (from 2021 to 2030), depending on whether reaching the target until 2030 is realistic
        years_to_2030_timing = parameters.vision_2030_timing

        # Building & Construction sectors
        self.sectors_construction = [self.sectors.index('Construction of buildings'), self.sectors.index(
            'Civil engineering'), self.sectors.index('Specialized construction activities')]
        # Manually set growth rate until end of startyear + vision_2030_timing (which is set to 9 at the moment)
        self.sectoral_total_targets_V2030_[
            self.sectors_construction] = parameters.construction_sector_V2030_target
        self.sectoral_yearly_target_growth_rates_V2030_[self.sectors_construction] = (
            1 + self.sectoral_total_targets_V2030_[self.sectors_construction]) ** (1 / years_to_2030_timing) - 1

        # Tourism sectors
        self.sectors_tourism = [
            self.sectors.index('Accommodation'),
            self.sectors.index('Food and beverage service activities'),
            self.sectors.index(
                'Travel agency, tour operator, reservation service and related activities'),
            self.sectors.index(
                'Libraries, archives, museums and other cultural activities'),
            self.sectors.index(
                'Sports activities and amusement and recreation activities')
        ]
        # Manually set growth rate until end of startyear + vision_2030_timing (which is set to 9 at the moment)
        self.sectoral_total_targets_V2030_[
            self.sectors_tourism] = parameters.tourism_sector_V2030_target
        self.sectoral_yearly_target_growth_rates_V2030_[self.sectors_tourism] = (
            1 + self.sectoral_total_targets_V2030_[self.sectors_tourism]) ** (1 / years_to_2030_timing) - 1

        # Plastic sector
        self.sectors_plastic = [
            self.sectors.index('Manufacture of rubber and plastics products')
        ]
        # Manually set growth rate until end of startyear + vision_2030_timing (which is set to 9 at the moment)
        self.sectoral_total_targets_V2030_[
            self.sectors_plastic] = parameters.plastic_sector_V2030_target
        self.sectoral_yearly_target_growth_rates_V2030_[self.sectors_plastic] = (
            1 + self.sectoral_total_targets_V2030_[self.sectors_plastic]) ** (1 / years_to_2030_timing) - 1

        # below) collapses to near-zero production in the Vision_2030 scenario due to the 30% forced
        # demand target (AI_hightech_sector_V2030_target) exceeding what the Leontief production
        # function can supply given low flex_intermediate_inputs/flex_capital. The plot function
        # (experiment_plots.py, plot_sectoral_gdp_comparison) contains a stability filter that
        # auto-replaces this sector in the sectoral GDP comparison figure. The root-cause fix should
        # address the demand-capacity mismatch — see also endogenize_input_output_matrix_A.py.
        self.sectors_AI_hightech = [
            self.sectors.index(
                'Manufacture of computer, electronic and optical products'),
            self.sectors.index('Manufacture of electrical equipment'),
            self.sectors.index(
                'Manufacture of machinery and equipment n.e.c.'),
            self.sectors.index('Programming and broadcasting activities'),
            self.sectors.index('Telecommunications'),
            self.sectors.index(
                'Computer programming, consultancy and related activities'),
            self.sectors.index('Information service activities'),
            self.sectors.index(
                'Architectural and engineering activities; technical testing and analysis'),
            self.sectors.index('Scientific research and development')
        ]
        # Manually set growth rate until end of startyear + vision_2030_timing (which is set to 9 at the moment)
        self.sectoral_total_targets_V2030_[
            self.sectors_AI_hightech] = parameters.AI_hightech_sector_V2030_target
        self.sectoral_yearly_target_growth_rates_V2030_[self.sectors_AI_hightech] = (
            1 + self.sectoral_total_targets_V2030_[self.sectors_AI_hightech]) ** (1 / years_to_2030_timing) - 1

        # After having defined the yearly target growth rates for Vision 2030 sectors dependent on total growth targets, put them in a vector of vision 2030 growth rates
        self.vision2030_sectors = self.sectors_construction + \
            self.sectors_tourism + self.sectors_AI_hightech + self.sectors_plastic
        # Sort from lowest to highest sector number
        self.vision2030_sectors = sorted(self.vision2030_sectors)
        for i in self.vision2030_sectors:
            self.sectoral_yearly_growth_rates_V2030_[i] = self.sectoral_yearly_target_growth_rates_V2030_[i]

        # Create a copy of the sectoral growth rates for non-oil export adjustment, so that we can complement with manual parameters later, if needed
        self.sectoral_yearly_growth_rates_non_oil_exports_ = np.copy(
            self.sectoral_yearly_growth_rates_V2030_)

        ###############################################################################################################################
        ###############################################################################################################################
        # 7.2 FDI data — net flows
        # Source: FDI Bulletin 2023, Sheets 2-1 (stocks), 2-2 (net flows), 2-3 (gross inflows)
        # 18 aggregate sectors disaggregated to 85 model sectors using dX weights
        ###############################################################################################################################
        ###############################################################################################################################
        self.ts_fdi_aggregate = pd.read_csv(
            "input_data/ts_fdi.csv", index_col=0)

        # Mapping from model sector index to FDI aggregate sector name (NACE-based)
        # Matches the NACE classification in postprocessing.py (lines 281-302)
        def _get_fdi_sector(idx):
            """Map model sector index to FDI aggregate sector name."""
            if 0 <= idx <= 2:    return "Agriculture, forestry and fishing"       # A
            elif 3 <= idx <= 7:  return "Mining and Quarrying"                     # B
            elif 8 <= idx <= 31: return "Manufacturing"                            # C
            elif idx == 32:      return "Electricity, Gas, Steam And Air Conditioning Supply"  # D
            elif 33 <= idx <= 36: return "Water supply; sewerage, waste management and remediation activities"  # E
            elif 37 <= idx <= 39: return "Construction"                            # F
            elif 40 <= idx <= 42: return "Wholesale And Retail Trade"              # G
            elif 43 <= idx <= 47: return "Transportation and Storage"              # H
            elif 48 <= idx <= 49: return "Accommodation And Food Service Activities"  # I
            elif 50 <= idx <= 55: return "Information and Communication"           # J
            elif 56 <= idx <= 58: return "Financial and Insurance Activities"      # K
            elif idx == 59:      return "Real Estate Activities"                   # L
            elif 60 <= idx <= 66: return "Professional, Scientific and Technical Activities"  # M
            elif 67 <= idx <= 72: return "Administrative and Support Service Activities"  # N
            elif idx == 73:      return None  # O - Public administration - no FDI
            elif idx == 74:      return "Education"                                # P
            elif 75 <= idx <= 77: return "Human Health and Social Work Activities" # Q
            elif 78 <= idx <= 80: return "Arts, entertainment and recreation"      # R
            elif 81 <= idx <= 83: return "Other service activities"                # S + T
            elif idx == 84:      return None  # Desalination - no FDI equivalent
            else:                return None

        self.fdi_sector_mapping = {idx: _get_fdi_sector(idx) for idx in range(len(self.sectors))}

        # Disaggregate FDI from 18 aggregate sectors to 85 model sectors using dX shares
        n_sectors = len(self.sectors)
        fdi_years = self.ts_fdi_aggregate.index.tolist()
        # Drop the 'Total' column if present
        fdi_agg = self.ts_fdi_aggregate.drop(columns=['Total'], errors='ignore')

        # Build disaggregated FDI array: rows = years, columns = model sectors
        fdi_disagg = np.zeros((len(fdi_years), n_sectors))
        for fdi_sec in fdi_agg.columns:
            # Find model sectors that map to this FDI sector
            model_indices = [i for i, v in self.fdi_sector_mapping.items() if v == fdi_sec]
            if len(model_indices) == 0:
                continue
            # Get dX shares for these model sectors and normalize within the group
            dX_group = self.dX[model_indices]
            dX_sum = np.sum(np.abs(dX_group))
            if dX_sum > 0:
                weights = np.abs(dX_group) / dX_sum
            else:
                # If all dX shares are zero, distribute equally
                weights = np.ones(len(model_indices)) / len(model_indices)
            # Distribute FDI across model sectors
            for yr_idx, yr in enumerate(fdi_years):
                fdi_val = fdi_agg.loc[yr, fdi_sec]
                for j, model_idx in enumerate(model_indices):
                    fdi_disagg[yr_idx, model_idx] = fdi_val * weights[j]

        self.ts_fdi = pd.DataFrame(fdi_disagg, index=fdi_years, columns=self.sectors)
        self.ts_fdi.index.name = 'Year'

        if verbose:
            print(f"FDI NET FLOW data loaded: {len(fdi_years)} years, {n_sectors} model sectors")
            print(f"  Total FDI net flow 2023 (aggregate): {self.ts_fdi_aggregate.loc[2023, 'Total'] if 'Total' in self.ts_fdi_aggregate.columns else self.ts_fdi.loc[2023].sum():,.0f} SAR")
            print(f"  Total FDI net flow 2023 (disaggregated): {self.ts_fdi.loc[2023].sum():,.0f} SAR")

        ###############################################################################################################################
        # 7.3 FDI stock data (cumulative position by sector)
        ###############################################################################################################################
        self.ts_fdi_stock_aggregate = pd.read_csv(
            "input_data/ts_fdi_stock.csv", index_col=0)

        # Disaggregate FDI stock from 18 aggregate sectors to 85 model sectors using dX shares
        fdi_stock_agg = self.ts_fdi_stock_aggregate.drop(columns=['Total'], errors='ignore')
        fdi_stock_years = self.ts_fdi_stock_aggregate.index.tolist()
        fdi_stock_disagg = np.zeros((len(fdi_stock_years), n_sectors))
        for fdi_sec in fdi_stock_agg.columns:
            model_indices = [i for i, v in self.fdi_sector_mapping.items() if v == fdi_sec]
            if len(model_indices) == 0:
                continue
            dX_group = self.dX[model_indices]
            dX_sum = np.sum(np.abs(dX_group))
            if dX_sum > 0:
                weights = np.abs(dX_group) / dX_sum
            else:
                weights = np.ones(len(model_indices)) / len(model_indices)
            for yr_idx, yr in enumerate(fdi_stock_years):
                fdi_val = fdi_stock_agg.loc[yr, fdi_sec]
                for j, model_idx in enumerate(model_indices):
                    fdi_stock_disagg[yr_idx, model_idx] = fdi_val * weights[j]

        self.ts_fdi_stock = pd.DataFrame(fdi_stock_disagg, index=fdi_stock_years, columns=self.sectors)
        self.ts_fdi_stock.index.name = 'Year'

        if verbose:
            print(f"FDI STOCK data loaded: {len(fdi_stock_years)} years, {n_sectors} model sectors")
            print(f"  Total FDI stock 2023 (aggregate): {self.ts_fdi_stock_aggregate.loc[2023, 'Total'] if 'Total' in self.ts_fdi_stock_aggregate.columns else self.ts_fdi_stock.loc[2023].sum():,.0f} SAR")
            print(f"  Total FDI stock 2023 (disaggregated): {self.ts_fdi_stock.loc[2023].sum():,.0f} SAR")

        ###############################################################################################################################
        # 7.4 FDI gross inflow data (by sector)
        ###############################################################################################################################
        self.ts_fdi_gross_aggregate = pd.read_csv(
            "input_data/ts_fdi_gross.csv", index_col=0)

        # Disaggregate FDI gross inflows from 18 aggregate sectors to 85 model sectors using dX shares
        fdi_gross_agg = self.ts_fdi_gross_aggregate.drop(columns=['Total'], errors='ignore')
        fdi_gross_years = self.ts_fdi_gross_aggregate.index.tolist()
        fdi_gross_disagg = np.zeros((len(fdi_gross_years), n_sectors))
        for fdi_sec in fdi_gross_agg.columns:
            model_indices = [i for i, v in self.fdi_sector_mapping.items() if v == fdi_sec]
            if len(model_indices) == 0:
                continue
            dX_group = self.dX[model_indices]
            dX_sum = np.sum(np.abs(dX_group))
            if dX_sum > 0:
                weights = np.abs(dX_group) / dX_sum
            else:
                weights = np.ones(len(model_indices)) / len(model_indices)
            for yr_idx, yr in enumerate(fdi_gross_years):
                fdi_val = fdi_gross_agg.loc[yr, fdi_sec]
                for j, model_idx in enumerate(model_indices):
                    fdi_gross_disagg[yr_idx, model_idx] = fdi_val * weights[j]

        self.ts_fdi_gross = pd.DataFrame(fdi_gross_disagg, index=fdi_gross_years, columns=self.sectors)
        self.ts_fdi_gross.index.name = 'Year'

        if verbose:
            print(f"FDI GROSS INFLOW data loaded: {len(fdi_gross_years)} years, {n_sectors} model sectors")
            print(f"  Total FDI gross inflow 2023 (aggregate): {self.ts_fdi_gross_aggregate.loc[2023, 'Total'] if 'Total' in self.ts_fdi_gross_aggregate.columns else self.ts_fdi_gross.loc[2023].sum():,.0f} SAR")
            print(f"  Total FDI gross inflow 2023 (disaggregated): {self.ts_fdi_gross.loc[2023].sum():,.0f} SAR")

        ###############################################################################################################################
        # 7.5 FDI stock accounting verification
        ###############################################################################################################################
        fdi_stock_agg_total = self.ts_fdi_stock_aggregate.drop(columns=['Total'], errors='ignore')
        fdi_gross_agg_total = self.ts_fdi_gross_aggregate.drop(columns=['Total'], errors='ignore')
        common_years = sorted(set(fdi_stock_agg_total.index) & set(fdi_gross_agg_total.index))
        if len(common_years) > 1:
            stock_check_results = []
            for i in range(1, len(common_years)):
                yr_prev = common_years[i - 1]
                yr_curr = common_years[i]
                stock_prev = fdi_stock_agg_total.loc[yr_prev].sum()
                stock_curr = fdi_stock_agg_total.loc[yr_curr].sum()
                gross_curr = fdi_gross_agg_total.loc[yr_curr].sum()
                implied_stock = stock_prev + gross_curr
                diff = stock_curr - implied_stock
                stock_check_results.append((yr_curr, stock_curr, implied_stock, diff))
            if verbose:
                print(f"\n  FDI Stock accounting check: Stock(t) vs Stock(t-1) + Gross_inflow(t)")
                for yr, actual, implied, diff in stock_check_results:
                    print(f"    {yr}: Stock={actual:,.0f}  Implied={implied:,.0f}  Diff={diff:,.0f}")
                max_diff = max(abs(d) for _, _, _, d in stock_check_results)
                if max_diff < 1:
                    print(f"    => PASS: Stock accounting identity holds (max diff = {max_diff:.0f})")
                else:
                    print(f"    => NOTE: Stock accounting does not exactly hold (max diff = {max_diff:,.0f})")
                    print(f"         This is expected: difference = valuation changes, exchange rate effects, etc.")


        ###############################################################################################################################
        ###############################################################################################################################
        # 7.6 FDI calibration — growth targets and base-year values
        ###############################################################################################################################
        ###############################################################################################################################

        # Base year (2021) FDI values — these anchor the FDI module's starting point
        # Use 2021 if available, otherwise use earliest year in data
        fdi_base_year = 2021 if 2021 in self.ts_fdi.index else self.ts_fdi.index[0]

        # Sectoral net FDI in base year (85-sector vector)
        self.FDI_base_year_net_ = self.ts_fdi.loc[fdi_base_year].values.copy()
        # Total net FDI in base year
        self.FDI_base_year_net_total = np.sum(self.FDI_base_year_net_)

        # Sectoral FDI stock in base year
        if fdi_base_year in self.ts_fdi_stock.index:
            self.FDI_base_year_stock_ = self.ts_fdi_stock.loc[fdi_base_year].values.copy()
        else:
            self.FDI_base_year_stock_ = np.zeros(n_sectors)

        # Sectoral gross FDI in base year
        if fdi_base_year in self.ts_fdi_gross.index:
            self.FDI_base_year_gross_ = self.ts_fdi_gross.loc[fdi_base_year].values.copy()
        else:
            self.FDI_base_year_gross_ = np.zeros(n_sectors)

        # FDI share of total investment in base year (how much of I_total is financed by FDI)
        if self.I_total > 0:
            self.FDI_share_of_investment_base = np.abs(self.FDI_base_year_net_total) / self.I_total
        else:
            self.FDI_share_of_investment_base = 0.0

        # Sectoral FDI shares of sectoral investment (85 sectors)
        # Use absolute values of FDI to handle potential negative net flows in some sectors
        self.FDI_share_of_investment_ = np.zeros(n_sectors)
        for i in range(n_sectors):
            if self.gross_capital_formation[i] > 0:
                self.FDI_share_of_investment_[i] = np.abs(self.FDI_base_year_net_[i]) / self.gross_capital_formation[i]

        # FDI growth target calibration (analogous to V2030 targets)
        # Convert total cumulative growth target to annual growth rate over vision_2030_timing years
        years_to_2030_timing = parameters.vision_2030_timing
        if parameters.FDI_target != 0 and years_to_2030_timing > 0:
            self.FDI_yearly_target_growth_rate = (1 + parameters.FDI_target) ** (1 / years_to_2030_timing) - 1
        else:
            self.FDI_yearly_target_growth_rate = 0.0

        # Sectoral FDI distribution weights (how to distribute aggregate FDI across sectors)
        # Uses same output-share (dX) weights as FDI disaggregation
        fdi_net_abs = np.abs(self.FDI_base_year_net_)
        fdi_net_total_abs = np.sum(fdi_net_abs)
        if fdi_net_total_abs > 0:
            self.FDI_sectoral_distribution_ = fdi_net_abs / fdi_net_total_abs
        else:
            self.FDI_sectoral_distribution_ = self.dX.copy()

        if verbose:
            print(f"\nFDI Calibration:")
            print(f"  Base year: {fdi_base_year}")
            print(f"  Base year net FDI total: {self.FDI_base_year_net_total:,.0f} SAR")
            print(f"  FDI share of total investment: {self.FDI_share_of_investment_base:.2%}")
            print(f"  FDI_target (cumulative): {parameters.FDI_target:.1%}")
            print(f"  FDI yearly growth rate: {self.FDI_yearly_target_growth_rate:.4%}")
            print(f"  enable_FDI: {parameters.enable_FDI}")

        ###############################################################################################################################
        # 7.7 Remittances data
        # Source: World Development Indicators, Personal remittances paid (current US$)
        # Converted to 1000 SAR at 1 USD = 3.75 SAR by process_remittances_data.py
        ###############################################################################################################################
        self.ts_remittances = pd.read_csv(
            "input_data/ts_remittances.csv", index_col=0).squeeze()
        self.ts_remittances.index.name = 'Year'

        # Base year remittances value (2021)
        self.remittances_base_year = self.ts_remittances.loc[2021]

        # Calibrate average remittances growth rate (compound annual growth from estimation window)
        remit_start = max(startyear, self.ts_remittances.dropna().index.min())
        remit_end = min(endyear, self.ts_remittances.dropna().index.max())
        remit_years = remit_end - remit_start
        if remit_years > 0 and self.ts_remittances.loc[remit_start] > 0:
            self.gRemittances_avg = (
                self.ts_remittances.loc[remit_end] / self.ts_remittances.loc[remit_start]
            ) ** (1.0 / remit_years) - 1.0
        else:
            self.gRemittances_avg = 0.0

        # Remittances as share of total wages (for reference)
        self.remittances_share_of_wages = self.remittances_base_year / np.sum(self.W_)

        # Re-calibrate α2 now that remittances_base_year is available.
        # Wage-financed gross consumption in the MODEL = α₁ × (YD_wage − remittances),
        # where YD_wage in the model already includes gov_exp_resid (social/transfer spending
        # added to household wage income in model.py §12.6).  We must use the same definition
        # here so that at the base year:
        #   C_gross = (YD_wage + gov_exp_resid − remittances) + α₂ × YD_profit
        # which implies α₂ = (C_gross − (YD_wage + gov_exp_resid − remittances)) / YD_profit.
        # Note: gov_exp_resid is already in the V equation via YD_wage; do NOT add it separately.
        # α2 can remain fixed in model runs because in Mode A (remittances_follow_wage_growth=True)
        # remittances and wages grow proportionally, keeping the leakage share stable over time.
        C_wage_financed = self.YD_wage + self.gov_exp_resid - self.remittances_base_year
        self.C_profit_inc_gross = self.C_gross - C_wage_financed
        self.α2 = self.C_profit_inc_gross / self.YD_profit
        if verbose:
            print(f"  α2 calibrated (with gov_exp_resid + remittances leakage): {self.α2:.4f}"
                  f"  (naive wages-only: {(self.C_gross - self.YD_wage) / self.YD_profit:.4f})")

        ###############################################################################################################################
        # 7.8 Balance of Payments initial values
        ###############################################################################################################################
        # Government Gross External Asset Stock (Gov_ext_assets) — initial stock
        # Proxy: PIF value (self.GV) represents the bulk of Saudi government foreign assets
        # In reality Gov_ext_assets = SAMA reserves + PIF foreign holdings (gross; liabilities tracked separately)
        # Using GV as approximation; can be refined with SAMA reserve data
        # Toggle: initialize the gross external asset stock from the PIF value or from the net investment position (GV).
        PIF_initialization = False  # If True, use PIF value as proxy for government gross external assets; if False, use GV (net international investment position)
        if PIF_initialization:
            self.Gov_ext_assets_initial = self.PIF
        else:
            self.Gov_ext_assets_initial = self.GV  # Using GV (net international investment position) as proxy for government gross external assets

        if verbose:
            print(f"\nRemittances data loaded: {len(self.ts_remittances)} years")
            print(f"  Base year (2021) remittances: {self.remittances_base_year:,.0f} (1000 SAR) = {self.remittances_base_year/1e6:.2f} Bn SAR")
            print(f"  Remittances / GDP (2021): {self.remittances_base_year / self.GDP:.4%}")
            print(f"  Remittances / Wages (2021): {self.remittances_share_of_wages:.4%}")
            print(f"  Avg. remittances growth rate ({remit_start}-{remit_end}): {self.gRemittances_avg:.4%}")
            print(f"  Gov_ext_assets initial (PIF proxy): {self.Gov_ext_assets_initial:,.0f} (1000 SAR) = {self.Gov_ext_assets_initial/1e6:.2f} Bn SAR")


        ###############################################################################################################################
        ###############################################################################################################################
        ###############################################################################################################################
        # VIII. Energy block — historical electricity data, RE costs, and investment
        ###############################################################################################################################
        ###############################################################################################################################
        ###############################################################################################################################

        self.GJ_PER_GWH = 3600  # 1 GWh = 1000 MWh = 3600 GJ
        # 1. Load NEW Historical Electricity Mix Data (GWh)
        # Historical electricity mix data (GWh)
        hist_data_str = """Oil   68103   2000     GWh
        Oil 63535   2001     GWh
        Oil 62584   2002     GWh
        Oil 69089   2003     GWh
        Oil 68840   2004     GWh
        Oil 76627   2005     GWh
        Oil 84809   2006     GWh
        Oil 94809   2007     GWh
        Oil 104498  2008     GWh
        Oil 119802  2009     GWh
        Oil 129298  2010     GWh
        Oil 141700  2011     GWh
        Oil 171961  2012     GWh
        Oil 169566  2013     GWh
        Oil 209319  2014     GWh
        Oil 225478  2015     GWh
        Oil 195868  2016     GWh
        Oil 174122  2017     GWh
        Oil 163398  2018     GWh
        Oil 168219  2019     GWh
        Oil 164077  2020     GWh
        Oil 168805  2021     GWh
        Oil 172378  2022     GWh
        Oil 181659  2023     GWh
        Natural gas 58088   2000     GWh
        Natural gas 70138   2001     GWh
        Natural gas 79152   2002     GWh
        Natural gas 83911   2003     GWh
        Natural gas 91035   2004     GWh
        Natural gas 99497   2005     GWh
        Natural gas 96625   2006     GWh
        Natural gas 95726   2007     GWh
        Natural gas 99702   2008     GWh
        Natural gas 97280   2009     GWh
        Natural gas 110769  2010     GWh
        Natural gas 108377  2011     GWh
        Natural gas 117368  2012     GWh
        Natural gas 133471  2013     GWh
        Natural gas 127756  2014     GWh
        Natural gas 134757  2015     GWh
        Natural gas 173072  2016     GWh
        Natural gas 205250  2017     GWh
        Natural gas 225639  2018     GWh
        Natural gas 219121  2019     GWh
        Natural gas 229485  2020     GWh
        Natural gas 238876  2021     GWh
        Natural gas 243932  2022     GWh
        Natural gas 257066  2023     GWh
        Wind        2000     GWh
        Wind        2001     GWh
        Wind        2002     GWh
        Wind        2003     GWh
        Wind        2004     GWh
        Wind        2005     GWh
        Wind        2006     GWh
        Wind        2007     GWh
        Wind        2008     GWh
        Wind        2009     GWh
        Wind        2010     GWh
        Wind        2011     GWh
        Wind        2012     GWh
        Wind        2013     GWh
        Wind        2014     GWh
        Wind        2015     GWh
        Wind        2016     GWh
        Wind        2017     GWh
        Wind        2018     GWh
        Wind        2019     GWh
        Wind        2020     GWh
        Wind    132 2021     GWh
        Wind    1588    2022     GWh
        Wind    1588    2023     GWh
        Solar PV        2000     GWh
        Solar PV        2001     GWh
        Solar PV        2002     GWh
        Solar PV        2003     GWh
        Solar PV        2004     GWh
        Solar PV        2005     GWh
        Solar PV        2006     GWh
        Solar PV    0   2007     GWh
        Solar PV        2008     GWh
        Solar PV        2009     GWh
        Solar PV    4   2010     GWh
        Solar PV    5   2011     GWh
        Solar PV    5   2012     GWh
        Solar PV    26  2013     GWh
        Solar PV    42  2014     GWh
        Solar PV    46  2015     GWh
        Solar PV    46  2016     GWh
        Solar PV    65  2017     GWh
        Solar PV    65  2018     GWh
        Solar PV    320 2019     GWh
        Solar PV    916 2020     GWh
        Solar PV    867 2021     GWh
        Solar PV    980 2022     GWh
        Solar PV    1049    2023     GWh
        Solar thermal       2000     GWh
        Solar thermal       2001     GWh
        Solar thermal       2002     GWh
        Solar thermal       2003     GWh
        Solar thermal       2004     GWh
        Solar thermal       2005     GWh
        Solar thermal       2006     GWh
        Solar thermal       2007     GWh
        Solar thermal       2008     GWh
        Solar thermal       2009     GWh
        Solar thermal       2010     GWh
        Solar thermal       2011     GWh
        Solar thermal       2012     GWh
        Solar thermal       2013     GWh
        Solar thermal       2014     GWh
        Solar thermal       2015     GWh
        Solar thermal       2016     GWh
        Solar thermal       2017     GWh
        Solar thermal   11  2018     GWh
        Solar thermal   135 2019     GWh
        Solar thermal   135 2020     GWh
        Solar thermal   135 2021     GWh
        Solar thermal   135 2022     GWh
        Solar thermal   135 2023     GWh"""

        parsed_data = []
        for line in hist_data_str.strip().split('\n'):
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                year = int(parts[-2])
                value = float(parts[-3])
            except ValueError:
                value = 0.0
            source = " ".join(parts[:-3])
            if not source:
                continue
            parsed_data.append({'Source': source, 'GWh': value, 'Year': year})

        df_hist = pd.DataFrame(parsed_data)
        df_hist['Source'] = df_hist['Source'].replace(
            ['Solar PV', 'Solar thermal'], 'Solar')
        df_agg = df_hist.groupby(['Year', 'Source'])['GWh'].sum().reset_index()
        df_pivot_gwh = df_agg.pivot(
            index='Year', columns='Source', values='GWh').fillna(0)

        # 2. Add Bioenergy Data
        # (Bioenergy is absent from the updated data source, so it is added back from earlier data)
        bioenergy_data = {
            'Year': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
            'Bioenergy': [25, 26, 28, 30, 32, 35, 38, 40, 42, 45]
        }
        df_bio = pd.DataFrame(bioenergy_data).set_index('Year')
        df_pivot_gwh = df_pivot_gwh.join(df_bio, how='left').fillna(0)

        # 3. Store Historical Data for Model Years (2021, 2022, 2023)
        # We store them as numpy arrays for t=0, t=1, t=2
        hist_years = [2021, 2022, 2023]
        df_hist_model_years = df_pivot_gwh.loc[hist_years]

        # Store values in GJ for the model to use
        self.hist_elec_oil_gj = (
            df_hist_model_years['Oil'].to_numpy() * self.GJ_PER_GWH)
        self.hist_elec_gas_gj = (
            df_hist_model_years['Natural gas'].to_numpy() * self.GJ_PER_GWH)
        self.hist_elec_solar_gj = (
            df_hist_model_years['Solar'].to_numpy() * self.GJ_PER_GWH)
        self.hist_elec_wind_gj = (
            df_hist_model_years['Wind'].to_numpy() * self.GJ_PER_GWH)
        self.hist_elec_bio_gj = (
            df_hist_model_years['Bioenergy'].to_numpy() * self.GJ_PER_GWH)
        # Scaling factor for historical electricity data (1.0 = no scaling applied)
        reduction_factor = 1
        self.hist_elec_oil_gj *= reduction_factor
        self.hist_elec_gas_gj *= reduction_factor
        self.hist_elec_solar_gj *= reduction_factor
        self.hist_elec_wind_gj *= reduction_factor
        self.hist_elec_bio_gj *= reduction_factor
        # 4. Calculate 2023 Shares for Future Projections
        # Get 2023 (t=2) generation values in GJ
        gen_2023_solar_gj = self.hist_elec_solar_gj[2]
        gen_2023_wind_gj = self.hist_elec_wind_gj[2]
        gen_2023_bio_gj = self.hist_elec_bio_gj[2]
        total_re_2023_gj = gen_2023_solar_gj + gen_2023_wind_gj + gen_2023_bio_gj

        # Calculate shares *within* the renewable mix
        self.re_solar_share = gen_2023_solar_gj / total_re_2023_gj
        self.re_wind_share = gen_2023_wind_gj / total_re_2023_gj
        self.re_bio_share = gen_2023_bio_gj / total_re_2023_gj

        # Get 2023 (t=2) fossil fuel generation for electricity
        gen_2023_oil_gj = self.hist_elec_oil_gj[2]
        gen_2023_gas_gj = self.hist_elec_gas_gj[2]
        total_ff_2023_gj = gen_2023_oil_gj + gen_2023_gas_gj

        # Calculate shares *within* the fossil fuel mix (for future gap-filling)
        self.ff_oil_share_2023 = gen_2023_oil_gj / total_ff_2023_gj
        self.ff_gas_share_2023 = gen_2023_gas_gj / total_ff_2023_gj

        # 5. Store Initial (t=0, 2021) Renewable Generation
        self.initial_RE_generation_gj = (
            self.hist_elec_solar_gj[0] +
            self.hist_elec_wind_gj[0] +
            self.hist_elec_bio_gj[0]
        )
        # 5. Rename initial RE generation for get_starting_values
        self.initial_RE_generation = self.initial_RE_generation_gj
        if verbose:
            print("--- New Historical Data & Shares Calibrated ---")
            print(f"  2023 RE Solar Share: {self.re_solar_share:.2%}")
            print(f"  2023 RE Wind Share: {self.re_wind_share:.2%}")
            print(f"  2023 RE Bio Share: {self.re_bio_share:.2%}")
            print(f"  2023 FF Oil Share: {self.ff_oil_share_2023:.2%}")
            print(f"  2023 FF Gas Share: {self.ff_gas_share_2023:.2%}")
            print("-------------------------------------------------")
            print("--- Calibrating Vision 2030 Ramp... ---")

        # 1. Calculate total generation for the historical years
        #    self.hist_..._gj is an array with 3 values: [2021_val, 2022_val, 2023_val]
        total_re_gj_hist = (
            self.hist_elec_solar_gj +
            self.hist_elec_wind_gj +
            self.hist_elec_bio_gj
        )

        total_all_gen_gj_hist = (
            self.hist_elec_oil_gj +
            self.hist_elec_gas_gj +
            total_re_gj_hist
        )

        # 2. Calculate the historical shares (element-wise division)
        #    This gives an array [share_2021, share_2022, share_2023]
        historical_shares = np.divide(
            total_re_gj_hist,
            total_all_gen_gj_hist,
            out=np.zeros_like(total_re_gj_hist, dtype=float),
            where=total_all_gen_gj_hist != 0
        )

        # 3. Extract the dynamically calculated shares
        base_share_2021 = historical_shares[0]  # For t=1 (2021)
        base_share_2022 = historical_shares[1]  # For t=2 (2022)
        start_ramp_share_2023 = historical_shares[2]  # For t=3 (2023)

        # 4. Define the target and create the ramp
        target_share_2030 = 0.50  # 50% target for t=10 (year 2030)

        # Create the ramp from 2023 (t=3) to 2030 (t=10)
        # This is 8 points total (t=3, 4, 5, 6, 7, 8, 9, 10)
        ramp_t3_to_t10 = np.linspace(
            start_ramp_share_2023,
            target_share_2030,
            num=8  # (2030 is t=10. 10 - 3 + 1 = 8 points)
        )

        # 5. Concatenate all years to create the final ramp
        # This list will have 10 items (indices 0-9), for t=1 to t=10
        self.re_target_share_ramp = np.concatenate(
            ([base_share_2021, base_share_2022], ramp_t3_to_t10)
        )

        if verbose:
            print(f"  t=1 (2021) Share: {self.re_target_share_ramp[0]:.4%}")
            print(f"  t=2 (2022) Share: {self.re_target_share_ramp[1]:.4%}")
            print(f"  t=3 (2023) Share: {self.re_target_share_ramp[2]:.4%}")
            print(f"  t=10 (2030) Share: {self.re_target_share_ramp[-1]:.4%}")
            print("--- Vision 2030 Ramp Shares Calibrated. ---")
            # --- END: NEW HISTORICAL DATA & RENEWABLE CALIBRATION ---

        ########################################################################
        # 8.1 Renewable energy cost and investment calibration
        ########################################################################

        # --- Unit Conversions and Constants ---
        USD_TO_SAR = 3.75
        THOUSAND = 1000
        GJ_PER_MWH = 3.6
        HOURS_PER_YEAR = 8760

        # --- Technology-Specific Assumptions ---
        technologies = {
            'Solar PV': {
                'capex_usd_per_mw': 691000,
                'lcoe_usd_per_mwh': 10.4,
                'capacity_factor': 0.28
            },
            'Onshore Wind': {
                'capex_usd_per_mw': 1041000,
                'lcoe_usd_per_mwh': 15.7,
                'capacity_factor': 0.40
            },
            'Bioenergy (WTE)': {
                'capex_usd_per_mw': 4900000,
                'lcoe_usd_per_mwh': 100.0,
                'capacity_factor': 0.85
            }
        }

# --- Calculate Weighted Average Costs based on 2023 RE Mix ---
        total_share = self.re_solar_share + self.re_wind_share + self.re_bio_share
        
        # 1. Calculate specific CAPEX per GJ (Thousand SAR / GJ Annual Capacity) for EACH Tech
        # Formula: (USD/MW * 3.75 / 1000) / (CapacityFactor * 8760 * 3.6)
        
        def get_capex_per_gj(tech_name):
            capex_mw = technologies[tech_name]['capex_usd_per_mw']
            cf = technologies[tech_name]['capacity_factor']
            # Cost per MW converted to Thousand SAR
            cost_thousand_sar = capex_mw * USD_TO_SAR / THOUSAND
            # Annual Output in GJ per MW
            annual_gj = cf * HOURS_PER_YEAR * GJ_PER_MWH
            return cost_thousand_sar / annual_gj

        self.capex_solar_per_gj = get_capex_per_gj('Solar PV')
        self.capex_wind_per_gj = get_capex_per_gj('Onshore Wind')
        self.capex_bio_per_gj = get_capex_per_gj('Bioenergy (WTE)')

        # 2. Store the weighted average (keep this for backward compatibility if needed)
        self.re_capex_per_gj_year = (
            (self.capex_solar_per_gj * self.re_solar_share) +
            (self.capex_wind_per_gj * self.re_wind_share) +
            (self.capex_bio_per_gj * self.re_bio_share)
        ) / total_share

        # OpEx in Thousand SAR per GJ of energy produced (used as price)
        # (We can keep using the average for OpEx or split it similarly if needed later)
        avg_lcoe_usd_per_mwh = (
            (technologies['Solar PV']['lcoe_usd_per_mwh'] * self.re_solar_share) +
            (technologies['Onshore Wind']['lcoe_usd_per_mwh'] * self.re_wind_share) +
            (technologies['Bioenergy (WTE)']['lcoe_usd_per_mwh'] * self.re_bio_share)
        ) / total_share
        
        self.re_opex_per_gj = (avg_lcoe_usd_per_mwh * USD_TO_SAR / THOUSAND) / GJ_PER_MWH

        # Assume a 30-year lifetime for renewable plants for depreciation
        self.delta_RE = 1 / 30

        # Define the sectoral composition of renewable energy investment (dI_RE)
        re_investment_shares = {
            'Construction of buildings': 20,
            'Civil engineering': 15,
            'Manufacture of machinery and equipment n.e.c.': 25,
            'Manufacture of electrical equipment': 20,
            'Manufacture of fabricated metal products, except machinery and equipment': 10,
            'Architectural and engineering activities; technical testing and analysis': 10
        }
        _dI_RE_raw = np.array(
            [re_investment_shares.get(sector, 0) for sector in self.sectors]
        )
        # Normalize robustly by the actual sum (consistent with dI_Effic_ and dI_Electrification_)
        # Using /100 assumed all 6 sector names matched exactly; /sum() is safe even if any name differs.
        self.dI_RE_ = _dI_RE_raw / _dI_RE_raw.sum() if _dI_RE_raw.sum() > 0 else _dI_RE_raw
        raw_re_2020_gj = df_pivot_gwh.loc[2020, ['Solar', 'Wind', 'Bioenergy']].sum() * self.GJ_PER_GWH
        raw_re_2021_gj = df_pivot_gwh.loc[2021, ['Solar', 'Wind', 'Bioenergy']].sum() * self.GJ_PER_GWH

        # 2. Determine the reduction factor currently in use
        # Compare the stored (reduced) 2021 value with the raw 2021 value
        if raw_re_2021_gj > 0:
            current_reduction_factor = self.initial_RE_generation_gj / raw_re_2021_gj
        else:
            current_reduction_factor = 1.0

        # 3. Apply this factor to 2020 data to be consistent
        re_2020_reduced_gj = raw_re_2020_gj * current_reduction_factor
        re_2021_reduced_gj = self.initial_RE_generation_gj # This is already reduced

        # 4. Calculate the capacity added in 2021
        # Formula: New Capacity = K(t) - K(t-1) * (1 - depreciation)
        depreciated_capacity_2020 = re_2020_reduced_gj * (1 - self.delta_RE)
        capacity_added_2021 = np.maximum(0, re_2021_reduced_gj - depreciated_capacity_2020)

        # 5. Calculate Investment in Thousand SAR (using calibrated CapEx cost)
        self.I_RE = capacity_added_2021 * self.re_capex_per_gj_year


        ########################################################################
        # 8.2 Water module integration
        ########################################################################
        # Determine scenario for water module
        water_scenario = "BAU"  # Default to BAU
        if scenario_config:
            if scenario_config.get('Baseline_scenario'):
                water_scenario = "BAU"
            elif scenario_config.get('Vision_2030_scenario'):
                water_scenario = "2030 Vision"
            elif scenario_config.get('Transformation_scenario'):
                water_scenario = "Transformation"
            elif scenario_config.get('Zero_growth_rates_steady_state'):
                water_scenario = "BAU"
           

        start_sim_year = 2021 # Assuming t=1 corresponds to 2021
        end_sim_year = 2065   # Max year in water module
        
        try:
            # get_agr_data returns dicts {Year: Value}
             # Modified signature in water_module.py to just take scenario_name
            crop_X_dict, agr_intensity_dict, agr_irrigation_invest_dict, agr_irrigation_om_dict, agr_irrigation_subsidy_dict = get_agr_data(water_scenario, verbose=verbose)
            
            self.crop_X_dict = crop_X_dict
            self.agr_water_use_intensity_dict = agr_intensity_dict
            self.agr_irrigation_invest_dict = agr_irrigation_invest_dict
            self.agr_irrigation_om_dict = agr_irrigation_om_dict
            self.agr_irrigation_subsidy_dict = agr_irrigation_subsidy_dict

       
            # Convert to arrays aligned with simulation time t
            # index i corresponds to year = start_sim_year + i
            max_t = end_sim_year - start_sim_year + 1
            self.crop_X = np.zeros(max_t)
            self.AGR_X = np.zeros(max_t)
            self.agr_water_use_intensity = np.zeros(max_t)
            self.agr_irrigation_invest = np.zeros(max_t)
            self.agr_irrigation_om = np.zeros(max_t)
            self.agr_irrigation_subsidy = np.zeros(max_t)
            
            for i in range(max_t):
                y = start_sim_year + i
                self.crop_X[i] = crop_X_dict.get(y, 0.0)
                # Calculate AGR_X: Total Output = Crop Output / 0.34
                if self.crop_X[i] > 0:
                    self.AGR_X[i] = self.crop_X[i] / 0.34
                else:
                    self.AGR_X[i] = 0.0
                self.agr_water_use_intensity[i] = agr_intensity_dict.get(y, 0.0)
                self.agr_irrigation_invest[i] = agr_irrigation_invest_dict.get(y, 0.0)
                self.agr_irrigation_om[i] = agr_irrigation_om_dict.get(y, 0.0)
                self.agr_irrigation_subsidy[i] = agr_irrigation_subsidy_dict.get(y, 0.0)
                
        except Exception as e:
            print(f"Error calling water module: {e}")
            # Initialize with zeros to avoid crashes if files are missing
            self.AGR_X = np.zeros(100)
            self.agr_water_use_intensity = np.zeros(100)
            self.agr_irrigation_invest = np.zeros(100)
            self.agr_irrigation_om = np.zeros(100)
            self.agr_irrigation_subsidy = np.zeros(100)

                

        ########################################################################
        # 8.3 Electrification strategy calibration
        ########################################################################
        
        # 1. Identify Fossil Fuel Intensive Sectors (Top 15 excluding Electricity sector)
        total_fossil_intensity = self.gas_intensities_ + self.oil_intensities_
        sorted_fossil_indices = np.argsort(total_fossil_intensity)
        
        elec_idx = self.electricity_sector_idx
        target_indices = [i for i in sorted_fossil_indices if i != elec_idx]
        
        # Select Top 15 most intensive
        self.electrification_target_sectors = np.array(target_indices[-15:])
        
        # 2. Define Investment Distribution (Machinery & Construction)
        electrification_inv_shares = {
            'Manufacture of machinery and equipment n.e.c.': 40,
            'Manufacture of electrical equipment': 30,
            'Specialized construction activities': 20,
            'Construction of buildings': 10
        }
        
        self.dI_Electrification_ = np.array(
            [electrification_inv_shares.get(sector, 0) for sector in self.sectors]
        )
        # Normalize
        if self.dI_Electrification_.sum() > 0:
            self.dI_Electrification_ = self.dI_Electrification_ / self.dI_Electrification_.sum()
    ###############################################################################################################################
    ###############################################################################################################################
    ###############################################################################################################################
    # IX. Starting values
    ###############################################################################################################################
    ###############################################################################################################################
    ###############################################################################################################################

    def get_starting_values(self):
        """These values are used to initialize dynamic variables in round zero."""
        return {
            "renewable_energy_generation": self.initial_RE_generation,
            "Y_":           self.Y_,
            "y_":           self.Y_,
            "Y_non_oil_":   self.Y_non_oil_,
            "Y_oil_":       self.Y_oil_,
            "Y_gov_":       self.Y_gov_,
            "y_gov_":       self.Y_gov_,
            "Y_net_taxes":  self.Y_net_taxes,
            "Y_other_gdp":  np.sum(self.Y_gov_) + self.Y_net_taxes,
            "X_Y_ratio_":   self.X_Y_ratio_,
            "A__":          self.A,
            "Z__":          self.io_table,
            # YD_wage at t=0 must match the model's own YD_wage formula:
            #   YD_wage[t] = W_[t] - Tax_wages[t] + gov_exp_resid
            # so initialise with wages + gov_exp_resid (Tax_wages=0 at base year).
            # This prevents a step-change in consumption at t=2 when the model formula kicks in.
            "YD_wage":      self.YD_wage + self.gov_exp_resid,
            # Remittances at t=0: use base-year value so that t=1 consumption correctly
            # deducts remittances (v.remittances[t-1=0] must not be zero).
            "remittances":  self.remittances_base_year,
            "YD_profit":    self.YD_profit,
            "GY":           self.GY,
            "GV":           self.GV,
            "Gov_net_wealth": self.Gov_net_wealth,
            "Bond_domestic": self.Bond_domestic,
            "Bond_external": self.Bond_external_init,
            "I_total":      self.I_total,
            "EX_":          self.EX_,
            "EX_oil_":      self.EX_oil_,
            "EX_non_oil_":  self.EX_non_oil_,
            "IM_":          self.IM_,
            "K_":           self.K_,
            "k_":           self.K_,  # Nominal = real at the start
            "L_":           self.L_,
            "W_":           self.W_,
            "X_":           self.X_,
            "x_":           self.X_,
            "dX_":          self.dX,
            "x_":           self.x_,
            "k_":           self.k_,
            "IntP_":        self.IntP_,
            "intP_":        self.IntP_,
            "IntP__":       self.io_table,  # THIS is the WHOLE IO table
            "IntS_":        self.IntS_,
            "intS_":        self.IntS_,
            "P_":           self.P_,
            "P_tot_":       self.P_tot_,
            "unit_cost_smooth_": np.where(self.X_ != 0, (self.X_ - self.P_tot_) / self.X_, 1.0),
            "inflation":    self.inflation,
            "inflation_consumers": self.inflation,
            "PB":           self.PB,
            "p_":           self.p_,
            "u_":           self.u_,
            "alpha_":       self.alpha_,
            "beta_":        self.beta_,
            "kappa_":       self.kappa_,
            "gY_":          self.gY_2021,
            "gI_trend_endog":   self.gI_avg,
            "invest_profit":    self.invest_profit,
            "gV":               0,
            "Gov_ext_assets":   self.Gov_ext_assets_initial,
        }

    ###############################################################################################################################
    ###############################################################################################################################
    ###############################################################################################################################
    # X. Helper methods
    ###############################################################################################################################
    ###############################################################################################################################
    ###############################################################################################################################

    def prepare_io_table(self) -> pd.DataFrame:
        """Load IO table and add extra row & col for desalination."""
        df = self.read_csv('input_data/io_table.csv')

        # Drop Activities of households as employers of domestic personnel
        df = df.drop(
            columns=["Activities of households as employers of domestic personnel"])
        df = df.drop(index=84)

        # Add extra sectors (desalination)
        if self.Add_new_wastewater_sector:
            extra_sectors = ["Desalination", "Wwater"]
        else:
            extra_sectors = ["Desalination"]
        for sector in extra_sectors:
            df[sector] = 0.
            df = self.add_row(df)
        return df

    def add_row(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a row to the DataFrame."""
        cols = df.columns
        row = pd.DataFrame([[0.] * len(cols)], columns=cols)
        return pd.concat([df, row], ignore_index=True)

    def read_csv(self, path: str, delimiter: str = ';', add_rows=0, add_cols=[]) -> pd.DataFrame:

        # Get column names (trim whitespaces)
        columns = list(pd.read_csv(path, delimiter=delimiter, nrows=0).columns)
        columns = [col.strip() for col in columns]

        # Read the CSV file as a string
        with open(path, 'r') as file:
            data = file.read()

        # Remove spaces and dashes
        data = data.replace(',', '.')

        # Convert the cleaned string to a DataFrame (Fill NaNs with 0)
        df = pd.read_csv(StringIO(data), delimiter=delimiter,
                         dtype=float).fillna(0)
        df.columns = columns

        for _ in range(add_rows):
            df = self.add_row(df)

        for col in add_cols:
            df[col] = 0.

        return df
