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
import numpy as np
import pandas as pd
import warnings
from .water_module import replace_agr_X
###################################################################################################################################################################################################################################################################################################
def endogenize_input_output_matrix_A(v: ModelVariables, p: ModelParameters, pc: ParametersCalibrated, t: int):
###################################################################################################################################################################################################################################################################################################    
    """
    Assumption: Directed industrial policy by the government in conjunction with 
    additional investment allows to increase the domestic production and decrease the import dependency over time
    This is merged with direct Vision 2030 sector growth effects directly in selected Vision 2030 sectors, their supplying sectors, and in the investment growth rate.
    Endogenous growth rate is above the trend level from calibrated data pc.gI_avg if excess imports prevail for a sector.
    """

    # Rank sectors by excess scarcity
    import_dependency_rank_ = np.argsort(-v.excess_import_[t])  # Descending order of scarcity
    ####################################################################################
    # Work with previous period's variable values before updating X and other variables
    ####################################################################################
    v.X_[t] = np.dot(np.linalg.inv(pc.I - v.A__[t]), v.Y_[t])
    
    # Re-create Z matrix from last period to have a direct representation of IO Shares
    v.Z__[t] = v.A__[t]  @ np.diag(v.X_[t])
    # This is the same as the Z matrix
    v.IntP__[t] = v.A__[t] * v.X_[t]  # Intermediate purchases as matrix, this is the Z matrix
    # Eq (14) Intermediate sales
    v.IntS_[t] = np.dot(v.A__[t], v.X_[t])
    # Eq (15) Intermediate purchases
    v.IntP_[t] = np.sum(v.A__[t], axis=0) * v.X_[t]
    # Derive intermediate-purchase shares (a proxy for A), used when endogenizing the A matrix
    v.IntP_shares__[t] = np.divide(v.IntP__[t], v.IntP_[t], out=np.zeros_like(v.IntP__[t], dtype=float), where=v.IntP_[t] != 0)  # Shares of intermediate purchases
    
    # p.Update_to_changes_A is set to true.
    # Directed industrial policy switch: governs endogenous import substitution in V2030 sectors.
    ###################################################################################################################################################################################################################################################################################################


    # VISION 2030 induced GROWTH STEP 1.: Calculate the amount of increase in DIRECT activity of the vision 2030 sectors itself, and reduce part of the imports of this sector accordingly 
    ###################################################################################################################################################################################################################################################################################################
    # Modify investments in the vision 2030 directly affected sector and in the supplying sectors accordingly, to reflect increased production capacity#
    # Simplified assumption now: investment growth is in line with the increased production growth rates of the vision 2030 direct and int. input supply sectors
    # Note: Vision 2030 sector growth is applied within the specified policy horizon (vision_2030_timing).
    # For extended scenarios, the extended_vision_2030_timing parameter governs the additional period.
    if p.Vision_2030_activity_increase and 1 <= t <= p.vision_2030_timing or 1 <= t <= p.extended_vision_2030_timing: 
        v.Y_[t][pc.vision2030_sectors] = v.Y_[t][pc.vision2030_sectors] * (1 + pc.sectoral_yearly_growth_rates_V2030_[pc.vision2030_sectors])  # Increase domestic production by the amount of vision 2030 induced growth
        vision_2030_sectors_import_decrease = v.Y_[t][pc.vision2030_sectors] *  pc.sectoral_yearly_growth_rates_V2030_[pc.vision2030_sectors]
        # ASSUME INDUSTRIAL POLICY here, i.e. that all increases in Vision 2030 targets are supplied by increasing domestic production
        if p.Endogenous_excess_import_reduction: # Here, the increased production is mirrored by a reduction in imports with an exogenous parameter
            v.IM_[t][pc.vision2030_sectors] = v.IM_[t][pc.vision2030_sectors]  -  vision_2030_sectors_import_decrease * p.import_excess_adjustment  # Reduce imports by the same amount
            v.gI_endog_[t][pc.vision2030_sectors] = v.gI_endog_[t][pc.vision2030_sectors] + pc.sectoral_yearly_growth_rates_V2030_[pc.vision2030_sectors]  * p.import_reduction_investment_effectivity_factor
    ###################################################################################################################################################################################################################################################################################################            




    # VISION 2030 induced GROWTH STEP 2.: Calculate the amount of increase in INDIRECT activity of the vision 2030 sectors through their intermediate input suppliers, and reduce part of the imports of these supplying sectors accordingly
    ###################################################################################################################################################################################################################################################################################################
    if p.Endogenous_import_reduction_cluster_growth: 
          if 1 <= t <= p.vision_2030_timing or 1 <= t <= p.extended_vision_2030_timing: 
            # Define Vision 2030 sectors columns
            v2030_cols = pc.vision2030_sectors
            
            # ITERATE over the columns as specified by v2030_cols, and increase the INTERMEDIATE INPUT FOR V2030 sectors with the exogenous V2030 growth rate
            v.IntP_V2030_proxy__[t] = v.IntP__[t].copy()
            for v2030col in v2030_cols:
                v.IntP_V2030_proxy__[t][:, v2030col] = v.IntP__[t][:, v2030col] * (1 + pc.sectoral_yearly_growth_rates_V2030_[v2030col])
            
            # Calculate the increase in intermediate input production needed by suppliers to the Vision 2030 sectors
            v2030_increase_by_supplier__ = v.IntP_V2030_proxy__[t] - v.IntP__[t]
            v2030_increase_by_supplier_relative__ = np.nan_to_num(v2030_increase_by_supplier__ / v.IntP__[t], nan=0.0, posinf=0.0, neginf=0.0)
            
            # Sum across all Vision 2030 demand sectors (columns) to get total increase per supplier sector
            # inputs provided to the Vision 2030 sectors (columns)
            v2030_increase_by_supplier_total_ = np.sum(v2030_increase_by_supplier__, axis=1)  # Shape: (86,)

            # Increase domestic production by the amount of increased intermediate input supply needs from other sectors to the vision 2030 sectors, scaled with an exogenous factor
            v.Y_[t] = v.Y_[t] + v2030_increase_by_supplier_total_ * p.cluster_growth_effectivity_factor 

            # Option: Decrease imports accordingly as production of suppliers to the Vision 2030 sectors increases
            if p.Endogenous_excess_import_reduction: # Here, the increased production is mirrored by a reduction in imports
                v.IM_[t] = v.IM_[t]  -  v2030_increase_by_supplier_total_ * p.import_excess_adjustment  * p.cluster_growth_effectivity_factor # Reduce imports by the same amount
                v.gI_endog_[t] = v.gI_endog_[t]  + pc.sectoral_yearly_growth_rates_V2030_ * p.import_reduction_investment_effectivity_factor
                      


    ###################################################################################################################################################################################################################################################################################################

        
    # IMPORT REDUCTION BASED ON EXCESS IMPORTS ONLY, MIRRORING DIRECTED INDUSTRIAL POLICY
    ###################################################################################################################################################################################################################################################################################################
    if p.Endogenous_excess_import_reduction:
        for sector in import_dependency_rank_:
            # This is a simple fall-back option of domestic production increase based on excess imports only, which mirrors a simplified version of directed industrial policy
            import_reducing_domestic_increase = np.maximum(v.excess_import_[t][sector] * p.import_excess_adjustment * (v.IM_[t][sector] - v.Y_[t][sector]),0)
            # HERE sectors are forced to grow TO REDUCE imports
            v.Y_[t][sector] = np.nan_to_num(v.Y_[t][sector] + import_reducing_domestic_increase)  # Increase domestic production by the excess import adjustment
            v.IM_[t][sector] = np.nan_to_num(v.IM_[t][sector] - import_reducing_domestic_increase)  # Reduce imports by the same amount
            v.gI_endog_[t][sector] = v.gI_endog_[t][sector] * (1 + v.excess_import_[t][sector] * p.import_excess_adjustment) * p.import_reduction_investment_effectivity_factor
    ###################################################################################################################################################################################################################################################################################################

    ###################################################################################################################################################################################################################################################################################################
    # Now calculate total Output X with the updated A matrix, and re-recalculate all other variables
    ###################################################################################################################################################################################################################################################################################################
    # Calculate total output X based on A matrix and all final demand components
    v.X_[t] = np.dot(np.linalg.inv(pc.I - v.A__[t]), v.Y_[t])
    replace_agr_X(v, pc, t)
    # Re-create Z matrix from last period to have a direct representation of IO Shares
    v.Z__[t] = v.A__[t]  @ np.diag(v.X_[t])
    # This is the same as the Z matrix
    v.IntP__[t] = v.A__[t] * v.X_[t]  # Intermediate purchases as matrix
    # Real output / physical quantities
    v.x_[t] = v.X_[t] / v.p_[t]
    # Eq (14) Intermediate sales
    v.IntS_[t] = np.dot(v.A__[t], v.X_[t])
    # Eq (15) Intermediate purchases
    v.IntP_[t] = np.sum(v.A__[t], axis=0) * v.X_[t]
    # Derive intermediate-purchase shares (a proxy for A), used when endogenizing the A matrix
    v.IntP_shares__[t] = np.divide(v.IntP__[t], v.IntP_[t], out=np.zeros_like(v.IntP__[t], dtype=float), where=v.IntP_[t] != 0)  # Shares of intermediate purchases




    ###################################################################################################################################################################################################################################################################################################
    # Proxy total output and intermediate inputs using last period's A matrix
    v.X_proxy_[t] = np.dot(np.linalg.inv(pc.I - v.A__[t-1]), v.Y_[t])
    v.IntP_proxy_[t] = np.sum(v.A__[t-1], axis=0) * v.X_proxy_[t]
    v.IntP_proxy__[t] = v.A__[t-1] * v.X_proxy_[t]
    v.IntP_shares_proxy__[t] = v.IntP_proxy__[t] / v.IntP_proxy_[t]
    v.A_proxy__[t] = v.IntP__[t] * (1/v.X_[t])  
    ###################################################################################################################################################################################################################################################################################################
    

    ###################################################################################################################################################################################################################################################################################################
    return v.Y_[t],v.IM_[t],v.X_[t],v.x_[t],v.IntS_[t],v.IntP_[t],v.gI_endog_[t],v.A__[t],v.gA__[t],v.Z__[t]
    ###################################################################################################################################################################################################################################################################################################

