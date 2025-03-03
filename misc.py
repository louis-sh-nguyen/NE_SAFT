import os
from datetime import datetime
import pandas as pd
import NET_SAFTgMie_master as NE_SAFT
import numpy as np
import re

def get_time_ID():
    time_ID = datetime.now().strftime("%y%m%d-%H%M")  # YYMMDD-HHMM
    return time_ID

def get_AAD_EQ_NE_multiT(
    T_list: list[float],
    ksw_list: list[float],
    rho20_list: list[float],
    sol: str,
    pol: str,
    MW2: float = None,
    xlxs_sheet_refno_list: list[str] = None,
    include_EQ: bool = True,
    include_NE: bool = True,
    save_data_dir: str = None,
    ) -> None:
    
    (
        _eos_mix,
        _eos_sol,
        _MW_1,
        _MW_2,
        _MW_monomer,
        _rho_2_am_dry,
        _k_sw,
    ) = NE_SAFT.get_mixture_info(sol, pol, MW2)    
    MW2 = MW2 if MW2 != None else _MW_2  # [g/mol]

    # Check T_list, ksw_list and rho20_list have the same length
    if len(T_list) != len(ksw_list) or len(T_list) != len(rho20_list):
        raise ValueError('T_list, ksw_list and rho20_list must have the same length')
    
    # Empty array to store results
    hasExpData = [None for i in range(len(T_list))]
    matched_sheets = [None for i in range(len(T_list))]
    ref_no = [None for i in range(len(T_list))]
    ref_ID = [None for i in range(len(T_list))]
    dict = {}  # dictionary of all matched sheet df
    
    # Import exp data
    for i, T in enumerate(T_list):
        try:
            # Read exp file
            path = os.path.join(os.path.dirname(__file__), 'litdata')
            refpath = path + '/references.xlsx'
            databasepath = path + '/%s-%s.xlsx' % (sol, pol)
            reffile = pd.ExcelFile(refpath, engine='openpyxl')
            datafile = pd.ExcelFile(databasepath, engine='openpyxl')
            
            # Get all sheets matching T
            # print(file.sheet_names)
            matched_sheets[i] = []
            if xlxs_sheet_refno_list == None:
                search_pattern = f'^S_{T-273}C (.*)'  # strat with S_{T-273}C ()
                for sheet in datafile.sheet_names:
                    if re.search(search_pattern, sheet):
                        matched_sheets[i].append(sheet)
            elif isinstance(xlxs_sheet_refno_list, list):  # check if a list
                for j in xlxs_sheet_refno_list:
                    search_pattern = f'^S_{T-273}C.\({j}\)'
                    for sheet in datafile.sheet_names:
                        if re.search(search_pattern, sheet):
                            matched_sheets[i].append(sheet)

            print(f'Sheets for {T-273}C: ', matched_sheets[i])

            ref_no[i] = []
            for sheet in matched_sheets[i]:
                dict[sheet] = pd.read_excel(datafile, sheet)
                dict[sheet].dropna(subset=['P [MPa]'], inplace=True)
                ref_no[i].append(sheet[sheet.find('(') + 1 : sheet.find(')')])
            print(ref_no[i])

            ref_ID[i] = []
            ref_df = pd.read_excel(reffile, 'references')
            # print(ref_df)
            for no in ref_no[i]:
                ref_ID[i].append(ref_df.loc[ref_df['# ref'] == f'[{no}]', 'refID'].item())
                # print(ref_ID[i])
            print(ref_ID[i])
        except Exception as e:
            print('')
            print('Error - importing exp data failed:')
            print(e)
        # print(len(matched_sheets[i]))
        hasExpData[i] = True if len(matched_sheets[i]) > 0 else False
    
    print('hasExpData = ', hasExpData)
    
    # Empty array to store results
    p_calc = [None for i in range(len(T_list))]
    solubility_exp = [None for i in range(len(T_list))]
    solubility_EQ = [None for i in range(len(T_list))]
    solubility_NE = [None for i in range(len(T_list))]
    AAD_EQ = [None for i in range(len(T_list))]
    AAD_NE = [None for i in range(len(T_list))]
    
    # Get pressure values
    for i, T in enumerate(T_list):
        # Empty dictionary to store results for each exp data sheet
        p_calc[i] = {}
        solubility_exp[i] = {}
        
        if include_EQ:
            solubility_EQ[i] = {}
            AAD_EQ[i] = {}
        
        if include_NE:
            solubility_NE[i] = {}
            AAD_NE[i] = {}
        
        if hasExpData[i]:            
            # Get pressues and solubility values
            for sheet in matched_sheets[i]:
                # Get pressure values
                p_calc[i][sheet] = dict[sheet]['P [MPa]'].to_numpy() * 1e6  # [Pa]
                
                # Get exp solubility values
                solubility_exp[i][sheet] = dict[sheet]['Solubility [g-sol/g-pol-am]'].to_numpy()    # [g_sol/g_pol_am]
                
                # Calculate EQ solubility and AAD%
                if include_EQ:
                    solubility_EQ[i][sheet] = [NE_SAFT.solve_solubility_EQ(T, p, sol, pol, MW2) for p in p_calc[i][sheet]]
                    # print(f'T = {T-273} C, Excel sheet = {sheet},  solubility_EQ = {solubility_EQ[i][sheet]}')
                    AAD_EQ[i][sheet] = NE_SAFT.get_fitting_AAD(solubility_exp[i][sheet], np.array(solubility_EQ[i][sheet]))
                    print(f'T = {T-273} C, Excel sheet = {sheet},  AAD% EQ: {AAD_EQ[i][sheet]*100}%')
                
                # Calculate NE solubility and AAD%
                if include_NE:
                    solubility_NE[i][sheet] = [NE_SAFT.solve_solubility_NE(T, p, sol, pol, MW2, ksw_list[i], rho20_list[i]) for p in p_calc[i][sheet]]
                    # print(f'T = {T-273} C, ref sheet = {sheet},  solubility_NE = {solubility_NE[i][sheet]}')
                    AAD_NE[i][sheet] = NE_SAFT.get_fitting_AAD(solubility_exp[i][sheet], np.array(solubility_NE[i][sheet]))
                    print(f'T = {T-273} C, ref sheet = {sheet},  AAD% NE: {AAD_NE[i][sheet]*100}%')
                
        else:
            print(f'No experimental data for T = {T-273} C')
            
    # Store results in DataFrame
    df_results = pd.DataFrame()
    for i, T in enumerate(T_list):
        if hasExpData[i]:
            for j, sheet in enumerate(matched_sheets[i]):
                result = {
                    'T / °C': [T-273] * len(p_calc[i][sheet]),
                    'Excel sheet': [sheet] * len(p_calc[i][sheet]),
                    'ref_ID': [ref_ID[i][j]] * len(p_calc[i][sheet]),
                    'p / MPa': p_calc[i][sheet]*1e-6,
                    'solubility_exp / g_sol g_pol^-1': solubility_exp[i][sheet],
                    'solubility_NE / g_sol g_pol^-1': solubility_NE[i][sheet] if include_NE else [np.nan] * len(p_calc[i][sheet]),
                    'solubility_EQ / g_sol g_pol^-1': solubility_EQ[i][sheet] if include_EQ else [np.nan] * len(p_calc[i][sheet]),
                }
                df_result = pd.DataFrame(result)
                df_results = pd.concat([df_results, df_result], ignore_index=True)
    
    print(df_results)
    
    # Store AAD results in DataFrame
    aad_results = []
    for i, T in enumerate(T_list):
        if hasExpData[i]:
            for j, sheet in enumerate(matched_sheets[i]):
                result = {
                    'T / °C': T - 273,
                    'Excel sheet sheet': sheet,
                    'ref_ID': ref_ID[i][j],
                    'AAD% EQ': AAD_EQ[i][sheet] * 100 if include_EQ else np.nan,
                    'AAD% NE': AAD_EQ[i][sheet] * 100 if include_NE else np.nan,
                }
                aad_results.append(result)
    
    df_aad_results = pd.DataFrame(aad_results)
    print(df_aad_results)

    # Save data
    if save_data_dir != None:
        with pd.ExcelWriter(save_data_dir, engine='openpyxl') as writer:
            df_results.to_excel(writer, sheet_name='Solubility results', index=False)
            df_aad_results.to_excel(writer, sheet_name='AAD% results', index=False)
        
        print(f'Data saved to {save_data_dir}')

if __name__ == '__main__':
    rho20_35C = {}
    rho20_51C = {}
    rho20_81C = {}
    ksw_35C = {}
    ksw_51C = {}
    ksw_81C = {}
    ksw_35C['PS'] = {}
    ksw_51C['PS'] = {}
    ksw_81C['PS'] = {}
    ksw_35C['PMMA'] = {}
    ksw_51C['PMMA'] = {}
    ksw_81C['PMMA'] = {}
    
    # From PVT
    rho20_35C['PS'] = 1.042
    rho20_51C['PS'] = 1.037
    rho20_81C['PS'] = 1.030
    rho20_35C['PMMA'] = 1.178
    rho20_51C['PMMA'] = 1.174
    rho20_81C['PMMA'] = 1.165
    
    # Predicted with default parameters
    ksw_35C['PS']['default'] = 0.00914
    ksw_51C['PS']['default'] = 0.00728
    ksw_81C['PS']['default'] = 0.00517
    ksw_35C['PMMA']['default'] = 0.01705
    ksw_51C['PMMA']['default'] = 0.01387
    ksw_81C['PMMA']['default'] = 0.01025
    
    # Predicted with fitted parameters
    ksw_35C['PS']['fitted'] = 0.00829
    ksw_51C['PS']['fitted'] = 0.00665
    ksw_81C['PS']['fitted'] = 0.00474
    ksw_35C['PMMA']['fitted'] = 0.01805
    ksw_51C['PMMA']['fitted'] = 0.01356
    ksw_81C['PMMA']['fitted'] = 0.00881
    
    T_list = [273.15, 283.15, 293.15, 303.15]
    eos_parameter_type = 'fitted'
    polymer = 'PMMA'
    # get_AAD_EQ_NE_multiT(T_list, ksw_list, rho20_list, sol, pol, save_data_dir='test.xlsx')
    get_AAD_EQ_NE_multiT([100+273], 
                        # rho20_list=[rho20_35C[polymer]],
                        # ksw_list=[ksw_35C[polymer][eos_parameter_type]],
                        rho20_list=[0],
                        ksw_list=[0],
                        include_EQ=True,
                        include_NE=False,
                         sol='CO2',
                         pol=polymer,
                        #  xlxs_sheet_refno_list=['8'],
                        #  save_data_dir='test.csv'
                         )