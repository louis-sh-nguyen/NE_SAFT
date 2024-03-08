import math
import os
import time
from datetime import datetime
import addcopyfighandler
import matplotlib.pyplot as plt
import pandas as pd
import NET_SAFTgMie_master as NE_SAFT
import numpy as np
import re
import shutil

rho20_x0_dict = {"PS": np.linspace(0.90, 1.30, 30), "PMMA": np.linspace(0.90, 1.30, 30)}
ksw_x0_dict = {"PS": np.linspace(0.0, 0.03, 30), "PMMA": np.linspace(0.0, 0.03, 30)}


def fitrho20_fitksw_predictksw_plotpermisotherm(solute: str, polymer: str, temp_list: list[float]):
    # Using time as ID for output files
    start_time = time.time()
    now = datetime.now()  # current time
    time_ID = now.strftime("%y%m%d_%H%M")  # YYMMDD_HHMM

    # current directory
    src_dir = os.path.dirname(__file__)
    # src_dir = os.path.abspath(src_dir)  # absolute path
    src_dir = r"\\?\%s" % src_dir  # extended path (for very long path length)

    # Create new directory to store results
    result_folder_dir = (
        src_dir + f"\\Anals\\rho20f_kswf_kswp_plotpermisotherm\\{solute}-{polymer}\\{time_ID}"
    )
    os.mkdir(result_folder_dir)
    print("Folder created: " + result_folder_dir)

    # Copy .py code used to result folder
    src_py_dir = src_dir + "\\Anals_NETSAFTgMie.py"
    dtn_py_dir = result_folder_dir + f"\\CodeUsed__Anals_NETSAFTgMie__{time_ID}.py"
    shutil.copy(src_py_dir, dtn_py_dir)
    print(f".py file copied: {dtn_py_dir}")

    src_NETpy_dir = src_dir + "\\NET_SAFTgMie_master.py"
    dtn_NETpy_dir = result_folder_dir + f"\\CodeUsed__NET_SAFTgMie__{time_ID}.py"
    shutil.copy(src_NETpy_dir, dtn_NETpy_dir)
    print(f".py file copied: {dtn_NETpy_dir}")

    # Create empty df to store future results, WITH index.
    # Then export to empty .csv file.
    df = pd.DataFrame(
        columns=[
            "sol",
            "pol",
            "T (C)",
            "ref no",
            "S xlsx sheet",
            "P xlsx sheet",
            "fitted rho20 (g/cm^3)",
            "AAD%_rho20_fit (%)",
            "fitted ksw (MPa^-1)",
            "AAD%_ksw_fit (%)",
            "predicted ksw (MPa^-1)",
            "AAD%_ksw_parity (%)",
        ]
    )
    csv_name = f"{solute}-{polymer}_rho20f-kswf-kswf-plotpermisotherm_Report_{time_ID}.csv"
    csv_dir = result_folder_dir + f"\\{csv_name}"
    df.to_csv(csv_dir)

    # get matching sheets
    path = os.path.join(os.path.dirname(__file__), "litdata")
    databasepath = path + "\\%s-%s.xlsx" % (solute, polymer)
    datafile = pd.ExcelFile(databasepath, engine="openpyxl")

    for Temp in temp_list:  # loop level 1
        # search for xlsx sheet with matching temp
        S_search_pattern = f"^S_{Temp-273}C (.*)"  # solubility start with {T}C
        P_search_pattern = f"^P_{Temp-273}C (.*)"  # permeability start with {T}C
        S_matched_sheets = []
        P_matched_sheets = []
        for sheet in datafile.sheet_names:
            if re.search(S_search_pattern, sheet):  # has both S and P data
                S_matched_sheets.append(sheet)
            if re.search(P_search_pattern, sheet):  # has both S and P data
                P_matched_sheets.append(sheet)

        # check if data of current Temp is available before continuing
        if len(S_matched_sheets) == 0 and len(P_search_pattern) == 0:
            continue

        # extract ref number from matched xlxs sheet
        S_refno_all = []
        P_refno_all = []
        for sheet in S_matched_sheets:
            S_refno_all.append(sheet[sheet.find("(") + 1 : sheet.find(")")])
        for sheet in P_matched_sheets:
            P_refno_all.append(sheet[sheet.find("(") + 1 : sheet.find(")")])
        refno_all = list(set(S_refno_all).intersection(P_refno_all))  # common between S and P sheet
        print(refno_all)

        # perform fitting in each sheet
        for refno in refno_all:  # loop level 2
            try:
                # * Fit rho20
                figname1 = f"{solute}-{polymer}_{Temp-273}C({refno})_fitrho20_{time_ID}.png"
                savedir1 = result_folder_dir + f"\\{figname1}"
                try:
                    rho20_f, rho20_AADpc_f = NE_SAFT.fit_rho20_NE(
                        T=Temp,
                        sol=solute,
                        pol=polymer,
                        xlxs_sheet_refno=refno,
                        rho20_x0_list=rho20_x0_dict[polymer],
                        display_plot=False,
                        save_plot_dir=savedir1,
                    )
                except Exception as e:
                    print("Error: ", e)
                    rho20_f = None
                    rho20_AADpc_f = None

                # * Fit ksw from obtained rho20
                figname2 = f"{solute}-{polymer}_{Temp-273}C({refno})_fitksw_{time_ID}.png"
                savedir2 = result_folder_dir + f"\\{figname2}"
                try:
                    ksw_f, ksw_AADpc_f = NE_SAFT.fit_ksw_NE(
                        T=Temp,
                        sol=solute,
                        pol=polymer,
                        xlxs_sheet_refno=refno,
                        rho20=rho20_f,
                        glassy_filter=True,
                        ksw_x0_list=ksw_x0_dict[polymer],
                        display_plot=False,
                        save_plot_dir=savedir2,
                    )
                except Exception as e:
                    print("Error: ", e)
                    ksw_f = None
                    ksw_AADpc_f = None

                # * Predict ksw
                try:
                    ksw_p = NE_SAFT.predict_ksw_NE(T=Temp, sol=solute, pol=polymer, rho20=rho20_f)
                except Exception as e:
                    print("Error: ", e)
                    ksw_p = None
                
                try:
                    ksw_parity_AADpc = (
                        NE_SAFT.get_fitting_AAD(np.array([ksw_f]).tolist(), np.array([ksw_p]).tolist()) * 100
                    )
                except Exception as e:
                    print("Error: ", e)
                    ksw_parity_AADpc = None
                    
                # * Plot permeability isotherm for ksw_f and ksw_p
                figname3 = f"{solute}-{polymer}_{Temp-273}C({refno})_permisotherm_{time_ID}.png"
                savedir3 = result_folder_dir + f"\\{figname3}"
                try:
                    NE_SAFT.plot_permeability_isotherm_NE(
                        T=Temp,
                        p_l=1,
                        p_u=50 * 101325,
                        no_of_points=40,
                        sol=solute,
                        pol=polymer,
                        ksw_list=[ksw_f, ksw_p],
                        rho20=rho20_f,
                        xlxs_sheet_refno_list=[refno],
                        display_plot=False,
                        save_plot_dir=savedir3,
                    )
                except Exception as e:
                    print("Error: ", e)

            except Exception as e:
                print(f"Error: Unsuccessful for {solute}-{polymer}_{Temp-273}C ({refno})")
                print(e)
            else:
                # add new entry to df
                S_sheet = f"S_{Temp-273}C ({refno})"
                P_sheet = f"P_{Temp-273}C ({refno})"
                ref = f"{refno}"
                df1 = pd.DataFrame(
                    {
                        "sol": [solute],
                        "pol": [polymer],
                        "T (C)": [Temp - 273],
                        "ref no": [ref],
                        "S xlsx sheet": [S_sheet],
                        "P xlsx sheet": [P_sheet],
                        "fitted rho20 (g/cm^3)": [rho20_f],
                        "AAD%_rho20_fit (%)": [rho20_AADpc_f],
                        "fitted ksw (MPa^-1)": [ksw_f],
                        "AAD%_ksw_fit (%)": [ksw_AADpc_f],
                        "predicted ksw (MPa^-1)": [ksw_p],
                        "AAD%_ksw_parity (%)": [ksw_parity_AADpc],
                    }
                )
                # append to current csv file
                df1.to_csv(
                    csv_dir,
                    mode="a",
                    index=True,
                    header=False,
                )
                print("")
                print(f"Data {solute}-{polymer}_{Temp-273}C ({ref}) appended to {csv_name}")
                print("")

    # Export dataframe as .csv
    print(f".csv result export finished: {csv_dir}")

    print("\n--- Run time %s-%s:\t%.0f seconds ---\n" % (solute, polymer, time.time() - start_time))

def rho20PVT_fitksw_predictksw_plotpermisotherm(solute: str, polymer: str, temp_list: list[float]):
    # Using time as ID for output files
    start_time = time.time()
    now = datetime.now()  # current time
    time_ID = now.strftime("%y%m%d_%H%M")  # YYMMDD_HHMM

    # current directory
    src_dir = os.path.dirname(__file__)
    # src_dir = os.path.abspath(src_dir)  # absolute path
    src_dir = r"\\?\%s" % src_dir  # extended path (for very long path length)

    # Create new directory to store results
    result_folder_dir = (
        src_dir + f"\\Anals\\rho20PVT_kswf_kswp_plotpermisotherm\\{solute}-{polymer}\\{time_ID}"
    )
    os.mkdir(result_folder_dir)
    print("Folder created: " + result_folder_dir)

    # Copy .py code used to result folder
    src_py_dir = src_dir + "\\Anals_NETSAFTgMie.py"
    dtn_py_dir = result_folder_dir + f"\\CodeUsed__Anals_NETSAFTgMie__{time_ID}.py"
    shutil.copy(src_py_dir, dtn_py_dir)
    print(f".py file copied: {dtn_py_dir}")

    src_NETpy_dir = src_dir + "\\NET_SAFTgMie_master.py"
    dtn_NETpy_dir = result_folder_dir + f"\\CodeUsed__NET_SAFTgMie__{time_ID}.py"
    shutil.copy(src_NETpy_dir, dtn_NETpy_dir)
    print(f".py file copied: {dtn_NETpy_dir}")

    # Create empty df to store future results, WITH index.
    # Then export to empty .csv file.
    df = pd.DataFrame(
        columns=[
            "sol",
            "pol",
            "T (C)",
            "ref no",
            "S xlsx sheet",
            "P xlsx sheet",
            "fitted rho20 (g/cm^3)",
            "AAD%_rho20_fit (%)",
            "rho20 PVT (g/cm^3)",        
            "AAD%_rho20_parity (%)",    
            "fitted ksw (MPa^-1)",
            "AAD%_ksw_fit (%)",
            "predicted ksw (MPa^-1)",
            "AAD%_ksw_parity (%)",
        ]
    )
    csv_name = f"{solute}-{polymer}_rho20PVT-rho20f-kswf-kswp-plotpermisotherm_Report_{time_ID}.csv"
    csv_dir = result_folder_dir + f"\\{csv_name}"
    df.to_csv(csv_dir)

    # get matching sheets
    path = os.path.join(os.path.dirname(__file__), "litdata")
    databasepath = path + "\\%s-%s.xlsx" % (solute, polymer)
    datafile = pd.ExcelFile(databasepath, engine="openpyxl")

    for Temp in temp_list:  # loop level 1
        # search for xlsx sheet with matching temp
        S_search_pattern = f"^S_{Temp-273}C (.*)"  # solubility start with {T}C
        P_search_pattern = f"^P_{Temp-273}C (.*)"  # permeability start with {T}C
        S_matched_sheets = []
        P_matched_sheets = []
        for sheet in datafile.sheet_names:
            if re.search(S_search_pattern, sheet):  # has both S and P data
                S_matched_sheets.append(sheet)
            if re.search(P_search_pattern, sheet):  # has both S and P data
                P_matched_sheets.append(sheet)

        # check if data of current Temp is available before continuing
        if len(S_matched_sheets) == 0 and len(P_search_pattern) == 0:
            continue

        # extract ref number from matched xlxs sheet
        S_refno_all = []
        P_refno_all = []
        for sheet in S_matched_sheets:
            S_refno_all.append(sheet[sheet.find("(") + 1 : sheet.find(")")])
        for sheet in P_matched_sheets:
            P_refno_all.append(sheet[sheet.find("(") + 1 : sheet.find(")")])
        refno_all = list(set(S_refno_all).intersection(P_refno_all))  # common between S and P sheet
        print(refno_all)

        # perform fitting in each sheet
        for refno in refno_all:  # loop level 2
            try:
                # * Fit rho20
                figname1 = f"{solute}-{polymer}_{Temp-273}C({refno})_fitrho20_{time_ID}.png"
                savedir1 = result_folder_dir + f"\\{figname1}"
                try:
                    rho20_f, rho20_AADpc_f = NE_SAFT.fit_rho20_NE(
                        T=Temp,
                        sol=solute,
                        pol=polymer,
                        xlxs_sheet_refno=refno,
                        rho20_x0_list=rho20_x0_dict[polymer],
                        display_plot=False,
                        save_plot_dir=savedir1,
                    )
                except Exception as e:
                    print("Error: ", e)
                    rho20_f = None
                    rho20_AADpc_f = None
                    
                # * Predict rho20 from PVT
                try:
                    rho20_PVT = 1 / NE_SAFT.get_V20_multiTait(T=Temp, p=0, pol=polymer)
                except Exception as e:
                    print("Error: ", e)
                    rho20_PVT = None

                try:
                    rho20_parity_AADpc = (
                        NE_SAFT.get_fitting_AAD(np.array([rho20_f]).tolist(), np.array([rho20_PVT]).tolist()) * 100
                    )
                except Exception as e:
                    print("Error: ", e)
                    rho20_parity_AADpc = None
                    
                # * Fit ksw from rho20_PVT
                figname2 = f"{solute}-{polymer}_{Temp-273}C({refno})_fitksw_{time_ID}.png"
                savedir2 = result_folder_dir + f"\\{figname2}"
                try:
                    ksw_f, ksw_AADpc_f = NE_SAFT.fit_ksw_NE(
                        T=Temp,
                        sol=solute,
                        pol=polymer,
                        xlxs_sheet_refno=refno,
                        rho20=rho20_PVT,
                        glassy_filter=True,
                        ksw_x0_list=ksw_x0_dict[polymer],
                        display_plot=False,
                        save_plot_dir=savedir2,
                    )
                except Exception as e:
                    print("Error: ", e)
                    ksw_f = None
                    ksw_AADpc_f = None

                # * Predict ksw
                try:
                    ksw_p = NE_SAFT.predict_ksw_NE(T=Temp, sol=solute, pol=polymer, rho20=rho20_PVT)
                except Exception as e:
                    print("Error: ", e)
                    ksw_p = None
                
                try:
                    ksw_parity_AADpc = (
                        NE_SAFT.get_fitting_AAD(np.array([ksw_f]).tolist(), np.array([ksw_p]).tolist()) * 100
                    )
                except Exception as e:
                    print("Error: ", e)
                    ksw_parity_AADpc = None
                    
                # * Plot permeability isotherm for ksw_f and ksw_p
                figname3 = f"{solute}-{polymer}_{Temp-273}C({refno})_permisotherm_{time_ID}.png"
                savedir3 = result_folder_dir + f"\\{figname3}"
                try:
                    NE_SAFT.plot_permeability_isotherm_NE(
                        T=Temp,
                        p_l=1,
                        p_u=50 * 101325,
                        no_of_points=40,
                        sol=solute,
                        pol=polymer,
                        ksw_list=[ksw_f, ksw_p],
                        rho20=rho20_PVT,
                        xlxs_sheet_refno_list=[refno],
                        display_plot=False,
                        save_plot_dir=savedir3,
                    )
                except Exception as e:
                    print("Error: ", e)

            except Exception as e:
                print(f"Error: Unsuccessful for {solute}-{polymer}_{Temp-273}C ({refno})")
                print(e)
            else:
                # add new entry to df
                S_sheet = f"S_{Temp-273}C ({refno})"
                P_sheet = f"P_{Temp-273}C ({refno})"
                ref = f"{refno}"
                df1 = pd.DataFrame(
                    {
                        "sol": [solute],
                        "pol": [polymer],
                        "T (C)": [Temp - 273],
                        "ref no": [ref],
                        "S xlsx sheet": [S_sheet],
                        "P xlsx sheet": [P_sheet],
                        "fitted rho20 (g/cm^3)": [rho20_f],
                        "AAD%_rho20_fit (%)": [rho20_AADpc_f],
                        "rho20 PVT (g/cm^3)": [rho20_PVT],        
                        "AAD%_rho20_parity (%)": [rho20_parity_AADpc],    
                        "fitted ksw (MPa^-1)": [ksw_f],
                        "AAD%_ksw_fit (%)": [ksw_AADpc_f],
                        "predicted ksw (MPa^-1)": [ksw_p],
                        "AAD%_ksw_parity (%)": [ksw_parity_AADpc],
                    }
                )
                # append to current csv file
                df1.to_csv(
                    csv_dir,
                    mode="a",
                    index=True,
                    header=False,
                )
                print("")
                print(f"Data {solute}-{polymer}_{Temp-273}C ({ref}) appended to {csv_name}")
                print("")

    # Export dataframe as .csv
    print(f".csv result export finished: {csv_dir}")

    print("\n--- Run time %s-%s:\t%.0f seconds ---\n" % (solute, polymer, time.time() - start_time))


def fitrho20_fitksw_predictksw_pg(solute: str, polymer: str, temp_list: list[float]):
    # Using time as ID for output files
    start_time = time.time()
    now = datetime.now()  # current time
    time_ID = now.strftime("%y%m%d_%H%M")  # YYMMDD_HHMM

    # current directory
    src_dir = os.path.dirname(__file__)
    src_dir = r"\\?\%s" % src_dir  # extended path (for very long path length)

    # Create new directory to store results
    result_folder_dir = src_dir + f"\\Anals\\rho20f_kswf_kswp_pg\\{solute}-{polymer}\\{time_ID}"
    os.mkdir(result_folder_dir)
    print("Folder created: " + result_folder_dir)

    # Copy .py files used to result folder
    src_analpy_dir = src_dir + "\\Anals_NETSAFTgMie.py"
    dtn_analpy_dir = result_folder_dir + f"\\CodeUsed__Anals_NETSAFTgMie__{time_ID}.py"
    shutil.copy(src_analpy_dir, dtn_analpy_dir)
    print(f".py file copied: {dtn_analpy_dir}")

    src_NETpy_dir = src_dir + "\\NET_SAFTgMie_master.py"
    dtn_NETpy_dir = result_folder_dir + f"\\CodeUsed__NET_SAFTgMie__{time_ID}.py"
    shutil.copy(src_NETpy_dir, dtn_NETpy_dir)
    print(f".py file copied: {dtn_NETpy_dir}")

    # Create empty df to store future results, WITH index.
    # Then eport to empty .csv file.
    df = pd.DataFrame(
        columns=[
            "sol",
            "pol",
            "T (C)",
            "ref no",
            "xlsx sheet",
            "fitted rho20 (g/cm^3)",
            "AAD%_rho20_fit (%)",
            "fitted ksw (MPa^-1)",
            "AAD%_ksw_fit (%)",
            "predicted ksw (MPa^-1)",
            "AAD%_ksw_parity (%)",
            "pg (Pa)",
            "Sg (g/g)",
        ]
    )
    csv_name = f"{solute}-{polymer}_rho20f-kswf-kswp-pg_Report_{time_ID}.csv"
    csv_dir = result_folder_dir + f"\\{csv_name}"
    df.to_csv(csv_dir)

    # get matching sheets
    path = os.path.join(os.path.dirname(__file__), "litdata")
    databasepath = path + "\\%s-%s.xlsx" % (solute, polymer)
    datafile = pd.ExcelFile(databasepath, engine="openpyxl")

    for Temp in temp_list:  # loop level 1
        # search for xlsx sheet with matching temp
        search_pattern = f"^S_{Temp-273}C (.*)"  # strat with {T}C
        matched_sheets = []
        for sheet in datafile.sheet_names:
            if re.search(search_pattern, sheet):
                matched_sheets.append(sheet)
        print("Sheets: ", matched_sheets)

        # check if data available before continuing
        if len(matched_sheets) == 0:
            continue

        # extract ref number from matched xlxs sheet
        refno_all = []
        for sheet in matched_sheets:
            refno_all.append(sheet[sheet.find("(") + 1 : sheet.find(")")])
        print(refno_all)

        # perform fitting in each sheet
        for refno in refno_all:  # loop level 2
            try:
                # * Fit rho20
                figname1 = f"{solute}-{polymer}_{Temp-273}C({refno})_fitrho20_{time_ID}.png"
                savedir1 = result_folder_dir + f"\\{figname1}"
                try:
                    rho20_f, rho20_AADpc_f = NE_SAFT.fit_rho20_NE(
                        T=Temp,
                        sol=solute,
                        pol=polymer,
                        xlxs_sheet_refno=refno,
                        rho20_x0_list=rho20_x0_dict[polymer],
                        display_plot=False,
                        save_plot_dir=savedir1,
                    )
                except Exception as e:
                    print("Error: ", e)
                    rho20_f = None
                    rho20_AADpc_f = None

                # * Fit ksw from fitted rho20
                figname2 = f"{solute}-{polymer}_{Temp-273}C({refno})_fitksw_{time_ID}.png"
                savedir2 = result_folder_dir + f"\\{figname2}"
                try:
                    ksw_f, ksw_AADpc_f = NE_SAFT.fit_ksw_NE(
                        T=Temp,
                        sol=solute,
                        pol=polymer,
                        xlxs_sheet_refno=refno,
                        rho20=rho20_f,
                        glassy_filter=True,
                        ksw_x0_list=ksw_x0_dict[polymer],
                        display_plot=False,
                        save_plot_dir=savedir2,
                    )
                except Exception as e:
                    print("Error: ", e)
                    ksw_f = None
                    ksw_AADpc_f = None

                # * Predict ksw and plot isotherm to compare with fitted ksw
                figname3 = f"{solute}-{polymer}_{Temp-273}C({refno})_predictksw_{time_ID}.png"
                savedir3 = result_folder_dir + f"\\{figname3}"
                try:
                    ksw_p = NE_SAFT.predict_ksw_NE(T=Temp, sol=solute, pol=polymer, rho20=rho20_f)
                except Exception as e:
                    print("Error: ", e)
                    ksw_p = None

                try:
                    ksw_parity_AADpc = (
                        NE_SAFT.get_fitting_AAD(np.array([ksw_f]).tolist(), np.array([ksw_p]).tolist()) * 100
                    )
                except Exception as e:
                    print("Error: ", e)
                    ksw_parity_AADpc = None

                try:
                    NE_SAFT.plot_isotherm_EQvNE(
                        p_l=1,
                        p_u=20e6,
                        no_of_points=40,
                        T=Temp,
                        sol=solute,
                        pol=polymer,
                        ksw_list=[ksw_f, ksw_p, 0.0],
                        rho20=rho20_f,
                        xlxs_sheet_refno_list=[refno],
                        display_plot=False,
                        save_plot_dir=savedir3,
                    )
                except Exception as e:
                    print("Error: ", e)
                    pass

                # * Plot V2 isotherm of EQ and NE
                figname4 = f"{solute}-{polymer}_{Temp-273}C({refno})_V2isotherm_{time_ID}.png"
                savedir4 = result_folder_dir + f"\\{figname4}"
                try:
                    pg, Sg = NE_SAFT.plot_V_isotherm_EQvNEvCombined(
                        p_l=1,
                        p_u=20e6,
                        no_of_points=40,
                        T=Temp,
                        sol=solute,
                        pol=polymer,
                        ksw_list=[ksw_f],
                        rho20=rho20_f,
                        display_plot=False,
                        save_plot_dir=savedir4,
                    )
                    pg = pg[0]
                    Sg = Sg[0]

                except Exception as e:
                    print("Error: ", e)
                    pg = None
                    Sg = None

                # * Plot EQ isotherm
                figname5 = f"{solute}-{polymer}_{Temp-273}C({refno})_isothermEQ_{time_ID}.png"
                savedir5 = result_folder_dir + f"\\{figname5}"
                try:
                    NE_SAFT.plot_isotherm_EQ(
                        p_l=1,
                        p_u=40e6,
                        no_of_points=40,
                        T_list=[Temp],
                        sol=solute,
                        pol=polymer,
                        xlxs_sheet_refno_list=[refno],
                        display_plot=False,
                        save_plot_dir=savedir5,
                    )
                except Exception as e:
                    print("Error: ", e)
            except:
                print(f"Error: unsuccessful for {solute}-{polymer}_{Temp-273}C ({refno})")
            else:
                # add new entry to df
                sheet = f"S_{Temp-273}C ({refno})"
                ref = f"{refno}"
                df1 = pd.DataFrame(
                    {
                        "sol": [solute],
                        "pol": [polymer],
                        "T (C)": [Temp - 273],
                        "ref no": [ref],
                        "xlsx sheet": [sheet],
                        "fitted rho20 (g/cm^3)": [rho20_f],
                        "AAD%_rho20_fit (%)": [rho20_AADpc_f],
                        "fitted ksw (MPa^-1)": [ksw_f],
                        "AAD%_ksw_fit (%)": [ksw_AADpc_f],
                        "predicted ksw (MPa^-1)": [ksw_p],
                        "AAD%_ksw_parity (%)": [ksw_parity_AADpc],
                        "pg (Pa)": [pg],
                        "Sg (g/g)": [Sg],
                    }
                )
                # append to current csv file
                df1.to_csv(
                    csv_dir,
                    mode="a",
                    index=True,
                    header=False,
                )
                print("")
                print(f"Data {solute}-{polymer}_{Temp-273}C ({ref}) appended to {csv_name}")
                print("")

    # Export dataframe as .csv
    print(f".csv result export finished: {csv_dir}")

    print("\n--- Run time %s-%s:\t%.0f seconds ---\n" % (solute, polymer, time.time() - start_time))


def rho20PVT_fitrho20_fitksw_predictksw_pg(solute: str, polymer: str, temp_list: list[float]):
    # Using time as ID for output files
    start_time = time.time()
    now = datetime.now()  # current time
    time_ID = now.strftime("%y%m%d_%H%M")  # YYMMDD_HHMM

    # current directory
    src_dir = os.path.dirname(__file__)
    src_dir = r"\\?\%s" % src_dir  # extended path (for very long path length)

    # Create new directory to store results
    result_folder_dir = src_dir + f"\\Anals\\rho20PVT_rho20f_kswf_kswp_pg\\{solute}-{polymer}\\{time_ID}"
    os.mkdir(result_folder_dir)
    print("Folder created: " + result_folder_dir)

    # Copy .py files used to result folder
    src_analpy_dir = src_dir + "\\Anals_NETSAFTgMie.py"
    dtn_analpy_dir = result_folder_dir + f"\\CodeUsed__Anals_NETSAFTgMie__{time_ID}.py"
    shutil.copy(src_analpy_dir, dtn_analpy_dir)
    print(f".py file copied: {dtn_analpy_dir}")

    src_NETpy_dir = src_dir + "\\NET_SAFTgMie_master.py"
    dtn_NETpy_dir = result_folder_dir + f"\\CodeUsed__NET_SAFTgMie__{time_ID}.py"
    shutil.copy(src_NETpy_dir, dtn_NETpy_dir)
    print(f".py file copied: {dtn_NETpy_dir}")

    # Create empty df to store future results, WITH index.
    # Then eport to empty .csv file.
    df = pd.DataFrame(
        columns=[
            "sol",
            "pol",
            "T (C)",
            "ref no",
            "xlsx sheet",
            "fitted rho20 (g/cm^3)",
            "AAD%_rho20_fit (%)",
            "rho20 PVT (g/cm^3)",
            "AAD%_rho20_parity (%)",
            "fitted ksw (MPa^-1)",
            "AAD%_ksw_fit (%)",
            "predicted ksw (MPa^-1)",
            "AAD%_ksw_parity (%)",
            "pg (Pa)",
            "Sg (g/g)",
        ]
    )
    csv_name = f"{solute}-{polymer}_rho20PVT_rho20f-kswf-kswp-pg_Report_{time_ID}.csv"
    csv_dir = result_folder_dir + f"\\{csv_name}"
    df.to_csv(csv_dir)

    # get matching sheets
    path = os.path.join(os.path.dirname(__file__), "litdata")
    databasepath = path + "\\%s-%s.xlsx" % (solute, polymer)
    datafile = pd.ExcelFile(databasepath, engine="openpyxl")

    for Temp in temp_list:  # loop level 1
        # search for xlsx sheet with matching temp
        search_pattern = f"^S_{Temp-273}C (.*)"  # strat with {T}C
        matched_sheets = []
        for sheet in datafile.sheet_names:
            if re.search(search_pattern, sheet):
                matched_sheets.append(sheet)
        print("Sheets: ", matched_sheets)

        # check if data available before continuing
        if len(matched_sheets) == 0:
            continue

        # extract ref number from matched xlxs sheet
        refno_all = []
        for sheet in matched_sheets:
            refno_all.append(sheet[sheet.find("(") + 1 : sheet.find(")")])
        print(refno_all)

        # perform fitting in each sheet
        for refno in refno_all:  # loop level 2
            try:
                # * Fit rho20
                figname1 = f"{solute}-{polymer}_{Temp-273}C({refno})_fitrho20_{time_ID}.png"
                savedir1 = result_folder_dir + f"\\{figname1}"
                try:
                    rho20_f, rho20_AADpc_f = NE_SAFT.fit_rho20_NE(
                        T=Temp,
                        sol=solute,
                        pol=polymer,
                        xlxs_sheet_refno=refno,
                        rho20_x0_list=rho20_x0_dict[polymer],
                        display_plot=False,
                        save_plot_dir=savedir1,
                    )
                except Exception as e:
                    print("Error: ", e)
                    rho20_f = None
                    rho20_AADpc_f = None

                # * Predict rho20 from PVT
                try:
                    rho20_PVT = 1 / NE_SAFT.get_V20_multiTait(T=Temp, p=0, pol=polymer)
                except Exception as e:
                    print("Error: ", e)
                    rho20_PVT = None

                try:
                    rho20_parity_AADpc = (
                        NE_SAFT.get_fitting_AAD(np.array([rho20_f]).tolist(), np.array([rho20_PVT]).tolist()) * 100
                    )
                except Exception as e:
                    print("Error: ", e)
                    rho20_parity_AADpc = None

                # * Fit ksw from rho20_PVT
                figname2 = f"{solute}-{polymer}_{Temp-273}C({refno})_fitksw_{time_ID}.png"
                savedir2 = result_folder_dir + f"\\{figname2}"
                try:
                    ksw_f, ksw_AADpc_f = NE_SAFT.fit_ksw_NE(
                        T=Temp,
                        sol=solute,
                        pol=polymer,
                        xlxs_sheet_refno=refno,
                        rho20=rho20_PVT,
                        glassy_filter=True,
                        ksw_x0_list=ksw_x0_dict[polymer],
                        display_plot=False,
                        save_plot_dir=savedir2,
                    )
                except Exception as e:
                    print("Error: ", e)
                    ksw_f = None
                    ksw_AADpc_f = None

                # * Predict ksw and plot isotherm to compare with fitted ksw
                figname3 = f"{solute}-{polymer}_{Temp-273}C({refno})_predictksw_{time_ID}.png"
                savedir3 = result_folder_dir + f"\\{figname3}"
                try:
                    ksw_p = NE_SAFT.predict_ksw_NE(T=Temp, sol=solute, pol=polymer, rho20=rho20_PVT)
                except Exception as e:
                    print("Error: ", e)
                    ksw_p = None

                try:
                    ksw_parity_AADpc = (
                        NE_SAFT.get_fitting_AAD(np.array([ksw_f]).tolist(), np.array([ksw_p]).tolist()) * 100
                    )
                except Exception as e:
                    print("Error: ", e)
                    ksw_parity_AADpc = None

                try:
                    NE_SAFT.plot_isotherm_EQvNE(
                        p_l=1,
                        p_u=20e6,
                        no_of_points=40,
                        T=Temp,
                        sol=solute,
                        pol=polymer,
                        ksw_list=[ksw_f, ksw_p, 0.0],
                        rho20=rho20_PVT,
                        xlxs_sheet_refno_list=[refno],
                        display_plot=False,
                        save_plot_dir=savedir3,
                    )
                except Exception as e:
                    print("Error: ", e)
                    pass

                # * Plot V2 isotherm of EQ and NE
                figname4 = f"{solute}-{polymer}_{Temp-273}C({refno})_V2isotherm_{time_ID}.png"
                savedir4 = result_folder_dir + f"\\{figname4}"
                try:
                    pg, Sg = NE_SAFT.plot_V_isotherm_EQvNEvCombined(
                        p_l=1,
                        p_u=20e6,
                        no_of_points=40,
                        T=Temp,
                        sol=solute,
                        pol=polymer,
                        ksw_list=[ksw_f],
                        rho20=rho20_PVT,
                        display_plot=False,
                        save_plot_dir=savedir4,
                    )
                    pg = pg[0]
                    Sg = Sg[0]

                except Exception as e:
                    print("Error: ", e)
                    pg = None
                    Sg = None

                # * Plot EQ isotherm
                figname5 = f"{solute}-{polymer}_{Temp-273}C({refno})_isothermEQ_{time_ID}.png"
                savedir5 = result_folder_dir + f"\\{figname5}"
                try:
                    NE_SAFT.plot_isotherm_EQ(
                        p_l=1,
                        p_u=40e6,
                        no_of_points=40,
                        T_list=[Temp],
                        sol=solute,
                        pol=polymer,
                        xlxs_sheet_refno_list=[refno],
                        display_plot=False,
                        save_plot_dir=savedir5,
                    )
                except Exception as e:
                    print("Error: ", e)
            except:
                print(f"Error: unsuccessful for {solute}-{polymer}_{Temp-273}C ({refno})")
            else:
                # add new entry to df
                sheet = f"S_{Temp-273}C ({refno})"
                ref = f"{refno}"
                df1 = pd.DataFrame(
                    {
                        "sol": [solute],
                        "pol": [polymer],
                        "T (C)": [Temp - 273],
                        "ref no": [ref],
                        "xlsx sheet": [sheet],
                        "fitted rho20 (g/cm^3)": [rho20_f],
                        "AAD%_rho20_fit (%)": [rho20_AADpc_f],
                        "rho20 PVT (g/cm^3)": [rho20_PVT],
                        "AAD%_rho20_parity (%)": [rho20_parity_AADpc],
                        "fitted ksw (MPa^-1)": [ksw_f],
                        "AAD%_ksw_fit (%)": [ksw_AADpc_f],
                        "predicted ksw (MPa^-1)": [ksw_p],
                        "AAD%_ksw_parity (%)": [ksw_parity_AADpc],
                        "pg (Pa)": [pg],
                        "Sg (g/g)": [Sg],
                    }
                )
                # append to current csv file
                df1.to_csv(
                    csv_dir,
                    mode="a",
                    index=True,
                    header=False,
                )
                print("")
                print(f"Data {solute}-{polymer}_{Temp-273}C ({ref}) appended to {csv_name}")
                print("")

    # Export dataframe as .csv
    print(f".csv result export finished: {csv_dir}")

    print("\n--- Run time %s-%s:\t%.0f seconds ---\n" % (solute, polymer, time.time() - start_time))


if __name__ == "__main__":
    CO2_PS_temp_list = [
        35 + 273,
        40 + 273,
        50 + 273,
        51 + 273,
        60 + 273,
        65 + 273,
        80 + 273,
        81 + 273,
        90 + 273,
        100 + 273,
        110 + 273,
        130 + 273,
        132 + 273,
        140 + 273,
        150 + 273,
        180 + 273,
        200 + 273,
    ]  # [K]

    CO2_PMMA_temp_list = [
        30 + 273,
        33 + 273,
        35 + 273,
        40 + 273,
        42 + 273,
        50 + 273,
        51 + 273,
        59 + 273,
        60 + 273,
        65 + 273,
        80 + 273,
        81 + 273,
        100 + 273,
        125 + 273,
        132 + 273,
        150 + 273,
        175 + 273,
        200 + 273,
    ]  # [K]

    CO2_PEMA_temp_list = [
        15 + 273,
        25 + 273,
        35 + 273,
        45 + 273,
    ]  # [K]
    
    # rho20PVT_fitrho20_fitksw_predictksw_pg(solute="CO2",polymer="PS",temp_list=CO2_PS_temp_list)
    # rho20PVT_fitrho20_fitksw_predictksw_pg(solute="CO2",polymer="PMMA",temp_list=CO2_PMMA_temp_list)
    # fitrho20_fitksw_predictksw_pg(solute="CO2", polymer="PS", temp_list=CO2_PS_temp_list)    
    # fitrho20_fitksw_predictksw_pg(solute="CO2", polymer="PMMA", temp_list=CO2_PMMA_temp_list)
    rho20PVT_fitksw_predictksw_plotpermisotherm(solute="CO2", polymer="PS", temp_list=[35+273])
    fitrho20_fitksw_predictksw_plotpermisotherm(solute="CO2", polymer="PS", temp_list=[35+273])