import math
import os
import time
from datetime import datetime
# import addcopyfighandler
import matplotlib.pyplot as plt
import pandas as pd
import NET_SAFTgMie_master as NE_SAFT
import numpy as np
import re
import logging
from functools import wraps

# Define decorator to log SQL operations
def logger(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logging.info(f"Calling function '{func.__name__}' with args: {args}, kwargs: {kwargs}")
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(f"Function '{func.__name__}' execution time: {execution_time:.4f} seconds")        
        return result
    
    return wrapper

@logger
def plot_isotherm_EQvNE(
    p_l: float,
    p_u: float,
    no_of_points: int,
    T: float,
    sol: str,
    pol: str,
    ksw_list: list[float],
    rho20: float = None,
    MW2: float = None,
    xlxs_sheet_refno_list: list[str] = None,
    display_plot: bool = True,
    save_plot_dir: str = None,
) -> None:
    (
        _eos_mix,
        _eos_sol,
        _MW_1,
        _MW_2,
        _MW_monomer,
        rho_2_am_dry,
        _k_sw,
    ) = NE_SAFT.get_mixture_info(sol, pol, MW2)
    rho20 = rho20 if rho20 != None else rho_2_am_dry  # [g/cm^3]
    MW2 = MW2 if MW2 != None else _MW_2  # [g/mol]

    # Import exp data
    try:
        # Read exp file
        path = os.path.join(os.path.dirname(__file__), "litdata")
        refpath = path + "/references.xlsx"
        databasepath = path + "/%s-%s.xlsx" % (sol, pol)
        reffile = pd.ExcelFile(refpath, engine="openpyxl")
        datafile = pd.ExcelFile(databasepath, engine="openpyxl")
        # Get all sheets matching T
        # print(file.sheet_names)
        matched_sheets = []
        if xlxs_sheet_refno_list == None:
            search_pattern = f"^S_{T-273}C (.*)"  # strat with S_{T-273}C
            for sheet in datafile.sheet_names:
                if re.search(search_pattern, sheet):
                    matched_sheets.append(sheet)
        elif isinstance(xlxs_sheet_refno_list, list):  # check if a list
            for i in xlxs_sheet_refno_list:
                search_pattern = f"^S_{T-273}C.*({i})"
                for sheet in datafile.sheet_names:
                    if re.search(search_pattern, sheet):
                        matched_sheets.append(sheet)
        print("Sheets: ", matched_sheets)
        dict = {}  # dictionary of all matched sheet df
        ref_no = []
        for sheet in matched_sheets:
            dict[sheet] = pd.read_excel(datafile, sheet)
            dict[sheet].dropna(subset=["P [MPa]"], inplace=True)
            ref_no.append(sheet[sheet.find("(") + 1 : sheet.find(")")])
        print(ref_no)
        ref_ID = []
        ref_df = pd.read_excel(reffile, "references")
        # print(ref_df)
        for i, no in enumerate(ref_no):
            ref_ID.append(ref_df.loc[ref_df["# ref"] == f"[{no}]", "refID"].item())
        print(ref_ID)
    except Exception as e:
        print("")
        print("Error - importing exp data failed:")
        print(e)

    hasExpData = True if len(matched_sheets) > 0 else False

    # calculating solubilityno_p_points
    p_calc = np.linspace(p_l, p_u, no_of_points)  # [Pa]
    p_MPa_calc = p_calc * 1e-6

    # Create empty placeholders, return None in case calculation fails
    solubility_NE_calc_list = [None for i in range(len(ksw_list))]
    p_MPa_exp_list = [None for i in range(len(matched_sheets))]
    solubility_exp_list = [None for i in range(len(matched_sheets))]
    solubility_calc_evaluation_NE_list = [None for i in range(len(ksw_list))]
    AAD_percent_NE = [None for i in range(len(ksw_list))]
    label_NE = [None for i in range(len(ksw_list))]
    print("p_cal = ", p_calc)
    # calculated NE solubility for each ksw
    for i, ksw_ in enumerate(ksw_list):
        solubility_NE_calc_list[i] = [NE_SAFT.solve_solubility_NE(T, _p_, sol, pol, MW2, ksw_list[i], rho20) for _p_ in p_calc]
        print(
            "solubility at ksw=%g =\t" % ksw_list[i], solubility_NE_calc_list[i]
        )  # calculated NE solubility with ksw != 0

    # calculated EQ solubility
    solubility_EQ_list = [NE_SAFT.solve_solubility_EQ(T, p_, sol, pol, MW2) for p_ in p_calc]
    print("\nsolubility_EQ = ", solubility_EQ_list)

    # Original label
    label_EQ = "EQ"
    label_NE = [r"NE $k_{sw} = %.3g \, MPa^{-1}$" % (ksw_) for ksw_ in ksw_list]

    # Importing exp data
    if hasExpData == True:
        for i, sheet in enumerate(matched_sheets):
            p_MPa_exp_list[i] = np.asarray(dict[sheet]["P [MPa]"])
            solubility_exp_list[i] = np.asarray(dict[sheet]["Solubility [g-sol/g-pol-am]"])
        # calculate AAD for NE and EQ when there is only 1 exp data sheet
        if len(matched_sheets) == 1:
            p_exp_list = p_MPa_exp_list[0] * 1e6  # [Pa]
            sol_exp_list = solubility_exp_list[0]
            solubility_calc_evaluation_EQ = [NE_SAFT.solve_solubility_EQ(T, p_, sol, pol, MW2) for p_ in p_exp_list]
            # try:
            #     AAD_percent_EQ = get_fitting_AAD(sol_exp_list, solubility_calc_evaluation_EQ) * 100  # [%]
            # except:
            #     AAD_percent_EQ = 0
            # print("AAD%% for EQ: AAD%% = %.1f%%" % (AAD_percent_EQ))
            # label_EQ += " (AAD%%=%.1f%%)" % AAD_percent_EQ
            # for i, ksw_ in enumerate(ksw_list):
            #     solubility_calc_evaluation_NE_list[i] = [
            #         solve_solubility_NE(T, _p_, sol, pol, MW2, ksw_list[i], rho20) for _p_ in p_exp_list
            #     ]
            #     try:
            #         AAD_percent_NE[i] = (
            #             get_fitting_AAD(sol_exp_list, solubility_calc_evaluation_NE_list[i]) * 100
            #         )  # [%]
            #     except:
            #         AAD_percent_NE[i] = 0
            #     print("AAD%% for NE ksw=%g: AAD%% = %.1f%%" % (ksw_, AAD_percent_NE[i]))
            #     label_NE[i] += " (AAD%%=%.1f%%)" % AAD_percent_NE[i]

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Exp solubility
    if hasExpData == True:
        for i, sheet in enumerate(matched_sheets):
            ax.plot(
                p_MPa_exp_list[i],
                solubility_exp_list[i],
                color=NE_SAFT.exp_style["color"],
                marker=NE_SAFT.custom_markers[i],
                linestyle=NE_SAFT.exp_style["linestyle"],
                markerfacecolor=NE_SAFT.exp_style["markerfacecolor"],
                label=f"exp: {ref_ID[i]} ({ref_no[i]})",
            )

    # EQ solubility
    ax.plot(
        p_MPa_calc,
        solubility_EQ_list,
        color=NE_SAFT.custom_colours[1],
        marker="None",
        linestyle="solid",
        label=label_EQ,
    )

    # NE solubility
    for i, ksw_ in enumerate(ksw_list):
        ax.plot(
            p_MPa_calc,
            solubility_NE_calc_list[i],
            color=NE_SAFT.custom_colours[2],
            marker=NE_SAFT.calc_style["marker"],
            linestyle=NE_SAFT.custom_linestyles[i],
            label=label_NE[i],
        )
    # labelling
    ax.set_xlabel(r"p (MPa)")
    ax.set_ylabel(r"Solubility ($g_{sol} / g_{pol\:am}$)")
    ax.set_title("%s-%s at %.0f°C " % (sol, pol, T - 273))
    # ax.annotate(
    #     r"NE $\rho_{20}$ = %.4f $g/cm^{-3}$" % rho20,
    #     xy=(1.0, -0.09),
    #     xycoords="axes fraction",
    #     ha="right",
    #     va="center",
    #     fontsize="xx-small",
    # )
    # styling
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    # ax.grid(visible=True)
    ax.tick_params(direction="in")
    legend_ncol = 1 if (len(matched_sheets) + len(ksw_list)) < 5 else 2
    ax.legend(ncol=legend_ncol).set_visible(True)
    if save_plot_dir != None:
        plt.savefig(save_plot_dir, dpi=1200)
        print(f"Plot saved: {save_plot_dir}")
        print("")
    if display_plot == True:
        plt.show()

@logger
def plot_isotherm_EQ(
    T: float,
    sol: str,
    pol: str,    
    p_l: float = None,
    p_u: float = None,
    no_p_points: int = 20,
    MW2: float = None,
    xlxs_sheet_refno_list: list[str] = None,
    display_plot: bool = True,
    save_plot_dir: str = None,
) -> None:
    (
        _eos_mix,
        _eos_sol,
        _MW_1,
        _MW_2,
        _MW_monomer,
        rho_2_am_dry,
        _k_sw,
    ) = NE_SAFT.get_mixture_info(sol, pol, MW2)
    MW2 = MW2 if MW2 != None else _MW_2  # [g/mol]

    # Import exp data
    try:
        # Read exp file
        path = os.path.join(os.path.dirname(__file__), "litdata")
        refpath = path + "/references.xlsx"
        databasepath = path + "/%s-%s.xlsx" % (sol, pol)
        reffile = pd.ExcelFile(refpath, engine="openpyxl")
        datafile = pd.ExcelFile(databasepath, engine="openpyxl")
        # Get all sheets matching T
        # print(file.sheet_names)
        matched_sheets = []
        if xlxs_sheet_refno_list == None:
            search_pattern = f"^S_{T-273}C (.*)"  # strat with S_{T-273}C
            for sheet in datafile.sheet_names:
                if re.search(search_pattern, sheet):
                    matched_sheets.append(sheet)
        elif isinstance(xlxs_sheet_refno_list, list):  # check if a list
            for i in xlxs_sheet_refno_list:
                search_pattern = f"^S_{T-273}C.\({i}\)"
                for sheet in datafile.sheet_names:
                    if re.search(search_pattern, sheet):
                        matched_sheets.append(sheet)
        
        print("Sheets: ", matched_sheets)
        dict = {}  # dictionary of all matched sheet df
        ref_no = []
        for sheet in matched_sheets:
            dict[sheet] = pd.read_excel(datafile, sheet)
            dict[sheet].dropna(subset=["P [MPa]"], inplace=True)
            ref_no.append(sheet[sheet.find("(") + 1 : sheet.find(")")])
        print(ref_no)
        ref_ID = []
        ref_df = pd.read_excel(reffile, "references")

        for i, no in enumerate(ref_no):
            ref_ID.append(ref_df.loc[ref_df["# ref"] == f"[{no}]", "refID"].item())
        print(ref_ID)
    except Exception as e:
        print("")
        print("Error - importing exp data failed:")
        print(e)

    hasExpData = True if len(matched_sheets) > 0 else False

    # Create empty placeholders, return None in case calculation fails
    p_MPa_exp_list = [None for i in range(len(matched_sheets))]
    solubility_exp_list = [None for i in range(len(matched_sheets))]  
    
    # Importing exp data
    if hasExpData == True:
        for i, sheet in enumerate(matched_sheets):
            p_MPa_exp_list[i] = np.asarray(dict[sheet]["P [MPa]"])
            solubility_exp_list[i] = np.asarray(dict[sheet]["Solubility [g-sol/g-pol-am]"])
            
    # Get pressure range
    if p_l != None and p_u != None:
        p_calc = np.linspace(p_l, p_u, no_p_points)  # [Pa]
    else:
        if hasExpData == True:
            for i, sheet in enumerate(matched_sheets):
                current_max_p_MPa = p_MPa_exp_list[i].max()
                if i == 0:
                    max_p_MPa = current_max_p_MPa
                else:
                    max_p_MPa = max(current_max_p_MPa, max_p_MPa)
        
        max_p = max_p_MPa * 1e6  # [Pa]
        p_calc = np.linspace(1, max_p, no_p_points)    # [Pa]
        
    p_MPa_calc = p_calc * 1e-6
    print("p_cal = ", p_calc)

    # Calculate EQ solubility
    solubility_EQ_list = [NE_SAFT.solve_solubility_EQ(T, p_, sol, pol, MW2) for p_ in p_calc]
    print("\nsolubility_EQ = ", solubility_EQ_list)
    
    # Calculate upper limits of x and y axis    
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    for i, sheet in enumerate(matched_sheets):
        x_min = min(x_min, min(p_MPa_exp_list[i]))
        x_max = max(x_max, max(p_MPa_exp_list[i]))
        y_min = min(y_min, min(solubility_exp_list[i]))
        y_max = max(y_max, max(solubility_exp_list[i]))        
    
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Exp solubility
    if hasExpData == True:
        for i, sheet in enumerate(matched_sheets):
            ax.plot(
                p_MPa_exp_list[i],
                solubility_exp_list[i],
                color=NE_SAFT.exp_style["color"],
                marker=NE_SAFT.custom_markers[i],
                linestyle=NE_SAFT.exp_style["linestyle"],
                markerfacecolor=NE_SAFT.exp_style["markerfacecolor"],
                label=f"exp: {ref_ID[i]} ({ref_no[i]})",
            )

    # EQ solubility
    ax.plot(
        p_MPa_calc,
        solubility_EQ_list,
        color=NE_SAFT.custom_colours[1],
        marker="None",
        linestyle="solid",
        label="EQ",
    )

    # Labelling
    ax.set_xlabel(r"p (MPa)")
    ax.set_ylabel(r"Solubility ($g_{sol} \: / \: g_{pol\:am}$)")
    ax.set_title("%s-%s at %.0f°C " % (sol, pol, T - 273))
    
    # Adjust x and y tick to cover all data
    # Get the length of major ticks on the x-axis
    x_major_tick_length = ax.get_xticks()[1] - ax.get_xticks()[0]
    
    # Get the length of major ticks on the y-axis
    y_major_tick_length = ax.get_yticks()[1] - ax.get_yticks()[0]
    
    # Set adjust x and y tick to cover all data
    ax.set_xlim(left=0, right=x_max + x_major_tick_length)
    ax.set_ylim(bottom=0, top=y_max + y_major_tick_length)

    # Set ticks to appear inside
    ax.tick_params(direction="in")
    
    # Dynamic column number of legend
    legend_ncol = 1 if (len(matched_sheets)) < 5 else 2
    ax.legend(ncol=legend_ncol, loc='upper left').set_visible(True)    
    
    if save_plot_dir != None:
        plt.savefig(save_plot_dir, dpi=1200)
        print(f"Plot saved: {save_plot_dir}")
        print("")
    if display_plot == True:
        plt.show()

if __name__ == "__main__":
    src_dir = os.path.dirname(__file__)
    src_dir = r"\\?\%s" % src_dir  # extended path (for very long path length)

    start_time = time.time()
    now = datetime.now()  # current time
    time_ID = now.strftime("%y%m%d_%H%M")  # YYMMDD_HHMM
    
    # Create new directory to store results
    result_folder_dir = src_dir 
    figname = f"CO2_PS_isotherm_{time_ID}.png"
    savedir = result_folder_dir + f"\\{figname}"
    
    # plot_isotherm_EQvNE(
        # p_l=1,
        # p_u=25e6,
        # no_of_points=40,
        # T=35+273,
        # sol="CO2",
        # pol="PS",
        # ksw_list=[],
        # rho20=1.0561,
        # xlxs_sheet_refno_list=["8"],
        # display_plot=True,
        # save_plot_dir=savedir,
    # )
    
    plot_isotherm_EQ(
        # p_l=1,
        # p_u=25e6,
        no_p_points=5,
        T=100+273,
        sol="CO2",
        pol="PS",
        # xlxs_sheet_refno_list=["8"],
        display_plot=True,
        # save_plot_dir=savedir,
    )