import math
import os
import time
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import NET_SAFTgMie_master as NE_SAFT
import numpy as np
import re
import logging
from functools import wraps
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from matplotlib.lines import Line2D

# Workaround to avoid error messages from addcopyfighandler
import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
warnings.filterwarnings("ignore")
import addcopyfighandler

matplotlib.rcParams["figure.figsize"] = [4.0, 3.5]  # in inches
matplotlib.rcParams["mathtext.default"] = "regular"  # same as regular text
matplotlib.rcParams["font.family"] = "DejaVu Sans"  # alternative: "serif"
matplotlib.rcParams["font.size"] = 10.0
matplotlib.rcParams["axes.titlesize"] = "small"  # relative to font.size
matplotlib.rcParams["axes.labelsize"] = "small"  # relative to font.size
matplotlib.rcParams["xtick.labelsize"] = "x-small"  # relative to font.size
matplotlib.rcParams["ytick.labelsize"] = "x-small"  # relative to font.size
matplotlib.rcParams["legend.fontsize"] = "xx-small"  # relative to font.size
matplotlib.rcParams["legend.frameon"] = False
matplotlib.rcParams["grid.linestyle"] = "-."
matplotlib.rcParams["grid.linewidth"] = 0.15  # in point units
matplotlib.rcParams["figure.autolayout"] = True

# Paper references
paper_ref_dict = {
    'Wissinger 1987': '32',
    'Pantoula 2006': '33',
    'Vogt 2003': '34',
    'Zhang 1997': '35',
    'Sato 1996': '36',
    'Sato 2001': '38',
    'Wong 1998': '39',
    'Conforti 1996': '40',
    'Ushiki 2019': '42',
    'Chiou 1986': '45',   
    'Edwards 1998': '55',
}

def update_subplot_ticks(ax, x_lo=None, y_lo=None, x_up=None, y_up=None):
    """Update x and y ticks of subplot ax to cover all data. Put ticks to inside.

    Args:
        ax: plot object.
    """
    # Adjust lower x and y ticks to start from 0
    if x_lo != None:
        ax.set_xlim(left=x_lo)
    if y_lo != None:
        ax.set_ylim(bottom=y_lo)
    if x_up != None:
        ax.set_xlim(right=x_up)
    if y_up != None:
        ax.set_ylim(top=y_up)
    
    # Get the largest and smallest x ticks and y ticks
    max_y_tick = max(ax.get_yticks())
    max_x_tick = max(ax.get_xticks())
    min_y_tick = min(ax.get_yticks())
    min_x_tick = min(ax.get_xticks())
    
    # Get the length of major ticks on the x-axis
    ax_x_major_tick_length = ax.get_xticks()[1] - ax.get_xticks()[0]
    ax_y_major_tick_length = ax.get_yticks()[1] - ax.get_yticks()[0]
    
    # Adjust upper x and y ticks to cover all data
    if x_up == None:        
        ax.set_xlim(right=max_x_tick + ax_x_major_tick_length)
    if y_up == None:
        ax.set_ylim(top=max_y_tick + ax_y_major_tick_length)
    if x_lo == None:
        ax.set_xlim(left=min_x_tick - ax_x_major_tick_length)
    if y_lo == None:
        ax.set_ylim(bottom=min_y_tick - ax_y_major_tick_length)
    
    # Put ticks to inside
    ax.tick_params(direction="in")

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
    T: float,
    ksw_list: list[float],
    rho20: float,
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

    #* Calculate EQ solubility
    # solubility_EQ_list = [NE_SAFT.solve_solubility_EQ(T, p_, sol, pol, MW2) for p_ in p_calc]
    # print("\nsolubility_EQ = ", solubility_EQ_list)

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
            # solubility_calc_evaluation_EQ = [NE_SAFT.solve_solubility_EQ(T, p_, sol, pol, MW2) for p_ in p_exp_list]
            # try:
            #     AAD_percent_EQ = get_fitting_AAD(sol_exp_list, solubility_calc_evaluation_EQ) * 100  # [%]
            # except:
            #     AAD_percent_EQ = 0
            # print("AAD%% for EQ: AAD%% = %.1f%%" % (AAD_percent_EQ))
            # label_EQ += " (AAD%%=%.1f%%)" % AAD_percent_EQ
            
            # Calculate AAD for NE
            # for i, ksw_ in enumerate(ksw_list):
            #     solubility_calc_evaluation_NE_list[i] = [
            #         NE_SAFT.solve_solubility_NE(T, _p_, sol, pol, MW2, ksw_list[i], rho20) for _p_ in p_exp_list
            #     ]
            #     try:
            #         AAD_percent_NE[i] = (
            #             NE_SAFT.get_fitting_AAD(sol_exp_list, solubility_calc_evaluation_NE_list[i]) * 100
            #         )  # [%]
            #     except:
            #         AAD_percent_NE[i] = 0
            #     print("AAD%% for NE ksw=%g: AAD%% = %.1f%%" % (ksw_, AAD_percent_NE[i]))
                # label_NE[i] += " (AAD%%=%.1f%%)" % AAD_percent_NE[i]

    # Calculate upper limits of x and y axis    
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    
    # Get min and max values for x and y axis from experimental data
    for i, sheet in enumerate(matched_sheets):
        x_min = min(x_min, min(p_MPa_exp_list[i]))
        x_max = max(x_max, max(p_MPa_exp_list[i]))
        y_min = min(y_min, min(solubility_exp_list[i]))
        y_max = max(y_max, max(solubility_exp_list[i]))   
        
    # Get min and max values for x and y axis from EQ solubility
    # y_min = min(y_min, min(solubility_EQ_list))
    # y_max = max(y_max, max(solubility_EQ_list))
    
    # Get min and max values for x and y axis from NE solubility
    for i, ksw_ in enumerate(ksw_list):
        y_min = min(y_min, min(solubility_NE_calc_list[i]))
        y_max = max(y_max, max(solubility_NE_calc_list[i])) 
    
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

    #* EQ solubility
    # ax.plot(
    #     p_MPa_calc,
    #     solubility_EQ_list,
    #     color=NE_SAFT.custom_colours[1],
    #     marker="None",
    #     linestyle="solid",
    #     label=label_EQ,
    # )

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
    ax.set_ylabel(r"Solubility ($g_{sol} / g_{pol}$)")
    ax.set_title("%s-%s at %.0f°C " % (sol, pol, T - 273))
    # ax.annotate(
    #     r"NE $\rho_{20}$ = %.4f $g/cm^{-3}$" % rho20,
    #     xy=(1.0, -0.09),
    #     xycoords="axes fraction",
    #     ha="right",
    #     va="center",
    #     fontsize="xx-small",
    # )
    
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
    legend_ncol = 1 if (len(matched_sheets) + len(ksw_list)) < 5 else 2
    ax.legend(ncol=legend_ncol).set_visible(True)
    
    if save_plot_dir != None:
        plt.savefig(save_plot_dir, dpi=1200)
        print(f"Plot saved: {save_plot_dir}")
        print("")
    if display_plot == True:
        plt.show()

@logger
def plot_isotherm_EQvNE_multiT(
    T_list: list[float],
    ksw_list: list[float],
    rho20_list: list[float],
    sol: str,
    pol: str,
    p_l: float = None,
    p_u: float = None,
    no_p_points: int = 20,
    MW2: float = None,
    xlxs_sheet_refno_list: list[str] = None,
    include_EQ: bool = True,
    include_NE: bool = True,
    display_plot: bool = True,
    fig_size: tuple = None,
    x_lo: float = None,
    x_up: float = None,
    y_lo: float = None,
    y_up: float = None,
    save_plot_dir: str = None,
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
        raise ValueError("T_list, ksw_list and rho20_list must have the same length")
    
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
            path = os.path.join(os.path.dirname(__file__), "litdata")
            refpath = path + "/references.xlsx"
            databasepath = path + "/%s-%s.xlsx" % (sol, pol)
            reffile = pd.ExcelFile(refpath, engine="openpyxl")
            datafile = pd.ExcelFile(databasepath, engine="openpyxl")
            
            # Get all sheets matching T
            # print(file.sheet_names)
            matched_sheets[i] = []
            if xlxs_sheet_refno_list == None:
                search_pattern = f"^S_{T-273}C (.*)"  # strat with S_{T-273}C ()
                for sheet in datafile.sheet_names:
                    if re.search(search_pattern, sheet):
                        matched_sheets[i].append(sheet)
            elif isinstance(xlxs_sheet_refno_list, list):  # check if a list
                for j in xlxs_sheet_refno_list:
                    search_pattern = f"^S_{T-273}C.\({j}\)"
                    for sheet in datafile.sheet_names:
                        if re.search(search_pattern, sheet):
                            matched_sheets[i].append(sheet)

            print(f"Sheets for {T-273}C: ", matched_sheets[i])

            ref_no[i] = []
            for sheet in matched_sheets[i]:
                dict[sheet] = pd.read_excel(datafile, sheet)
                dict[sheet].dropna(subset=["P [MPa]"], inplace=True)
                ref_no[i].append(sheet[sheet.find("(") + 1 : sheet.find(")")])
            print(ref_no[i])

            ref_ID[i] = []
            ref_df = pd.read_excel(reffile, "references")
            # print(ref_df)
            for no in ref_no[i]:
                ref_ID[i].append(ref_df.loc[ref_df["# ref"] == f"[{no}]", "refID"].item())
                # print(ref_ID[i])
            print(ref_ID[i])
        except Exception as e:
            print("")
            print("Error - importing exp data failed:")
            print(e)
        # print(len(matched_sheets[i]))
        hasExpData[i] = True if len(matched_sheets[i]) > 0 else False
    print("hasExpData = ", hasExpData)
    
    # Empty array to store results
    solubility_EQ = [None for i in range(len(T_list))]
    solubility_NE = [None for i in range(len(T_list))]
    
    # Get pressure range
    if p_l != None and p_u != None:
        p_calc = np.linspace(p_l, p_u, no_p_points)  # [Pa]
    else:
        for i, T in enumerate(T_list):
            if hasExpData[i] == True:
                for j, sheet in enumerate(matched_sheets[i]):
                    current_max_p_MPa = np.asarray(dict[sheet]["P [MPa]"]).max()
                    if i == 0 and j == 0:
                        max_p_MPa = current_max_p_MPa
                    else:
                        max_p_MPa = max(current_max_p_MPa, max_p_MPa)
        
        max_p = max_p_MPa * 1e6 # [Pa]
        p_calc = np.linspace(1, max_p, no_p_points)    # [Pa]
        
    p_MPa_calc = p_calc * 1e-6  # [MPa]        
        
    for i, T in enumerate(T_list):
        
        #* Calculate EQ solubility
        if include_EQ == True:
            solubility_EQ[i] = [NE_SAFT.solve_solubility_EQ(T, p_, sol, pol, MW2) for p_ in p_calc]
            print("\nSolubility_EQ at %s°C: " % (T - 273), solubility_EQ[i])
        
        #* Calculate NE solubility
        if include_NE == True:
            solubility_NE[i] = [NE_SAFT.solve_solubility_NE(T, p_, sol, pol, MW2, ksw_list[i], rho20_list[i]) for p_ in p_calc]
            print("\nSolubility_NE at %s°C and ksw = %g:" % (T - 273, ksw_list[i]), solubility_NE[i])

    # Save data to dataframe
    data = {
        "T [°C]": np.repeat(np.array(T_list)-273, no_p_points),
        "p [MPa]": np.tile(p_MPa_calc, len(T_list)),
        "solubility_EQ [g-sol/g-pol]": np.concatenate(solubility_EQ) if include_EQ else [np.nan] * len(T_list) * no_p_points,
        "solubility_NE [g-sol/g-pol]": np.concatenate(solubility_NE) if include_NE else [np.nan] * len(T_list) * no_p_points,
    }
    
    df = pd.DataFrame(data)
    
    print(df)
    
    if save_data_dir != None:
        df.to_csv(save_data_dir, index=False)
        print(f"Data saved: {save_data_dir}")
        print("")
    
    # Plotting
    if fig_size == None:
        fig = plt.figure()  # Default        
    elif isinstance(fig_size, tuple):
        fig = plt.figure(figsize=fig_size)  # Big plot        
    else:
        fig = plt.figure()  # Default
    
    ax = fig.add_subplot(111)

    for i, T in enumerate(T_list):
        
        # Exp solubility
        if hasExpData[i] == True:
            for j, sheet in enumerate(matched_sheets[i]):
                ax.plot(dict[sheet]['P [MPa]'],
                        dict[sheet]['Solubility [g-sol/g-pol-am]'],
                        color=NE_SAFT.custom_colours[i],
                        marker=NE_SAFT.custom_markers[j],
                        # markersize=2,
                        linestyle='None',
                        markerfacecolor='None',
                        label=f"exp {T-273} °C: {ref_ID[i][j]}",
                        )
            
        # EQ calc
        if include_EQ == True:
            ax.plot(p_MPa_calc,
                    solubility_EQ[i],
                    color=NE_SAFT.custom_colours[i],
                    marker='None',
                    linestyle='solid',
                    label=f"EQ model {T-273} °C",
                    )

        # NE solubility    
        if include_NE == True:
            ax.plot(p_MPa_calc,
                    solubility_NE[i],
                    color=NE_SAFT.custom_colours[i],
                    marker='None',
                    linestyle='dashed', 
                    label=f"NE model {T-273} °C",
                    )
        
    # Labelling
    ax.set_xlabel(r"p / MPa")
    ax.set_ylabel(r"Solubility $/$ $g_{s} \; g^{-1}_{p}$")
    # ax.set_title(r"%s-%s" % (sol, pol)) 
    
    # Update ticks to cover all data points
    update_subplot_ticks(ax,  x_lo=x_lo, x_up=x_up, y_lo=y_lo, y_up=y_up)
    
    # Dynamic column number of legend
    ax.legend(loc='upper left').set_visible(True)
    
    if save_plot_dir != None:
        plt.savefig(save_plot_dir, dpi=1200)
        print(f"Plot saved: {save_plot_dir}")
        print("")
    
    if display_plot == True:
        plt.show()

@logger
def plot_isotherm_EQvNE_multiT_custom(
    T_list: list[float],
    ksw_list: list[float],
    rho20_list: list[float],
    sol: str,
    pol: str,
    p_l: float = None,
    p_u: float = None,
    no_p_points: int = 20,
    MW2: float = None,
    xlxs_sheet_refno_list: list[str] = None,
    include_EQ: bool = True,
    include_NE: bool = True,
    display_plot: bool = True,
    fig_size: tuple = None,
    x_lo: float = None,
    x_up: float = None,
    y_lo: float = None,
    y_up: float = None,
    save_plot_dir: str = None,
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
        raise ValueError("T_list, ksw_list and rho20_list must have the same length")
    
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
            path = os.path.join(os.path.dirname(__file__), "litdata")
            refpath = path + "/references.xlsx"
            databasepath = path + "/%s-%s.xlsx" % (sol, pol)
            reffile = pd.ExcelFile(refpath, engine="openpyxl")
            datafile = pd.ExcelFile(databasepath, engine="openpyxl")
            
            # Get all sheets matching T
            # print(file.sheet_names)
            matched_sheets[i] = []
            if xlxs_sheet_refno_list == None:
                search_pattern = f"^S_{T-273}C (.*)"  # strat with S_{T-273}C ()
                for sheet in datafile.sheet_names:
                    if re.search(search_pattern, sheet):
                        matched_sheets[i].append(sheet)
            elif isinstance(xlxs_sheet_refno_list, list):  # check if a list
                for j in xlxs_sheet_refno_list:
                    search_pattern = f"^S_{T-273}C.\({j}\)"
                    for sheet in datafile.sheet_names:
                        if re.search(search_pattern, sheet):
                            matched_sheets[i].append(sheet)

            print(f"Sheets for {T-273}C: ", matched_sheets[i])

            ref_no[i] = []
            for sheet in matched_sheets[i]:
                dict[sheet] = pd.read_excel(datafile, sheet)
                dict[sheet].dropna(subset=["P [MPa]"], inplace=True)
                ref_no[i].append(sheet[sheet.find("(") + 1 : sheet.find(")")])
            print(ref_no[i])

            ref_ID[i] = []
            ref_df = pd.read_excel(reffile, "references")
            # print(ref_df)
            for no in ref_no[i]:
                ref_ID[i].append(ref_df.loc[ref_df["# ref"] == f"[{no}]", "refID"].item())
                # print(ref_ID[i])
            print(ref_ID[i])
        except Exception as e:
            print("")
            print("Error - importing exp data failed:")
            print(e)
        # print(len(matched_sheets[i]))
        hasExpData[i] = True if len(matched_sheets[i]) > 0 else False
    print("hasExpData = ", hasExpData)
    
    # Empty array to store results
    solubility_EQ = [None for i in range(len(T_list))]
    solubility_NE = [None for i in range(len(T_list))]
    
    # Get pressure range
    if p_l != None and p_u != None:
        p_calc = np.linspace(p_l, p_u, no_p_points)  # [Pa]
    else:
        for i, T in enumerate(T_list):
            if hasExpData[i] == True:
                for j, sheet in enumerate(matched_sheets[i]):
                    current_max_p_MPa = np.asarray(dict[sheet]["P [MPa]"]).max()
                    if i == 0 and j == 0:
                        max_p_MPa = current_max_p_MPa
                    else:
                        max_p_MPa = max(current_max_p_MPa, max_p_MPa)
        
        max_p = max_p_MPa * 1e6 # [Pa]
        p_calc = np.linspace(1, max_p, no_p_points)    # [Pa]
        
    p_MPa_calc = p_calc * 1e-6  # [MPa]        
        
    for i, T in enumerate(T_list):
        
        #* Calculate EQ solubility
        if include_EQ == True:
            solubility_EQ[i] = [NE_SAFT.solve_solubility_EQ(T, p_, sol, pol, MW2) for p_ in p_calc]
            print("\nSolubility_EQ at %s°C: " % (T - 273), solubility_EQ[i])
        
        #* Calculate NE solubility
        if include_NE == True:
            solubility_NE[i] = [NE_SAFT.solve_solubility_NE(T, p_, sol, pol, MW2, ksw_list[i], rho20_list[i]) for p_ in p_calc]
            print("\nSolubility_NE at %s°C and ksw = %g:" % (T - 273, ksw_list[i]), solubility_NE[i])

    # Save data to dataframe
    data = {
        "T [°C]": np.repeat(np.array(T_list)-273, no_p_points),
        "p [MPa]": np.tile(p_MPa_calc, len(T_list)),
        "solubility_EQ [g-sol/g-pol]": np.concatenate(solubility_EQ) if include_EQ else [None] * len(T_list) * no_p_points,
        "solubility_NE [g-sol/g-pol]": np.concatenate(solubility_NE) if include_NE else [None] * len(T_list) * no_p_points,
    }
    
    df = pd.DataFrame(data)
    
    print(df)
    
    if save_data_dir != None:
        df.to_csv(save_data_dir, index=False)
        print(f"Data saved: {save_data_dir}")
        print("")
    
    # Plotting
    if fig_size == None:
        fig = plt.figure()  # Default        
    elif isinstance(fig_size, tuple):
        fig = plt.figure(figsize=fig_size)  # Big plot        
    else:
        fig = plt.figure()  # Default
    
    ax = fig.add_subplot(111)

    for i, T in enumerate(T_list):
        
        # Exp solubility
        if hasExpData[i] == True:
            for j, sheet in enumerate(matched_sheets[i]):
                ax.plot(dict[sheet]['P [MPa]'],
                        dict[sheet]['Solubility [g-sol/g-pol-am]'],
                        color=NE_SAFT.custom_colours[i],
                        marker=NE_SAFT.custom_markers[j],
                        # markersize=2,
                        linestyle='None',
                        markerfacecolor='None',
                        label=f"exp {T-273} °C: {ref_ID[i][j]}",
                        )
            
        # EQ calc
        if include_EQ == True:
            ax.plot(p_MPa_calc,
                    solubility_EQ[i],
                    color=NE_SAFT.custom_colours[i],
                    marker='None',
                    linestyle='dashdot',
                    label=f"EQ model {T-273} °C",
                    )

        # NE solubility    
        if include_NE == True:
            ax.plot(p_MPa_calc,
                    solubility_NE[i],
                    color=NE_SAFT.custom_colours[i],
                    marker='None',
                    linestyle='dashed', 
                    label=f"NE model {T-273} °C",
                    )
        
    # Labelling
    ax.set_xlabel(r"p / MPa")
    ax.set_ylabel(r"Solubility $/$ $g_{s} \; g^{-1}_{p}$")
    # ax.set_title(r"%s-%s" % (sol, pol)) 
    
    # Update ticks to cover all data points
    update_subplot_ticks(ax,  x_lo=x_lo, x_up=x_up, y_lo=y_lo, y_up=y_up)
    
    # Dynamic column number of legend
    ax.legend(loc='upper left', handlelength=2.6).set_visible(True)
    
    if save_plot_dir != None:
        plt.savefig(save_plot_dir, dpi=1200)
        print(f"Plot saved: {save_plot_dir}")
        print("")
    
    if display_plot == True:
        plt.show()

def plot_isotherm_EQvNE_multiT_custom2(
    T_list: list[float],
    ksw_list: list[float],
    rho20_list: list[float],
    sol: str,
    pol: str,
    p_l: float = None,
    p_u: float = None,
    no_p_points: int = 20,
    MW2: float = None,
    xlxs_sheet_refno_list: list[str] = None,
    include_EQ: bool = True,
    include_NE: bool = True,
    display_plot: bool = True,
    fig_size: tuple = None,
    x_lo: float = None,
    x_up: float = None,
    y_lo: float = None,
    y_up: float = None,
    save_plot_dir: str = None,
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
        raise ValueError("T_list, ksw_list and rho20_list must have the same length")
    
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
            path = os.path.join(os.path.dirname(__file__), "litdata")
            refpath = path + "/references.xlsx"
            databasepath = path + "/%s-%s.xlsx" % (sol, pol)
            reffile = pd.ExcelFile(refpath, engine="openpyxl")
            datafile = pd.ExcelFile(databasepath, engine="openpyxl")
            
            # Get all sheets matching T
            # print(file.sheet_names)
            matched_sheets[i] = []
            if xlxs_sheet_refno_list == None:
                search_pattern = f"^S_{T-273}C (.*)"  # strat with S_{T-273}C ()
                for sheet in datafile.sheet_names:
                    if re.search(search_pattern, sheet):
                        matched_sheets[i].append(sheet)
            elif isinstance(xlxs_sheet_refno_list, list):  # check if a list
                for j in xlxs_sheet_refno_list:
                    search_pattern = f"^S_{T-273}C.\({j}\)"
                    for sheet in datafile.sheet_names:
                        if re.search(search_pattern, sheet):
                            matched_sheets[i].append(sheet)

            print(f"Sheets for {T-273}C: ", matched_sheets[i])

            ref_no[i] = []
            for sheet in matched_sheets[i]:
                dict[sheet] = pd.read_excel(datafile, sheet)
                dict[sheet].dropna(subset=["P [MPa]"], inplace=True)
                ref_no[i].append(sheet[sheet.find("(") + 1 : sheet.find(")")])
            print(ref_no[i])

            ref_ID[i] = []
            ref_df = pd.read_excel(reffile, "references")
            # print(ref_df)
            for no in ref_no[i]:
                ref_ID[i].append(ref_df.loc[ref_df["# ref"] == f"[{no}]", "refID"].item())
                # print(ref_ID[i])
            print(ref_ID[i])
        except Exception as e:
            print("")
            print("Error - importing exp data failed:")
            print(e)
        # print(len(matched_sheets[i]))
        hasExpData[i] = True if len(matched_sheets[i]) > 0 else False
    print("hasExpData = ", hasExpData)
    
    # Empty array to store results
    solubility_EQ = [None for i in range(len(T_list))]
    solubility_NE = [None for i in range(len(T_list))]
    
    # Get pressure range
    if p_l != None and p_u != None:
        p_calc = np.linspace(p_l, p_u, no_p_points)  # [Pa]
    else:
        for i, T in enumerate(T_list):
            if hasExpData[i] == True:
                for j, sheet in enumerate(matched_sheets[i]):
                    current_max_p_MPa = np.asarray(dict[sheet]["P [MPa]"]).max()
                    if i == 0 and j == 0:
                        max_p_MPa = current_max_p_MPa
                    else:
                        max_p_MPa = max(current_max_p_MPa, max_p_MPa)
        
        max_p = max_p_MPa * 1e6 # [Pa]
        p_calc = np.linspace(1, max_p, no_p_points)    # [Pa]
        
    p_MPa_calc = p_calc * 1e-6  # [MPa]        
        
    for i, T in enumerate(T_list):
        
        #* Calculate EQ solubility
        if include_EQ == True:
            solubility_EQ[i] = [NE_SAFT.solve_solubility_EQ(T, p_, sol, pol, MW2) for p_ in p_calc]
            print("\nSolubility_EQ at %s°C: " % (T - 273), solubility_EQ[i])
        
        #* Calculate NE solubility
        if include_NE == True:
            solubility_NE[i] = [NE_SAFT.solve_solubility_NE(T, p_, sol, pol, MW2, ksw_list[i], rho20_list[i]) for p_ in p_calc]
            print("\nSolubility_NE at %s°C and ksw = %g:" % (T - 273, ksw_list[i]), solubility_NE[i])

    # Save data to dataframe
    data = {
        "T [°C]": np.repeat(np.array(T_list)-273, no_p_points),
        "p [MPa]": np.tile(p_MPa_calc, len(T_list)),
        "solubility_EQ [g-sol/g-pol]": np.concatenate(solubility_EQ) if include_EQ else [None] * len(T_list) * no_p_points,
        "solubility_NE [g-sol/g-pol]": np.concatenate(solubility_NE) if include_NE else [None] * len(T_list) * no_p_points,
    }
    
    df = pd.DataFrame(data)
    
    print(df)
    
    if save_data_dir != None:
        df.to_csv(save_data_dir, index=False)
        print(f"Data saved: {save_data_dir}")
        print("")
    
    # Plotting
    if fig_size == None:
        fig = plt.figure()  # Default        
    elif isinstance(fig_size, tuple):
        fig = plt.figure(figsize=fig_size)  # Big plot        
    else:
        fig = plt.figure()  # Default
    
    ax = fig.add_subplot(111)

    for i, T in enumerate(T_list):
        
        # Exp solubility
        if hasExpData[i] == True:
            for j, sheet in enumerate(matched_sheets[i]):
                ax.plot(dict[sheet]['P [MPa]'],
                        dict[sheet]['Solubility [g-sol/g-pol-am]'],
                        color=NE_SAFT.custom_colours[i],
                        marker=NE_SAFT.custom_markers[j],
                        # markersize=2,
                        linestyle='None',
                        markerfacecolor='None',
                        label=f"exp {T-273} °C: {ref_ID[i][j]}",
                        )
            
        # EQ calc
        if include_EQ == True:
            ax.plot(p_MPa_calc,
                    solubility_EQ[i],
                    color=NE_SAFT.custom_colours[i],
                    marker='None',
                    linestyle='dashdot',
                    label=f"EQ model {T-273} °C",
                    )

        # NE solubility    
        if include_NE == True:
            ax.plot(p_MPa_calc,
                    solubility_NE[i],
                    color=NE_SAFT.custom_colours[i],
                    marker='None',
                    linestyle='dotted', 
                    label=f"NE model {T-273} °C",
                    )
        
    # Labelling
    ax.set_xlabel(r"p / MPa")
    ax.set_ylabel(r"Solubility $/$ $g_{s} \; g^{-1}_{p}$")
    # ax.set_title(r"%s-%s" % (sol, pol)) 
    
    # Update ticks to cover all data points
    update_subplot_ticks(ax,  x_lo=x_lo, x_up=x_up, y_lo=y_lo, y_up=y_up)
    
    # Set legend location and handlelength
    ax.legend(loc='upper left', handlelength=2.6).set_visible(True)
    
    if save_plot_dir != None:
        plt.savefig(save_plot_dir, dpi=1200)
        print(f"Plot saved: {save_plot_dir}")
        print("")
    
    if display_plot == True:
        plt.show()

def plot_isotherm_EQvNE_multiT_custom3(
    T_list: list[float],
    ksw_list: list[float],
    rho20_list: list[float],
    sol: str,
    pol: str,
    p_l: float = None,
    p_u: float = None,
    no_p_points: int = 20,
    MW2: float = None,
    xlxs_sheet_refno_list: list[str] = None,
    include_EQ: bool = True,
    include_NE: bool = True,
    fig_size: tuple = None,
    x_lo: float = None,
    x_up: float = None,
    y_lo: float = None,
    y_up: float = None,
    legend_loc: str = 'best',
    label: str = None,
    display_plot: bool = True,
    save_plot_dir: str = None,
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
        raise ValueError("T_list, ksw_list and rho20_list must have the same length")
    
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
            path = os.path.join(os.path.dirname(__file__), "litdata")
            refpath = path + "/references.xlsx"
            databasepath = path + "/%s-%s.xlsx" % (sol, pol)
            reffile = pd.ExcelFile(refpath, engine="openpyxl")
            datafile = pd.ExcelFile(databasepath, engine="openpyxl")
            
            # Get all sheets matching T
            # print(file.sheet_names)
            matched_sheets[i] = []
            if xlxs_sheet_refno_list == None:
                search_pattern = f"^S_{T-273}C (.*)"  # strat with S_{T-273}C ()
                for sheet in datafile.sheet_names:
                    if re.search(search_pattern, sheet):
                        matched_sheets[i].append(sheet)
            elif isinstance(xlxs_sheet_refno_list, list):  # check if a list
                for j in xlxs_sheet_refno_list:
                    search_pattern = f"^S_{T-273}C.\({j}\)"
                    for sheet in datafile.sheet_names:
                        if re.search(search_pattern, sheet):
                            matched_sheets[i].append(sheet)

            print(f"Sheets for {T-273}C: ", matched_sheets[i])

            ref_no[i] = []
            for sheet in matched_sheets[i]:
                dict[sheet] = pd.read_excel(datafile, sheet)
                dict[sheet].dropna(subset=["P [MPa]"], inplace=True)
                ref_no[i].append(sheet[sheet.find("(") + 1 : sheet.find(")")])
            print(ref_no[i])

            ref_ID[i] = []
            ref_df = pd.read_excel(reffile, "references")
            # print(ref_df)
            for no in ref_no[i]:
                ref_ID[i].append(ref_df.loc[ref_df["# ref"] == f"[{no}]", "refID"].item())
                # print(ref_ID[i])
            print(ref_ID[i])
        except Exception as e:
            print("")
            print("Error - importing exp data failed:")
            print(e)
        # print(len(matched_sheets[i]))
        hasExpData[i] = True if len(matched_sheets[i]) > 0 else False
    print("hasExpData = ", hasExpData)
    
    # Empty array to store results
    solubility_EQ = [None for i in range(len(T_list))]
    solubility_NE = [None for i in range(len(T_list))]
    
    # Get pressure range
    if p_l != None and p_u != None:
        p_calc = np.linspace(p_l, p_u, no_p_points)  # [Pa]
    else:
        for i, T in enumerate(T_list):
            if hasExpData[i] == True:
                for j, sheet in enumerate(matched_sheets[i]):
                    current_max_p_MPa = np.asarray(dict[sheet]["P [MPa]"]).max()
                    if i == 0 and j == 0:
                        max_p_MPa = current_max_p_MPa
                    else:
                        max_p_MPa = max(current_max_p_MPa, max_p_MPa)
        
        max_p = max_p_MPa * 1e6 # [Pa]
        p_calc = np.linspace(1, max_p, no_p_points)    # [Pa]
        
    p_MPa_calc = p_calc * 1e-6  # [MPa]        
        
    for i, T in enumerate(T_list):        
        #* Calculate EQ solubility
        if include_EQ == True:
            solubility_EQ[i] = [NE_SAFT.solve_solubility_EQ(T, p_, sol, pol, MW2) for p_ in p_calc]
            print("\nSolubility_EQ at %s°C: " % (T - 273), solubility_EQ[i])
        
        #* Calculate NE solubility
        if include_NE == True:
            solubility_NE[i] = [NE_SAFT.solve_solubility_NE(T, p_, sol, pol, MW2, ksw_list[i], rho20_list[i]) for p_ in p_calc]
            print("\nSolubility_NE at %s°C and ksw = %g:" % (T - 273, ksw_list[i]), solubility_NE[i])

    # Save data to dataframe
    data = {
        "T [°C]": np.repeat(np.array(T_list)-273, no_p_points),
        "p [MPa]": np.tile(p_MPa_calc, len(T_list)),
        "solubility_EQ [g-sol/g-pol]": np.concatenate(solubility_EQ) if include_EQ else [np.nan] * len(T_list) * no_p_points,
        "solubility_NE [g-sol/g-pol]": np.concatenate(solubility_NE) if include_NE else [np.nan] * len(T_list) * no_p_points,
    }
    
    df = pd.DataFrame(data)
    
    print(df)
    
    if save_data_dir != None:
        df.to_csv(save_data_dir, index=False)
        print(f"Data saved: {save_data_dir}")
        print("")
    
    # Plotting
    if fig_size == None:
        fig = plt.figure()  # Default        
    elif isinstance(fig_size, tuple):
        fig = plt.figure(figsize=fig_size)      
    else:
        fig = plt.figure()  # Default
    
    ax = fig.add_subplot(111)

    for i, T in enumerate(T_list):
        
        # Exp solubility
        if hasExpData[i] == True:
            for j, sheet in enumerate(matched_sheets[i]):
                ax.plot(dict[sheet]['P [MPa]'],
                        dict[sheet]['Solubility [g-sol/g-pol-am]'],
                        color=NE_SAFT.custom_colours[i],
                        marker=NE_SAFT.custom_markers[j],
                        markersize=5,
                        linestyle='None',
                        markerfacecolor='None',
                        label=f"{ref_ID[i][j]} [{paper_ref_dict[ref_ID[i][j]]}]",
                        )
            
        # EQ calc
        if include_EQ == True:
            ax.plot(p_MPa_calc,
                    solubility_EQ[i],
                    color=NE_SAFT.custom_colours[i],
                    marker='None',
                    linestyle='solid',
                    label=f"EQ model {T-273} °C",
                    )

        # NE solubility    
        if include_NE == True:
            ax.plot(p_MPa_calc,
                    solubility_NE[i],
                    color=NE_SAFT.custom_colours[i],
                    marker='None',
                    linestyle='dashed', 
                    label=f"NE model {T-273} °C",
                    )
        
    # Labelling
    ax.set_xlabel(r"p / MPa")
    ax.set_ylabel(r"Solubility $/$ $g_{s} \; g^{-1}_{p}$")
    # ax.set_title(r"%s-%s" % (sol, pol)) 
    
    # Update ticks to cover all data points
    update_subplot_ticks(ax,  x_lo=x_lo, x_up=x_up, y_lo=y_lo, y_up=y_up)
    
    # Dynamic column number of legend
    ax.legend(loc=legend_loc).set_visible(True)
    
    # Add label
    if label != None:
        fig.text(0., 0.98, f'{label}', ha='left', va='top', transform=fig.transFigure)
    
    if save_plot_dir != None:
        plt.savefig(save_plot_dir, dpi=1200)
        print(f"Plot saved: {save_plot_dir}")
        print("")
    
    if display_plot == True:
        plt.show()

def plot_isotherm_EQvNE_multiT_custom4(
    T_list: list[float],
    ksw_list: list[float],
    rho20_list: list[float],
    sol: str,
    pol: str,
    p_l: float = None,
    p_u: float = None,
    no_p_points: int = 20,
    MW2: float = None,
    xlxs_sheet_refno_list: list[str] = None,
    include_EQ: bool = True,
    include_NE: bool = True,
    display_plot: bool = True,
    fig_size: tuple = None,
    x_lo: float = None,
    x_up: float = None,
    y_lo: float = None,
    y_up: float = None,
    legend_loc: str = 'best',
    save_plot_dir: str = None,
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
        raise ValueError("T_list, ksw_list and rho20_list must have the same length")
    
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
            path = os.path.join(os.path.dirname(__file__), "litdata")
            refpath = path + "/references.xlsx"
            databasepath = path + "/%s-%s.xlsx" % (sol, pol)
            reffile = pd.ExcelFile(refpath, engine="openpyxl")
            datafile = pd.ExcelFile(databasepath, engine="openpyxl")
            
            # Get all sheets matching T
            # print(file.sheet_names)
            matched_sheets[i] = []
            if xlxs_sheet_refno_list == None:
                search_pattern = f"^S_{T-273}C (.*)"  # strat with S_{T-273}C ()
                for sheet in datafile.sheet_names:
                    if re.search(search_pattern, sheet):
                        matched_sheets[i].append(sheet)
            elif isinstance(xlxs_sheet_refno_list, list):  # check if a list
                for j in xlxs_sheet_refno_list:
                    search_pattern = f"^S_{T-273}C.\({j}\)"
                    for sheet in datafile.sheet_names:
                        if re.search(search_pattern, sheet):
                            matched_sheets[i].append(sheet)

            print(f"Sheets for {T-273}C: ", matched_sheets[i])

            ref_no[i] = []
            for sheet in matched_sheets[i]:
                dict[sheet] = pd.read_excel(datafile, sheet)
                dict[sheet].dropna(subset=["P [MPa]"], inplace=True)
                ref_no[i].append(sheet[sheet.find("(") + 1 : sheet.find(")")])
            print(ref_no[i])

            ref_ID[i] = []
            ref_df = pd.read_excel(reffile, "references")
            # print(ref_df)
            for no in ref_no[i]:
                ref_ID[i].append(ref_df.loc[ref_df["# ref"] == f"[{no}]", "refID"].item())
                # print(ref_ID[i])
            print(ref_ID[i])
        except Exception as e:
            print("")
            print("Error - importing exp data failed:")
            print(e)
        # print(len(matched_sheets[i]))
        hasExpData[i] = True if len(matched_sheets[i]) > 0 else False
    print("hasExpData = ", hasExpData)
    
    # Empty array to store results
    solubility_EQ = [None for i in range(len(T_list))]
    solubility_NE = [None for i in range(len(T_list))]
    
    # Get pressure range
    if p_l != None and p_u != None:
        p_calc = np.linspace(p_l, p_u, no_p_points)  # [Pa]
    else:
        for i, T in enumerate(T_list):
            if hasExpData[i] == True:
                for j, sheet in enumerate(matched_sheets[i]):
                    current_max_p_MPa = np.asarray(dict[sheet]["P [MPa]"]).max()
                    if i == 0 and j == 0:
                        max_p_MPa = current_max_p_MPa
                    else:
                        max_p_MPa = max(current_max_p_MPa, max_p_MPa)
        
        max_p = max_p_MPa * 1e6 # [Pa]
        p_calc = np.linspace(1, max_p, no_p_points)    # [Pa]
        
    p_MPa_calc = p_calc * 1e-6  # [MPa]        
        
    for i, T in enumerate(T_list):
        
        #* Calculate EQ solubility
        if include_EQ == True:
            solubility_EQ[i] = [NE_SAFT.solve_solubility_EQ(T, p_, sol, pol, MW2) for p_ in p_calc]
            print("\nSolubility_EQ at %s°C: " % (T - 273), solubility_EQ[i])
        
        #* Calculate NE solubility
        if include_NE == True:
            solubility_NE[i] = [NE_SAFT.solve_solubility_NE(T, p_, sol, pol, MW2, ksw_list[i], rho20_list[i]) for p_ in p_calc]
            print("\nSolubility_NE at %s°C and ksw = %g:" % (T - 273, ksw_list[i]), solubility_NE[i])

    # Save data to dataframe
    data = {
        "T [°C]": np.repeat(np.array(T_list)-273, no_p_points),
        "p [MPa]": np.tile(p_MPa_calc, len(T_list)),
        "solubility_EQ [g-sol/g-pol]": np.concatenate(solubility_EQ) if include_EQ else [np.nan] * len(T_list) * no_p_points,
        "solubility_NE [g-sol/g-pol]": np.concatenate(solubility_NE) if include_NE else [np.nan] * len(T_list) * no_p_points,
    }
    
    df = pd.DataFrame(data)
    
    print(df)
    
    if save_data_dir != None:
        df.to_csv(save_data_dir, index=False)
        print(f"Data saved: {save_data_dir}")
        print("")
    
    # Variables to store plot objects
    curves = {}
    
    # Plotting
    if fig_size == None:
        fig = plt.figure()  # Default        
    elif isinstance(fig_size, tuple):
        fig = plt.figure(figsize=fig_size)  # Big plot        
    else:
        fig = plt.figure()  # Default
    
    ax = fig.add_subplot(111)

    for i, T in enumerate(T_list):
        curves[T] = []
        # Exp solubility
        if hasExpData[i] == True:
            for j, sheet in enumerate(matched_sheets[i]):
                curve_exp, = ax.plot(dict[sheet]['P [MPa]'],
                        dict[sheet]['Solubility [g-sol/g-pol-am]'],
                        color=NE_SAFT.custom_colours[i],
                        marker=NE_SAFT.custom_markers[i],
                        markersize=5,
                        linestyle='None',
                        markerfacecolor='None',
                        # label=f"{ref_ID[i][j]} [{paper_ref_dict[ref_ID[i][j]]}]",
                        )

                
                curves[T].append(curve_exp)
        
        # curve_empty =  Line2D([0], [0], 
        #                 color='white',
        #                 marker='None',
        #                 linestyle='solid',)
        # curves[T].append(curve_empty)
        # EQ calc
        if include_EQ == True:
            ax.plot(p_MPa_calc,
                    solubility_EQ[i],
                    color=NE_SAFT.custom_colours[i],
                    marker='None',
                    linestyle='solid',
                    # label=f"EQ model {T-273} °C",
                    )
            
            curve_EQ = Line2D([0], [0], 
                              color=NE_SAFT.custom_colours[i],
                              marker='None',
                              linestyle='solid',)
            
            curves[T].append(curve_EQ)

        # curves[T].append(curve_empty)
        # NE solubility    
        if include_NE == True:
            ax.plot(p_MPa_calc,
                    solubility_NE[i],
                    color=NE_SAFT.custom_colours[i],
                    marker='None',
                    linestyle='dashed', 
                    # label=f"NE model {T-273} °C",
                    )
            
            curve_NE = Line2D([0], [0], 
                    color=NE_SAFT.custom_colours[i],
                    marker='None',
                    linestyle='dashed', 
                    )
            curves[T].append(curve_NE)
            
    # Labelling
    ax.set_xlabel(r"p / MPa")
    ax.set_ylabel(r"Solubility $/$ $g_{s} \; g^{-1}_{p}$")
    # ax.set_title(r"%s-%s" % (sol, pol)) 
    
    # Update ticks to cover all data points
    update_subplot_ticks(ax,  x_lo=x_lo, x_up=x_up, y_lo=y_lo, y_up=y_up)
    
    # Dynamic column number of legend
    # ax.legend(loc=legend_loc).set_visible(True)
    
    # Add custom legend to the plot
    ax.legend([tuple(curves[T]) for T in T_list], 
              [f"{T-273} °C" for T in T_list],
              numpoints=1,
              handler_map={tuple: HandlerTuple(ndivide=None, pad=-0.2)},
            #   handletextpad=1,
            #   columnspacing=2.0,
            #   handlelength=3,
              loc=legend_loc).set_visible(True)
    
    if save_plot_dir != None:
        plt.savefig(save_plot_dir, dpi=1200)
        print(f"Plot saved: {save_plot_dir}")
        print("")
    
    if display_plot == True:
        plt.show()


def plot_isotherm_EQvNE_multiT_zoomed(
    T_list: list[float],
    ksw_list: list[float],
    rho20_list: list[float],
    sol: str,
    pol: str,
    p_l: float = None,
    p_u: float = None,
    no_p_points: int = 20,
    MW2: float = None,
    xlxs_sheet_refno_list: list[str] = None,
    include_EQ: bool = True,
    include_NE: bool = True,
    x_lo: float = None,
    x_up: float = None,
    y_lo: float = None,
    y_up: float = None,
    fig_size: tuple = None,
    display_plot: bool = True,
    save_plot_dir: str = None,
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
        raise ValueError("T_list, ksw_list and rho20_list must have the same length")
    
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
            path = os.path.join(os.path.dirname(__file__), "litdata")
            refpath = path + "/references.xlsx"
            databasepath = path + "/%s-%s.xlsx" % (sol, pol)
            reffile = pd.ExcelFile(refpath, engine="openpyxl")
            datafile = pd.ExcelFile(databasepath, engine="openpyxl")
            
            # Get all sheets matching T
            # print(file.sheet_names)
            matched_sheets[i] = []
            if xlxs_sheet_refno_list == None:
                search_pattern = f"^S_{T-273}C (.*)"  # strat with S_{T-273}C ()
                for sheet in datafile.sheet_names:
                    if re.search(search_pattern, sheet):
                        matched_sheets[i].append(sheet)
            elif isinstance(xlxs_sheet_refno_list, list):  # check if a list
                for j in xlxs_sheet_refno_list:
                    search_pattern = f"^S_{T-273}C.\({j}\)"
                    for sheet in datafile.sheet_names:
                        if re.search(search_pattern, sheet):
                            matched_sheets[i].append(sheet)

            print(f"Sheets for {T-273}C: ", matched_sheets[i])

            ref_no[i] = []
            for sheet in matched_sheets[i]:
                dict[sheet] = pd.read_excel(datafile, sheet)
                dict[sheet].dropna(subset=["P [MPa]"], inplace=True)
                ref_no[i].append(sheet[sheet.find("(") + 1 : sheet.find(")")])
            print(ref_no[i])

            ref_ID[i] = []
            ref_df = pd.read_excel(reffile, "references")
            # print(ref_df)
            for no in ref_no[i]:
                ref_ID[i].append(ref_df.loc[ref_df["# ref"] == f"[{no}]", "refID"].item())
                # print(ref_ID[i])
            print(ref_ID[i])
        except Exception as e:
            print("")
            print("Error - importing exp data failed:")
            print(e)
        # print(len(matched_sheets[i]))
        hasExpData[i] = True if len(matched_sheets[i]) > 0 else False
    print("hasExpData = ", hasExpData)
    
    # Empty array to store results
    solubility_EQ = [None for i in range(len(T_list))]
    solubility_NE = [None for i in range(len(T_list))]
    
    # Get pressure range
    if p_l != None and p_u != None:
        p_calc = np.linspace(p_l, p_u, no_p_points)  # [Pa]
    else:
        for i, T in enumerate(T_list):
            if hasExpData[i] == True:
                for j, sheet in enumerate(matched_sheets[i]):
                    current_max_p_MPa = np.asarray(dict[sheet]["P [MPa]"]).max()
                    if i == 0 and j == 0:
                        max_p_MPa = current_max_p_MPa
                    else:
                        max_p_MPa = max(current_max_p_MPa, max_p_MPa)
        
        max_p = max_p_MPa * 1e6 # [Pa]
        p_calc = np.linspace(1, max_p, no_p_points)    # [Pa]
        
    p_MPa_calc = p_calc * 1e-6  # [MPa]        
        
    for i, T in enumerate(T_list):
        
        #* Calculate EQ solubility
        if include_EQ == True:
            solubility_EQ[i] = [NE_SAFT.solve_solubility_EQ(T, p_, sol, pol, MW2) for p_ in p_calc]
            print("\nSolubility_EQ at %s°C: " % (T - 273), solubility_EQ[i])
        
        #* Calculate NE solubility
        if include_NE == True:
            solubility_NE[i] = [NE_SAFT.solve_solubility_NE(T, p_, sol, pol, MW2, ksw_list[i], rho20_list[i]) for p_ in p_calc]
            print("\nSolubility_NE at %s°C and ksw = %g:" % (T - 273, ksw_list[i]), solubility_NE[i])

    # Save data to dataframe
    data = {
        "T [°C]": np.repeat(np.array(T_list)-273, no_p_points),
        "p [MPa]": np.tile(p_MPa_calc, len(T_list)),
        "solubility_EQ [g-sol/g-pol]": np.concatenate(solubility_EQ) if include_EQ else [None] * len(T_list) * no_p_points,
        "solubility_NE [g-sol/g-pol]": np.concatenate(solubility_NE) if include_NE else [None] * len(T_list) * no_p_points,
    }
    
    df = pd.DataFrame(data)
    
    print(df)
    
    if save_data_dir != None:
        df.to_csv(save_data_dir, index=False)
        print(f"Data saved: {save_data_dir}")
        print("")
    
    # Plotting
    if fig_size == None:
        fig = plt.figure(figsize=(1.7, 1.5))  # Small plots
    elif isinstance(fig_size, tuple):
        fig = plt.figure(figsize=fig_size)  # Big plot    
    else:
        fig = plt.figure(figsize=(1.7, 1.5))  # Small plots
    
    ax = fig.add_subplot(111)

    for i, T in enumerate(T_list):
        
        # Exp solubility
        if hasExpData[i] == True:
            for j, sheet in enumerate(matched_sheets[i]):
                ax.plot(
                    dict[sheet]["P [MPa]"],
                    dict[sheet]["Solubility [g-sol/g-pol-am]"],
                    color=NE_SAFT.custom_colours[i],
                    marker=NE_SAFT.custom_markers[j],
                    # markersize=2,
                    linestyle="None",
                    markerfacecolor="None",
                    label=f"exp {T-273} °C: {ref_ID[i][j]}",
                )
                
        #* EQ calc
        if include_EQ == True:
            ax.plot(
                p_MPa_calc,
                solubility_EQ[i],
                color=NE_SAFT.custom_colours[i],
                marker="None",
                linestyle="dashed",
                label=f"EQ model {T-273} °C",
            )

        # NE solubility    
        if include_NE == True:
            ax.plot(
                p_MPa_calc,
                solubility_NE[i],
                color=NE_SAFT.custom_colours[i],
                marker="None",
                linestyle="solid", 
                label=f"NE model {T-273} °C",
            )
        
    # Labelling
    # ax.set_xlabel(r"p (MPa)")
    # ax.set_ylabel(r"Solubility ($g_{sol} \; / \;g_{pol}$)")
    # ax.set_title(r"%s-%s" % (sol, pol)) 
    
    # Update ticks to cover all data points
    update_subplot_ticks(ax,  x_lo=x_lo, x_up=x_up, y_lo=y_lo, y_up=y_up)
    
    # Dynamic column number of legend
    # ax.legend(loc='upper left').set_visible(True)
    
    if save_plot_dir != None:
        plt.savefig(save_plot_dir, dpi=1200, transparent=True)
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
            search_pattern = f"^S_{T-273}C (.*)"  # start with S_{T-273}C
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
    y_min = min(y_min, min(solubility_EQ_list))
    y_max = max(y_max, max(solubility_EQ_list))
    
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
    
@logger
def plot_isotherm_EQ_multiT(
    T_list: list[float],
    sol: str,
    pol: str,
    p_l: float = None,
    p_u: float = None,
    no_p_points: int = 20,
    MW2: float = None,
    xlxs_sheet_refno_list: list[str] = None,
    display_plot: bool = True,
    save_plot_dir: str = None,
):
    (
        _eos_mix,
        _eos_sol,
        _MW_1,
        _MW_2,
        _MW_monomer,
        rho_2_am_dry,
        _k_sw,
    ) = NE_SAFT.get_mixture_info(sol, pol, MW2)
    MW2 = MW2 if MW2 != None else _MW_2
    
    hasExpData = [None for i in range(len(T_list))]
    matched_sheets = [None for i in range(len(T_list))]
    ref_no = [None for i in range(len(T_list))]
    ref_ID = [None for i in range(len(T_list))]
    dict = {}  # dictionary of all matched sheet df
    
    # Import exp data
    for i, T in enumerate(T_list):
        try:
            # Read exp file
            path = os.path.join(os.path.dirname(__file__), "litdata")
            refpath = path + "/references.xlsx"
            databasepath = path + "/%s-%s.xlsx" % (sol, pol)
            reffile = pd.ExcelFile(refpath, engine="openpyxl")
            datafile = pd.ExcelFile(databasepath, engine="openpyxl")
            
            # Get all sheets matching T
            # print(file.sheet_names)
            matched_sheets[i] = []
            if xlxs_sheet_refno_list == None:
                search_pattern = f"^S_{T-273}C (.*)"  # strat with S_{T-273}C ()
                for sheet in datafile.sheet_names:
                    if re.search(search_pattern, sheet):
                        matched_sheets[i].append(sheet)
            elif isinstance(xlxs_sheet_refno_list, list):  # check if a list
                for j in xlxs_sheet_refno_list:
                    search_pattern = f"^S_{T-273}C.\({j}\)"
                    for sheet in datafile.sheet_names:
                        if re.search(search_pattern, sheet):
                            matched_sheets[i].append(sheet)

            print(f"Sheets for {T-273}C: ", matched_sheets[i])

            ref_no[i] = []
            for sheet in matched_sheets[i]:
                dict[sheet] = pd.read_excel(datafile, sheet)
                dict[sheet].dropna(subset=["P [MPa]"], inplace=True)
                ref_no[i].append(sheet[sheet.find("(") + 1 : sheet.find(")")])
            print(ref_no[i])

            ref_ID[i] = []
            ref_df = pd.read_excel(reffile, "references")
            # print(ref_df)
            for no in ref_no[i]:
                ref_ID[i].append(ref_df.loc[ref_df["# ref"] == f"[{no}]", "refID"].item())
                # print(ref_ID[i])
            print(ref_ID[i])
        except Exception as e:
            print("")
            print("Error - importing exp data failed:")
            print(e)
        # print(len(matched_sheets[i]))
        hasExpData[i] = True if len(matched_sheets[i]) > 0 else False
    print("hasExpData = ", hasExpData)
    
    # Empty array to store results
    # solubility_EQ = [None for i in range(len(T_list))]
    
    # Get pressure range
    if p_l != None and p_u != None:
        p_calc = np.linspace(p_l, p_u, no_p_points)  # [Pa]
    else:
        for i, T in enumerate(T_list):
            if hasExpData[i] == True:
                for j, sheet in enumerate(matched_sheets[i]):
                    current_max_p_MPa = np.asarray(dict[sheet]["P [MPa]"]).max()
                    if i == 0 and j == 0:
                        max_p_MPa = current_max_p_MPa
                    else:
                        max_p_MPa = max(current_max_p_MPa, max_p_MPa)
        
        max_p = max_p_MPa * 1e6 # [Pa]
        p_calc = np.linspace(1, max_p, no_p_points)    # [Pa]
        p_MPa_calc = p_calc * 1e-6  # [MPa]        
    
    for i, T in enumerate(T_list):
        # Calculate EQ solubility
        # solubility_EQ[i] = [NE_SAFT.solve_solubility_EQ(T, p_, sol, pol, MW2) for p_ in p_calc]
        # print("\nsolubility_EQ at %s°C = " % (T - 273), solubility_EQ[i])
        continue

    # Calculate upper limits of x and y axis    
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    for i, T in enumerate(T_list):
        for j, sheet in enumerate(matched_sheets[i]):
            x_min = min(x_min, min(dict[sheet]["P [MPa]"]))
            x_max = max(x_max, max(dict[sheet]["P [MPa]"]))
            y_min = min(y_min, min(dict[sheet]["Solubility [g-sol/g-pol-am]"]))
            y_max = max(y_max, max(dict[sheet]["Solubility [g-sol/g-pol-am]"]))
        # y_min = min(y_min, min(solubility_EQ[i]))
        # y_max = max(y_max, max(solubility_EQ[i]))
    
    # Plotting
    fig = plt.figure()  # Default
    # fig = plt.figure(figsize=(5.,4.5))  # Big plot
    # fig = plt.figure(figsize=(3., 2.5))  # For side-by-side plots in paper
    ax = fig.add_subplot(111)
    nlegendcount = 0
    for i, T in enumerate(T_list):
        
        # EQ calc
        # ax.plot(
        #     p_MPa_calc,
        #     solubility_EQ[i],
        #     color=NE_SAFT.custom_colours[i],
        #     marker="None",
        #     linestyle="solid",
        #     label=f"EQ model {T-273} °C",
        # )
        # nlegendcount += 1
        
        # Exp data
        if hasExpData[i] == True:
            for j, sheet in enumerate(matched_sheets[i]):
                ax.plot(
                    dict[sheet]["P [MPa]"],
                    dict[sheet]["Solubility [g-sol/g-pol-am]"],
                    color=NE_SAFT.custom_colours[i],
                    marker=NE_SAFT.custom_markers[j],
                    linestyle="None",
                    markerfacecolor="None",
                    label=f"exp {T-273} °C: {ref_ID[i][j]}",
                )
                nlegendcount += 1

    # Labelling
    # ax.set_xlabel(r"p (MPa)")
    # ax.set_ylabel(r"Solubility ($g_{sol} \; / \;g_{pol}$)"12)
    # ax.set_title(r"%s-%s" % (sol, pol))
    
    # Adjust x and y tick to cover all data
    # Get the length of major ticks on the x-axis
    x_major_tick_length = ax.get_xticks()[1] - ax.get_xticks()[0]
    
    # Get the length of major ticks on the y-axis
    y_major_tick_length = ax.get_yticks()[1] - ax.get_yticks()[0]
    
    # Set adjust x and y tick to cover all data
    ax.set_xlim(left=0, right=x_max + x_major_tick_length)
    # ax.set_ylim(bottom=0, top=y_max + y_major_tick_length)  # Default    
    ax.set_ylim(bottom=0, top=y_max + 2*y_major_tick_length)  # Paper plot

    ax.set_xlim(left=0, right=5)
    ax.set_ylim(bottom=0, top=0.14)
    # Set ticks to appear inside
    ax.tick_params(direction="in")
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    # Dynamic column number of legend
    legend_ncol = 1 if (len(matched_sheets)) < 5 else 2
    # ax.legend(fontsize='xx-small', loc='upper left').set_visible(True)
    
    if save_plot_dir != None:
        plt.savefig(save_plot_dir, dpi=1200)
        print(f"Plot saved: {save_plot_dir}")
        print("")
    if display_plot == True:
        plt.show()

def plot_density_EQvNE_multiT(
    T_list: list[float],
    ksw_list: list[float],
    rho20_list: list[float],
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
        _rho_2_am_dry,
        _k_sw,
    ) = NE_SAFT.get_mixture_info(sol, pol, MW2)    
    MW2 = MW2 if MW2 != None else _MW_2  # [g/mol]

    # Check T_list, ksw_list and rho20_list have the same length
    if len(T_list) != len(ksw_list) or len(T_list) != len(rho20_list):
        raise ValueError("T_list, ksw_list and rho20_list must have the same length")

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
            path = os.path.join(os.path.dirname(__file__), "litdata")
            refpath = path + "/references.xlsx"
            databasepath = path + "/%s-%s.xlsx" % (sol, pol)
            reffile = pd.ExcelFile(refpath, engine="openpyxl")
            datafile = pd.ExcelFile(databasepath, engine="openpyxl")
            
            # Get all sheets matching T
            # print(file.sheet_names)
            matched_sheets[i] = []
            if xlxs_sheet_refno_list == None:
                search_pattern = f"^S_{T-273}C (.*)"  # strat with S_{T-273}C ()
                for sheet in datafile.sheet_names:
                    if re.search(search_pattern, sheet):
                        matched_sheets[i].append(sheet)
            elif isinstance(xlxs_sheet_refno_list, list):  # check if a list
                for j in xlxs_sheet_refno_list:
                    search_pattern = f"^S_{T-273}C.\({j}\)"
                    for sheet in datafile.sheet_names:
                        if re.search(search_pattern, sheet):
                            matched_sheets[i].append(sheet)

            print(f"Sheets for {T-273}C: ", matched_sheets[i])

            ref_no[i] = []
            for sheet in matched_sheets[i]:
                dict[sheet] = pd.read_excel(datafile, sheet)
                dict[sheet].dropna(subset=["P [MPa]"], inplace=True)
                ref_no[i].append(sheet[sheet.find("(") + 1 : sheet.find(")")])
            print(ref_no[i])

            ref_ID[i] = []
            ref_df = pd.read_excel(reffile, "references")
            # print(ref_df)
            for no in ref_no[i]:
                ref_ID[i].append(ref_df.loc[ref_df["# ref"] == f"[{no}]", "refID"].item())
                # print(ref_ID[i])
            print(ref_ID[i])
        except Exception as e:
            print("")
            print("Error - importing exp data failed:")
            print(e)
        # print(len(matched_sheets[i]))
        hasExpData[i] = True if len(matched_sheets[i]) > 0 else False
    print("hasExpData = ", hasExpData)
        
    # Empty array to store results
    rho_EQ = [None for i in range(len(T_list))]
    rho_NE = [None for i in range(len(T_list))]
    
    # Get pressure range
    if p_l != None and p_u != None:
        p_calc = np.linspace(p_l, p_u, no_p_points)  # [Pa]
    else:
        for i, T in enumerate(T_list):
            if hasExpData[i] == True:
                for j, sheet in enumerate(matched_sheets[i]):
                    current_max_p_MPa = np.asarray(dict[sheet]["P [MPa]"]).max()
                    if i == 0 and j == 0:
                        max_p_MPa = current_max_p_MPa
                    else:
                        max_p_MPa = max(current_max_p_MPa, max_p_MPa)
        
        max_p = max_p_MPa * 1e6 # [Pa]
        p_calc = np.linspace(1, max_p, no_p_points)    # [Pa]
    
    p_MPa_calc = p_calc * 1e-6  # [MPa]
    
    for i, T in enumerate(T_list):
        # Calculate EQ density
        rho_EQ[i] = [NE_SAFT.solve_solubility_EQ(T, p_, sol, pol, MW2, return_extended=True)[3] for p_ in p_calc]
        print("\nDensity_EQ at %s°C: " % (T - 273), rho_EQ[i])
        
        # Calculate NE density
        rho_NE[i] = [NE_SAFT.solve_solubility_NE(T, p_, sol, pol, MW2, ksw_list[i], rho20_list[i], return_extended=True)[3] for p_ in p_calc]
        print("\nDensity_NE at %s°C and ksw=%g: " % (T - 273, ksw_list[i]), rho_NE[i])
        
    # Calculate upper limits of x and y axis
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    for i, T in enumerate(T_list):
        for j, sheet in enumerate(matched_sheets[i]):
            x_min = min(x_min, min(dict[sheet]["P [MPa]"]))
            x_max = max(x_max, max(dict[sheet]["P [MPa]"]))
            
        y_min = min(y_min, min(rho_EQ[i]))
        y_min = min(y_min, min(rho_NE[i]))
        y_max = max(y_max, max(rho_EQ[i]))
        y_max = max(y_max, max(rho_NE[i]))

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i, T in enumerate(T_list):                
        # EQ calc
        ax.plot(
            p_MPa_calc,
            rho_EQ[i],
            color=NE_SAFT.custom_colours[i],
            marker='None',            
            linestyle='solid',
            label=f'EQ model {T-273} °C',
        )

        # NE solubility    
        ax.plot(
            p_MPa_calc,
            rho_NE[i],
            color=NE_SAFT.custom_colours[i],
            marker='None',
            linestyle='dashed',
            label=f'NE model {T-273} °C',
        )

        # Labelling
        ax.set_xlabel(r'p (MPa)')
        ax.set_ylabel(r'$\rho_{pol}$ ($g \; / \; cm^{-3}$)')
        ax.set_title(r'%s-%s' % (sol, pol))
    # Adjust x and y tick to cover all data
    # Get the length of major ticks on the x-axis
    x_major_tick_length = ax.get_xticks()[1] - ax.get_xticks()[0]
    
    # Get the length of major ticks on the y-axis
    y_major_tick_length = ax.get_yticks()[1] - ax.get_yticks()[0]
    
    # Set adjust x and y tick to cover all data
    ax.set_xlim(left=0, right=x_max + x_major_tick_length)
    # ax.set_ylim(bottom=y_min - y_major_tick_length, top=y_max + y_major_tick_length)  # Default
    ax.set_ylim(bottom=y_min - y_major_tick_length, top=y_max + 2*y_major_tick_length)  # Paper plot 

    # Set ticks to appear inside
    ax.tick_params(direction="in")
    
    # Dynamic column number of legend
    ax.legend(fontsize='xx-small').set_visible(True)    
    
    if save_plot_dir != None:
        plt.savefig(save_plot_dir, dpi=1200)
        print(f"Plot saved: {save_plot_dir}")
        print("")
        
    if display_plot == True:
        plt.show()

def plot_isotherm_density_EQvNE_multiT(
    T_list: list[float],
    ksw_list: list[float],
    rho20_list: list[float],
    sol: str,
    pol: str,
    p_l: float = None,
    p_u: float = None,
    no_p_points: int = 20,
    MW2: float = None,
    xlxs_sheet_refno_list: list[str] = None,
    include_EQ: bool = True,
    include_NE: bool = True,
    display_plot: bool = True,
    save_plot_dir: str = None,
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
        raise ValueError("T_list, ksw_list and rho20_list must have the same length")
    
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
            path = os.path.join(os.path.dirname(__file__), "litdata")
            refpath = path + "/references.xlsx"
            databasepath = path + "/%s-%s.xlsx" % (sol, pol)
            reffile = pd.ExcelFile(refpath, engine="openpyxl")
            datafile = pd.ExcelFile(databasepath, engine="openpyxl")
            
            # Get all sheets matching T
            # print(file.sheet_names)
            matched_sheets[i] = []
            if xlxs_sheet_refno_list == None:
                search_pattern = f"^S_{T-273}C (.*)"  # strat with S_{T-273}C ()
                for sheet in datafile.sheet_names:
                    if re.search(search_pattern, sheet):
                        matched_sheets[i].append(sheet)
            elif isinstance(xlxs_sheet_refno_list, list):  # check if a list
                for j in xlxs_sheet_refno_list:
                    search_pattern = f"^S_{T-273}C.\({j}\)"
                    for sheet in datafile.sheet_names:
                        if re.search(search_pattern, sheet):
                            matched_sheets[i].append(sheet)

            print(f"Sheets for {T-273}C: ", matched_sheets[i])

            ref_no[i] = []
            for sheet in matched_sheets[i]:
                dict[sheet] = pd.read_excel(datafile, sheet)
                dict[sheet].dropna(subset=["P [MPa]"], inplace=True)
                ref_no[i].append(sheet[sheet.find("(") + 1 : sheet.find(")")])
            print(ref_no[i])

            ref_ID[i] = []
            ref_df = pd.read_excel(reffile, "references")
            # print(ref_df)
            for no in ref_no[i]:
                ref_ID[i].append(ref_df.loc[ref_df["# ref"] == f"[{no}]", "refID"].item())
                # print(ref_ID[i])
            print(ref_ID[i])
        except Exception as e:
            print("")
            print("Error - importing exp data failed:")
            print(e)
        # print(len(matched_sheets[i]))
        hasExpData[i] = True if len(matched_sheets[i]) > 0 else False
    print("hasExpData = ", hasExpData)
    
    # Empty array to store results
    solubility_EQ = [None for i in range(len(T_list))]
    solubility_NE = [None for i in range(len(T_list))]
    rho_EQ = [None for i in range(len(T_list))]
    rho_NE = [None for i in range(len(T_list))]
    
    # Get pressure range
    if p_l != None and p_u != None:
        p_calc = np.linspace(p_l, p_u, no_p_points)  # [Pa]
    else:
        for i, T in enumerate(T_list):
            if hasExpData[i] == True:
                for j, sheet in enumerate(matched_sheets[i]):
                    current_max_p_MPa = np.asarray(dict[sheet]["P [MPa]"]).max()
                    if i == 0 and j == 0:
                        max_p_MPa = current_max_p_MPa
                    else:
                        max_p_MPa = max(current_max_p_MPa, max_p_MPa)
        
        max_p = max_p_MPa * 1e6 # [Pa]
        p_calc = np.linspace(1, max_p, no_p_points)    # [Pa]
        
    p_MPa_calc = p_calc * 1e-6  # [MPa]        
        
    for i, T in enumerate(T_list):
        
        #* Calculate EQ solubility
        if include_EQ == True:
            # EQ solubility
            solubility_EQ[i] = [NE_SAFT.solve_solubility_EQ(T, p_, sol, pol, MW2) for p_ in p_calc]
            print("\nSolubility_EQ at %s°C: " % (T - 273), solubility_EQ[i])
            
            # EQ polymer partial density
            rho_EQ[i] = [NE_SAFT.solve_solubility_EQ(T, p_, sol, pol, MW2, return_extended=True)[3] for p_ in p_calc]
            print("\nDensity_EQ at %s°C: " % (T - 273), rho_EQ[i])
        
        #* Calculate NE solubility
        if include_NE == True:
            # NE solubility
            solubility_NE[i] = [NE_SAFT.solve_solubility_NE(T, p_, sol, pol, MW2, ksw_list[i], rho20_list[i]) for p_ in p_calc]
            print("\nSolubility_NE at %s°C and ksw = %g:" % (T - 273, ksw_list[i]), solubility_NE[i])
            
            # NE polymer partial density
            rho_NE[i] = [NE_SAFT.solve_solubility_NE(T, p_, sol, pol, MW2, ksw_list[i], rho20_list[i], return_extended=True)[3] for p_ in p_calc]
            print("\nDensity_NE at %s°C and ksw=%g: " % (T - 273, ksw_list[i]), rho_NE[i])

    # Save data to dataframe
    data = {
        "T [°C]": np.repeat(np.array(T_list)-273, no_p_points),
        "p [MPa]": np.tile(p_MPa_calc, len(T_list)),
        "solubility_EQ [g-sol/g-pol]": np.concatenate(solubility_EQ) if include_EQ else [None] * len(T_list) * no_p_points,
        "solubility_NE [g-sol/g-pol]": np.concatenate(solubility_NE) if include_NE else [None] * len(T_list) * no_p_points,
        "rho_pol_EQ [g-pol/cm3-mix]": np.concatenate(rho_EQ) if include_EQ else [None] * len(T_list) * no_p_points,
        "rho_pol_NE [g-pol/cm3-mix]": np.concatenate(rho_NE) if include_NE else [None] * len(T_list) * no_p_points,
    }
    
    df = pd.DataFrame(data)
    
    print(df)
    
    if save_data_dir != None:
        df.to_csv(save_data_dir, index=False)
        print(f"Data saved: {save_data_dir}")
        print("")
    
    # Create a figure with 2 subplots stacked vertically, sharing the same x-axis
    fig = plt.figure(figsize=(5.0, 9.0))  # Big plot
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    # Plot solubility
    for i, T in enumerate(T_list):
        # Exp solubility
        if hasExpData[i] == True:
            for j, sheet in enumerate(matched_sheets[i]):
                ax1.plot(
                    dict[sheet]['P [MPa]'],
                    dict[sheet]['Solubility [g-sol/g-pol-am]'],
                    color=NE_SAFT.custom_colours[i],
                    marker=NE_SAFT.custom_markers[j],
                    # markersize=2,
                    linestyle='None',
                    markerfacecolor='None',
                    label=f"exp {T-273} °C: {ref_ID[i][j]}",
                )
                
        # EQ solubility
        if include_EQ == True:
            ax1.plot(
                p_MPa_calc,
                solubility_EQ[i],
                color=NE_SAFT.custom_colours[i],
                marker='None',
                linestyle='solid',
                label=f"EQ model {T-273} °C",
            )

        # NE solubility    
        if include_NE == True:
            ax1.plot(
                p_MPa_calc,
                solubility_NE[i],
                color=NE_SAFT.custom_colours[i],
                marker='None',
                linestyle='dashed', 
                label=f"NE model {T-273} °C",
            )
        
    # Labelling
    ax1.set_xlabel(r'p (MPa)')
    ax1.set_ylabel(r'Solubility $/$ $g_{s}  / g_{p}$')
    # ax.set_title(r"%s-%s" % (sol, pol)) 
    # Update ticks to cover all data points
    update_subplot_ticks(ax1, x_lo=0., y_lo=0.)
    
    # Dynamic column number of legend
    ax1.legend(loc='upper left').set_visible(True)
    
    # Plot partial polymer density
    for i, T in enumerate(T_list):                
        # EQ calc
        ax2.plot(
            p_MPa_calc,
            rho_EQ[i],
            color=NE_SAFT.custom_colours[i],
            marker="None",            
            linestyle="dashed",
            label=f"EQ model {T-273} °C",
        )

    # NE solubility    
        ax2.plot(
            p_MPa_calc,
            rho_NE[i],
            color=NE_SAFT.custom_colours[i],
            marker="None",
            linestyle="solid",
            label=f"NE model {T-273} °C",
        )

    # Labelling
    ax2.set_xlabel(r"p (MPa)")
    ax2.set_ylabel(r"$\rho_{pol}$ $/$ $g_{pol}  /  cm^{3}_{mix}$")
    # ax2.set_title(r"%s-%s" % (sol, pol))
    
    # Hide x-ticks for the first plot to prevent overlap
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    # Set the same x-ticks for both plots
    ax2.set_xticks(ax1.get_xticks())
    
    # Show ticks inside
    ax1.tick_params(axis='both', direction='in')
    ax2.tick_params(axis='both', direction='in')

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
    result_folder_dir = f'{src_dir}\\Anals\\Paper plots'
    
    # ksw_base = 0.008293414047454917
    # plot_isotherm_EQvNE(
    #     # p_l=1,
    #     # p_u=25e6,
    #     no_p_points=10,
    #     T=35+273,
    #     sol="CO2",
    #     pol="PS",
    #     # ksw_list=[ksw_base*0.90, ksw_base*0.95, ksw_base, ksw_base*1.05, ksw_base*1.10],
    #     ksw_list=[ksw_base],
    #     rho20=1.0418247536493246,
    #     xlxs_sheet_refno_list=["8"],
    #     display_plot=True,
    #     # save_plot_dir=savedir,
    # )
    
    # Fit rho20
    # rho20_f, rho20_AADpc_f = NE_SAFT.fit_rho20_NE(
    #     T=81+273,
    #     sol='CO2',
    #     pol='PS',
    #     xlxs_sheet_refno='8',
    #     rho20_x0_list=np.linspace(0.90, 1.30, 30),
    #     # display_plot=True,
    #     # save_plot_dir=savedir,
    # )
    
    # rho20_35C = 1/NE_SAFT.get_V20_multiTait(T=35+273, p=0, pol="PMMA")
    # rho20_51C = 1/NE_SAFT.get_V20_multiTait(T=51+273, p=0, pol="PMMA")
    # rho20_81C = 1/NE_SAFT.get_V20_multiTait(T=81+273, p=0, pol="PMMA")
    # rho20_35C = 1.178
    # rho20_51C = 1.174
    # rho20_81C = 1.165
    
    # Predicted ksw
    # ksw_35C = NE_SAFT.predict_ksw_NE(T=35+273, sol="CO2", pol="PMMA", rho20=rho20_35C)
    # ksw_51C = NE_SAFT.predict_ksw_NE(T=51+273, sol="CO2", pol="PMMA", rho20=rho20_51C)
    # ksw_81C = NE_SAFT.predict_ksw_NE(T=81+273, sol="CO2", pol="PMMA", rho20=rho20_81C)
    # ksw_35C = 0.0174503
    # ksw_51C = 0.0138055
    # ksw_81C = 0.0087252
    
    # Fitted ksw
    # ksw_35C, AADpc_35C = NE_SAFT.fit_ksw_NE(T=35+273, sol="CO2", pol="PMMA", xlxs_sheet_refno="8", rho20=rho20_35C, ksw_x0_list=np.linspace(0.005,0.01,50), display_plot=False)
    # ksw_51C, AADpc_51C = NE_SAFT.fit_ksw_NE(T=51+273, sol="CO2", pol="PMMA", xlxs_sheet_refno="8", rho20=rho20_51C, ksw_x0_list=np.linspace(0.005,0.01,50), display_plot=False)
    # ksw_81C, AADpc_81C = NE_SAFT.fit_ksw_NE(T=81+273, sol="CO2", pol="PMMA", xlxs_sheet_refno="8", rho20=rho20_81C, ksw_x0_list=np.linspace(0.005,0.01,50), display_plot=False)
    # print(f'35°C, rho20 = {rho20_35C}, ksw = {ksw_35C}')
    # print(f'51°C, rho20 = {rho20_51C}, ksw = {ksw_51C}')
    # print(f'81°C, rho20 = {rho20_81C}, ksw = {ksw_81C}')
    
    # Get AAD for NE
    # plot_isotherm_EQvNE(
    #     # p_l=1,
    #     # p_u=25e6,
    #     no_p_points=5,
    #     T=81+273,
    #     sol="CO2",
    #     pol="PMMA",
    #     ksw_list=[0.0087252],
    #     rho20=rho20_81C,
    #     xlxs_sheet_refno_list=["8"],
    #     display_plot=False,
    #     # save_plot_dir=savedir,
    # )
    
    #* Plot EOS vs. NET-GP results
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
    # PMMA from PVT
    
    # polymer = 'PS'
    # polymer = 'PMMA'
    
    #* 35, 51, 81 °C
    # plot_isotherm_EQvNE_multiT(
    #     # p_l=1,
    #     # p_u=25e6,
    #     no_p_points=60,
    #     T_list=[35+273, 51+273, 81+273],
    #     sol="CO2",
    #     pol=polymer,
    #     rho20_list=[rho20_35C[polymer], rho20_51C[polymer], rho20_81C[polymer]],
    #     ksw_list=[ksw_35C[polymer], ksw_51C[polymer], ksw_81C[polymer]],
    #     xlxs_sheet_refno_list=['8'],
    #     include_NE=True,
    #     include_EQ=True,
    #     display_plot=True,
    #     save_plot_dir=result_folder_dir + f'\\CO2-{polymer}_35-51-81C_fittedEoSParameters_kswFugacity_EoSvsNETGP_{time_ID}.png',
    # )
    
    # polymer = 'PMMA'
    # parameterType = 'fitted'
    # plot_isotherm_EQvNE_multiT(
    #     # p_l=1,
    #     # p_u=4e6,
    #     no_p_points=5,
    #     T_list=[81+273],
    #     sol="CO2",
    #     pol=polymer,
    #     rho20_list=[rho20_81C[polymer]],
    #     ksw_list=[ksw_81C[polymer][parameterType]],
    #     xlxs_sheet_refno_list=['8'],
    #     include_NE=True,
    #     include_EQ=True,
    #     display_plot=True,
    #     # save_plot_dir=result_folder_dir + f'\\CO2-{polymer}_35C_{parameterType}EoSParameters_kswFugacity_EQvsNE_{time_ID}.svg',
    # )
    
    #* Plot solubility isotherm
    # eos_parameter_type = 'fitted'
    # for polymer in ['PS', 'PMMA']:
    #     plot_isotherm_EQvNE_multiT(
    #         # p_l=1,
    #         # p_u=25e6,
    #         no_p_points=60,
    #         T_list=[35+273],
    #         sol="CO2",
    #         pol=polymer,
    #         rho20_list=[rho20_35C[polymer]],
    #         ksw_list=[ksw_35C[polymer][eos_parameter_type]],
    #         xlxs_sheet_refno_list=['8'],
    #         # include_NE=True,
    #         include_EQ=True,
    #         display_plot=False,
    #         save_plot_dir=result_folder_dir + f'\\CO2-{polymer}_35C_{eos_parameter_type}EoSParameters_kswFugacity_EQvsNE_{time_ID}.svg',
    #         # save_data_dir=f'Results\\CO2-{polymer}_35C_{eos_parameter_type}EoSParameters_kswFugacity_EQvsNE_{time_ID}.csv',
    #     )    
    #     plot_isotherm_EQvNE_multiT(
    #         # p_l=1,
    #         # p_u=25e6,
    #         no_p_points=60,
    #         T_list=[51+273],
    #         sol="CO2",
    #         pol=polymer,
    #         rho20_list=[rho20_51C[polymer]],
    #         ksw_list=[ksw_51C[polymer][eos_parameter_type]],
    #         xlxs_sheet_refno_list=['8'],
    #         # include_NE=True,
    #         include_EQ=True,
    #         display_plot=False,
    #         save_plot_dir=result_folder_dir + f'\\CO2-{polymer}_51C_{eos_parameter_type}EoSParameters_kswFugacity_EQvsNE_{time_ID}.svg',
    #         # save_data_dir=f'Results\\CO2-{polymer}_51C_{eos_parameter_type}EoSParameters_kswFugacity_EQvsNE_{time_ID}.csv',
    #     )    
    #     plot_isotherm_EQvNE_multiT(
    #         # p_l=1,
    #         # p_u=25e6,
    #         no_p_points=60,
    #         T_list=[81+273],
    #         sol="CO2",
    #         pol=polymer,
    #         rho20_list=[rho20_81C[polymer]],
    #         ksw_list=[ksw_81C[polymer][eos_parameter_type]],
    #         xlxs_sheet_refno_list=['8'],
    #         # include_NE=True,
    #         include_EQ=True,
    #         display_plot=False,
    #         save_plot_dir=result_folder_dir + f'\\CO2-{polymer}_81C_{eos_parameter_type}EoSParameters_kswFugacity_EQvsNE_{time_ID}.svg',
    #         # save_data_dir=f'Results\\CO2-{polymer}_81C_{eos_parameter_type}EoSParameters_kswFugacity_EQvsNE_{time_ID}.csv',
    #     )

    #* Plot solubility isotherm and partial polymer density
    # eos_parameter_type = 'fitted'
    # for polymer in ['PS', 'PMMA']:
    #     plot_isotherm_density_EQvNE_multiT(
    #         # p_l=1,
    #         # p_u=25e6,
    #         no_p_points=60,
    #         T_list=[35+273],
    #         sol='CO2',
    #         pol=polymer,
    #         rho20_list=[rho20_35C[polymer]],
    #         ksw_list=[ksw_35C[polymer][eos_parameter_type]],
    #         xlxs_sheet_refno_list=['8'],
    #         include_NE=True,
    #         include_EQ=True,
    #         display_plot=False,
    #         save_plot_dir=result_folder_dir + f'\\CO2-{polymer}_35C_{eos_parameter_type}EoSParameters_kswFugacity_solubilty-density_EQvsNE_{time_ID}.svg',
    #         save_data_dir=f'Results\\CO2-{polymer}_35C_{eos_parameter_type}EoSParameters_kswFugacity_solubilty-density_EQvsNE_{time_ID}.csv',
    #     )
    #     plot_isotherm_density_EQvNE_multiT(
    #         # p_l=1,
    #         # p_u=25e6,
    #         no_p_points=60,
    #         T_list=[51+273],
    #         sol='CO2',
    #         pol=polymer,
    #         rho20_list=[rho20_51C[polymer]],
    #         ksw_list=[ksw_51C[polymer][eos_parameter_type]],
    #         xlxs_sheet_refno_list=['8'],
    #         include_NE=True,
    #         include_EQ=True,
    #         display_plot=False,
    #         save_plot_dir=result_folder_dir + f'\\CO2-{polymer}_51C_{eos_parameter_type}EoSParameters_kswFugacity_solubilty-density_EQvsNE_{time_ID}.svg',
    #         save_data_dir=f'Results\\CO2-{polymer}_51C_{eos_parameter_type}EoSParameters_kswFugacity_solubilty-density_EQvsNE_{time_ID}.csv',
    #     )    
    #     plot_isotherm_density_EQvNE_multiT(
    #         # p_l=1,
    #         # p_u=25e6,
    #         no_p_points=60,
    #         T_list=[81+273],
    #         sol='CO2',
    #         pol=polymer,
    #         rho20_list=[rho20_81C[polymer]],
    #         ksw_list=[ksw_81C[polymer][eos_parameter_type]],
    #         xlxs_sheet_refno_list=['8'],
    #         include_NE=True,
    #         include_EQ=True,
    #         display_plot=False,
    #         save_plot_dir=result_folder_dir + f'\\CO2-{polymer}_81C_{eos_parameter_type}EoSParameters_kswFugacity_solubilty-density_EQvsNE_{time_ID}.svg',
    #         save_data_dir=f'Results\\CO2-{polymer}_81C_{eos_parameter_type}EoSParameters_kswFugacity_EQvsNE_{time_ID}.csv',
    #     )

    #* Plot zooomed solubility isotherm 
    # eos_parameter_type = 'fitted'
    # for polymer in ['PMMA']:
    #     plot_isotherm_EQvNE_multiT_zoomed(
    #         p_l=1,
    #         p_u=5e6,
    #         no_p_points=10,
    #         T_list=[35+273],
    #         sol='CO2',
    #         pol=polymer,
    #         rho20_list=[rho20_35C[polymer]],
    #         ksw_list=[ksw_35C[polymer][eos_parameter_type]],
    #         xlxs_sheet_refno_list=['8'],
    #         include_NE=True,
    #         include_EQ=True,
    #         x_lo=0.0,
    #         x_up=4.0,
    #         y_lo=0.0,
    #         y_up=0.15,
    #         display_plot=False,
    #         save_plot_dir=f'Anals/Paper plots/CO2-{polymer}_35C_{eos_parameter_type}EoSParameters_kswFugacity_solubilty_EQvsNE_zoomed_{time_ID}.svg',
    #         save_data_dir=f'Results/CO2-{polymer}_35C_{eos_parameter_type}EoSParameters_kswFugacity_solubilty_EQvsNE_{time_ID}.csv',
    #     )
    #     plot_isotherm_EQvNE_multiT_zoomed(
    #         p_l=1,
    #         p_u=5e6,
    #         no_p_points=10,
    #         T_list=[51+273],
    #         sol='CO2',
    #         pol=polymer,
    #         rho20_list=[rho20_51C[polymer]],
    #         ksw_list=[ksw_51C[polymer][eos_parameter_type]],
    #         xlxs_sheet_refno_list=['8'],
    #         include_NE=True,
    #         include_EQ=True,
    #         x_lo=0.0,
    #         x_up=4.0,
    #         y_lo=0.0,
    #         y_up=0.10,
    #         display_plot=False,
    #         save_plot_dir=f'Anals/Paper plots/CO2-{polymer}_51C_{eos_parameter_type}EoSParameters_kswFugacity_solubilty_EQvsNE_zoomed_{time_ID}.svg',
    #         save_data_dir=f'Results/CO2-{polymer}_51C_{eos_parameter_type}EoSParameters_kswFugacity_solubilty_EQvsNE_{time_ID}.csv',
    #     )    
    #     plot_isotherm_EQvNE_multiT_zoomed(
    #         p_l=1,
    #         p_u=7e6,
    #         no_p_points=10,
    #         T_list=[81+273],
    #         sol='CO2',
    #         pol=polymer,
    #         rho20_list=[rho20_81C[polymer]],
    #         ksw_list=[ksw_81C[polymer][eos_parameter_type]],
    #         xlxs_sheet_refno_list=['8'],
    #         include_NE=True,
    #         include_EQ=True,
    #         x_lo=0.0,
    #         x_up=6.0,
    #         y_lo=0.0,
    #         y_up=0.06,
    #         display_plot=False,
    #         save_plot_dir=f'Anals/Paper plots/CO2-{polymer}_81C_{eos_parameter_type}EoSParameters_kswFugacity_solubilty-density_EQvsNE_{time_ID}.svg',
    #         save_data_dir=f'Results/CO2-{polymer}_81C_{eos_parameter_type}EoSParameters_kswFugacity_solubility_EQvsNE_{time_ID}.csv',
    #     )
    
    #* Plot solubility isotherms at 35, 51 and 81 °C with default parameters
    # eos_parameter_type = 'default'
    # for polymer in ['PS', 'PMMA']:
        # plot_isotherm_EQvNE_multiT_custom2(
        #     # p_l=1,
        #     # p_u=25e6,
        #     no_p_points=60,
        #     T_list=[35+273],
        #     sol='CO2',
        #     pol=polymer,
        #     rho20_list=[rho20_35C[polymer]],
        #     ksw_list=[ksw_35C[polymer][eos_parameter_type]],
        #     xlxs_sheet_refno_list=['8'],
        #     include_NE=True,
        #     include_EQ=True,
        #     x_lo=0.0, y_lo=0.0,
        #     display_plot=False,
        #     save_plot_dir=result_folder_dir + f'\\CO2-{polymer}_35C_{eos_parameter_type}EoSParameters_kswFugacity_solubilty-density_EQvsNE_{time_ID}.svg',
        #     save_data_dir=f'Results\\CO2-{polymer}_35C_{eos_parameter_type}EoSParameters_kswFugacity_solubilty-density_EQvsNE_{time_ID}.csv',
        # )
        # plot_isotherm_EQvNE_multiT_custom2(
        #     # p_l=1,
        #     # p_u=25e6,
        #     no_p_points=60,
        #     T_list=[51+273],
        #     sol='CO2',
        #     pol=polymer,
        #     rho20_list=[rho20_51C[polymer]],
        #     ksw_list=[ksw_51C[polymer][eos_parameter_type]],
        #     xlxs_sheet_refno_list=['8'],
        #     include_NE=True,
        #     include_EQ=True,
        #     x_lo=0.0, y_lo=0.0,
        #     display_plot=False,
        #     # save_data_dir=f'Results\\CO2-{polymer}_51C_{eos_parameter_type}EoSParameters_kswFugacity_solubilty-density_EQvsNE_{time_ID}.csv',
        #     save_plot_dir=result_folder_dir + f'\\CO2-{polymer}_51C_{eos_parameter_type}EoSParameters_kswFugacity_solubilty-density_EQvsNE_{time_ID}.svg',
        # )    
        # plot_isotherm_EQvNE_multiT_custom2(
        #     # p_l=1,
        #     # p_u=25e6,
        #     no_p_points=60,
        #     T_list=[81+273],
        #     sol='CO2',
        #     pol=polymer,
        #     rho20_list=[rho20_81C[polymer]],
        #     ksw_list=[ksw_81C[polymer][eos_parameter_type]],
        #     xlxs_sheet_refno_list=['8'],
        #     include_NE=True,
        #     include_EQ=True,
        #     display_plot=False,
        #     x_lo=0.0, y_lo=0.0,
        #     # save_data_dir=f'Results\\CO2-{polymer}_81C_{eos_parameter_type}EoSParameters_kswFugacity_EQvsNE_{time_ID}.csv',
        #     save_plot_dir=result_folder_dir + f'\\CO2-{polymer}_81C_{eos_parameter_type}EoSParameters_kswFugacity_solubilty-density_EQvsNE_{time_ID}.svg',
        # )

    
# Export data for 100, 132 °C
# eos_parameter_type = 'fitted'

# for polymer in ['PS', 'PMMA']:
#     plot_isotherm_EQvNE_multiT(
#         # p_l=1,
#         # p_u=25e6,
#         no_p_points=60,
#         T_list=[100+273, 132+273],
#         sol="CO2",
#         pol=polymer,
#         rho20_list=[0, 0],    # irrelevant for EQ
#         ksw_list=[0, 0],      # irrelevant for EQ
#         xlxs_sheet_refno_list=['8'],
#         include_NE=False,
#         include_EQ=True,
#         display_plot=False,
#         # save_plot_dir=result_folder_dir + f'\\CO2-{polymer}_35C_{eos_parameter_type}EoSParameters_kswFugacity_EQvsNE_{time_ID}.svg',
#         save_data_dir=f'Results\\CO2-{polymer}_35C_{eos_parameter_type}EoSParameters_EQ_{time_ID}.csv',
#     )
    
    # plot_isotherm_EQ(
    #     # p_l=1,
    #     # p_u=25e6,
    #     no_p_points=40,
    #     T=35+273,
    #     sol="CO2",
    #     pol="PS",
    #     # xlxs_sheet_refno_list=["4"],
    #     display_plot=True,
    #     # save_plot_dir=savedir,
    # )
    
    # plot_isotherm_EQ_multiT(
    #     # p_l=1,
    #     # p_u=25e6,
    #     no_p_points=40,
    #     T_list=[175+273, 200+273],
    #     sol="CO2",
    #     pol="PMMA",
    #     xlxs_sheet_refno_list=["4"],
    #     # display_plot=True,
    #     save_plot_dir=savedir,
    # )
    
    # plot_density_EQvNE_multiT(
    #     p_l=1,
    #     p_u=10e6,
    #     no_p_points=20,
    #     sol="CO2",
    #     pol="PS",
    #     T_list=[35+273, 51+273, 81+273],
    #     ksw_list=[0.008293414047454917, 0.006648969545681969, 0.004739022681128743],
    #     rho20_list=[1.0418247536493246, 1.0372232644871846, 1.0297523453933208],    
    #     xlxs_sheet_refno_list=["8"],
    #     display_plot=True,
    #     save_plot_dir=savedir,
    # )
    
    #* Plot for PS with default parameters    
    # plot_isotherm_EQ_multiT(
    #     # p_l=1,
    #     # p_u=25e6,
    #     no_p_points=40,
    #     T_list=[35+273, 51+273, 81+273],
    #     sol="CO2",
    #     pol="PS",
    #     xlxs_sheet_refno_list=["8"],
    #     display_plot=False,
    #     save_plot_dir=result_folder_dir+f"\\CO2-PS_35-51-81C_default_EoS_{time_ID}.png",
    # )
    # plot_isotherm_EQ_multiT(
    #     # p_l=1,
    #     # p_u=25e6,
    #     no_p_points=40,
    #     T_list=[35+273, 51+273, 81+273],
    #     sol="CO2",
    #     pol="PMMA",
    #     xlxs_sheet_refno_list=["8"],
    #     display_plot=False,
    # )
    
    #* Plot all exp in references at 35 °C
    # polymer = 'PMMA'
    # plot_isotherm_EQ_multiT(
    #     # p_l=1,
    #     # p_u=5e6,
    #     no_p_points=5,
    #     T_list=[35+273],
    #     sol="CO2", pol=polymer,
    #     xlxs_sheet_refno_list=None,
    #     display_plot=False,
    #     save_plot_dir=result_folder_dir+f"\\CO2-{polymer}_35C_allRefs_zoomed_{time_ID}.svg",
    # ) 
    #* Plot common exp data
    # plot_isotherm_EQ_multiT(
    #     # p_l=1,
    #     # p_u=25e6,
    #     no_p_points=5,
    #     T_list=[100+273],
    #     sol="CO2",
    #     pol="PS",
    #     xlxs_sheet_refno_list=['8', '21'],
    #     display_plot=True,
    #     save_plot_dir=result_folder_dir+f"\\CO2-PS_100C_exp_PantoulaVsSanto_{time_ID}.svg",        
    # )

    #* Fitting results
    # eos_parameter_type = 'fitted'
    # polymer = 'PMMA'
    # plot_isotherm_EQvNE_multiT(
    #     # p_l=1,
    #     # p_u=25e6,
    #     no_p_points=40,
    #     T_list=[100+273, 132+273],
    #     sol="CO2",
    #     pol=polymer,
    #     rho20_list=[None, None],
    #     ksw_list=[None, None],
    #     xlxs_sheet_refno_list=['8'],
    #     include_NE=False,
    #     include_EQ=True,
    #     display_plot=False,
    #     save_plot_dir=result_folder_dir + f'\\CO2-{polymer}_100-132C_{eos_parameter_type}EoSParameters_kswFugacity_EQvsNE_{time_ID}.svg',
    # )

    plot_isotherm_EQvNE_multiT_custom4(
        # p_l=1,
        # p_u=25e6,
        no_p_points=2,
        T_list=[35+273, 81+273],
        sol='CO2',
        pol='PS',
        rho20_list=[0,0],
        ksw_list=[0,0],
        xlxs_sheet_refno_list=['8'],
        # include_NE=True,
        include_EQ=True,
        display_plot=True,
        x_lo=0.0, y_lo=0.0,
        # save_data_dir=f'Results\\CO2-{polymer}_81C_{eos_parameter_type}EoSParameters_kswFugacity_EQvsNE_{time_ID}.csv',
        # save_plot_dir=result_folder_dir + f'\\CO2-{polymer}_81C_{eos_parameter_type}EoSParameters_kswFugacity_solubilty-density_EQvsNE_{time_ID}.svg',
    )
