import math
import os
import time
from datetime import datetime
# import addcopyfighandler
import matplotlib.pyplot as plt
import pandas as pd
import NET_SAFTgMie_master as NE_SAFT
from plot_isotherm_main import update_subplot_ticks
import numpy as np
import re
import shutil
from colour import Color

# def update_subplot_ticks(ax, x_lo=None, y_lo=None, x_up=None, y_up=None):
#     """Update x and y ticks of subplot ax to cover all data. Put ticks to inside.

#     Args:
#         ax: plot object.
#     """
#     # Adjust lower x and y ticks to start from 0
#     if x_lo != None:
#         ax.set_xlim(left=x_lo)
#     if y_lo != None:
#         ax.set_ylim(bottom=y_lo)
#     if x_up != None:
#         ax.set_xlim(right=x_up)
#     if y_up != None:
#         ax.set_ylim(top=y_up)
    
#     # Get the largest and smallest x ticks and y ticks
#     max_y_tick = max(ax.get_yticks())
#     max_x_tick = max(ax.get_xticks())
#     min_y_tick = min(ax.get_yticks())
#     min_x_tick = min(ax.get_xticks())
    
#     # Get the length of major ticks on the x-axis
#     ax_x_major_tick_length = ax.get_xticks()[1] - ax.get_xticks()[0]
#     ax_y_major_tick_length = ax.get_yticks()[1] - ax.get_yticks()[0]
    
#     # Adjust upper x and y ticks to cover all data
#     if x_up == None:        
#         ax.set_xlim(right=max_x_tick + ax_x_major_tick_length)
#     if y_up == None:
#         ax.set_ylim(top=max_y_tick + ax_y_major_tick_length)
#     if x_lo == None:
#         ax.set_xlim(left=min_x_tick - ax_x_major_tick_length)
#     if y_lo == None:
#         ax.set_ylim(bottom=min_y_tick - ax_y_major_tick_length)
    
#     # Put ticks to inside
#     ax.tick_params(direction="in")

def plot_pol_pvt_EosPrediction(
    pol: str,
    MW2: float = None,
    pol_state: str = "rubbery",
    x_lo: float = None,
    x_up: float = None,
    y_lo: float = None,
    y_up: float = None,
    fig_size: tuple = None,
    display_plot: bool = True,
    save_plot_dir: str = None,
):
    eos_pol, _MW_2, _MW_monomer, _rho_pol_am = NE_SAFT.get_mixture_info(sol=None, pol=pol)
    MW2 = MW2 if MW2 != None else _MW_2

    # Read data
    try:
        # Read exp file
        path = os.path.join(os.path.dirname(__file__), "litdata")
        path = r"\\?\%s" % path  # extended path (for very long path length)
        refpath = path + "\\references.xlsx"
        databasepath = path + "\\pol_PVT.xlsx"
        reffile = pd.ExcelFile(refpath, engine="openpyxl")
        datafile = pd.ExcelFile(databasepath, engine="openpyxl")
        data_sheet = f"{pol}_{pol_state}"
        df = pd.read_excel(datafile, data_sheet)
        print(df)
        df.dropna(subset=["P (Pa)"], inplace=True)  # drop NaN rows
    except Exception as e:
        print("")
        print("Error - importing exp data failed:")
        print("e")
    else:
        p_unq_list = sorted(list(set(df["P (Pa)"].values.tolist())))
        print(p_unq_list)
        rho_EQ_calc = [None for i in p_unq_list]
        rho_exp = [None for i in p_unq_list]
        T_exp = [None for i in p_unq_list]
        for i, p in enumerate(p_unq_list):
            df1 = df[(df["P (Pa)"] == p)]
            rho_exp[i] = df1["rho_pol (g/cm3)"]
            T_exp[i] = df1["T (K)"].values.tolist()
            rho_EQ_calc[i] = [NE_SAFT.get_pol_prop_EQ(T=_T_, p=p, pol=pol, MW2=MW2)[0] for _T_ in T_exp[i]]  # [g/cm^3]

        # Calculate upper limits of x and y axis
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')
        for i, p in enumerate(p_unq_list):
            x_min = min(x_min, min(T_exp[i])-273)
            x_max = max(x_max, max(T_exp[i])-273)            
            y_min = min(y_min, min(rho_exp[i].values.tolist() + rho_EQ_calc[i]))    # tolist() converts pandas series to list for joining two lists
            y_max = max(y_max, max(rho_exp[i].values.tolist() + rho_EQ_calc[i]))    
            
        # Plotting
        if fig_size == None:
            fig = plt.figure(figsize=(4.8, 3.5))    # Default            
        elif isinstance(fig_size, tuple):
            fig = plt.figure(figsize=fig_size)      # Big plot        
        else:
            fig = plt.figure(figsize=(4.8, 3.5))    # Default
        
        ax = fig.add_subplot(111)
        colours = list(Color("silver").range_to(Color("maroon"), len(p_unq_list)))  # colour gradient
        for i, p in enumerate(p_unq_list):
            # EQ Calc data
            ax.plot(
                [T - 273 for T in T_exp[i]],
                rho_EQ_calc[i],
                color='%s' % colours[i],
                marker='None',
                linestyle='solid',
                label='{:.0f} MPa'.format(p * 1e-6),
            )
            # Exp data
            ax.plot(
                [T - 273 for T in T_exp[i]],
                rho_exp[i],
                color='%s' % colours[i],
                marker='x',
                linestyle='None',
                # label= '{:.0f} MPa'.format(p*1e-6),
            )

        ax.set_xlabel(f'T $/$ °C')
        ax.set_ylabel(r'$\rho_{pol}^{0}$ $/$ $g \; cm^{-3}$')
        # ax.set_title(f'{pol} pure density')
        
        # Update ticks to cover all data points
        update_subplot_ticks(ax,  x_lo=x_lo, x_up=x_up, y_lo=y_lo, y_up=y_up)
        
        # Set legend outside the plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left").set_visible(True)
        
        if save_plot_dir != None:
            plt.savefig(save_plot_dir, dpi=1200)
            print(f"Plot saved: {save_plot_dir}")
            print("")
        
        if display_plot == True:
            plt.show()

def plot_pol_pvt_EosPrediction_custom(
    pol: str,
    MW2: float = None,
    pol_state: str = "rubbery",
    x_lo: float = None,
    x_up: float = None,
    y_lo: float = None,
    y_up: float = None,
    fig_size: tuple = None,
    display_plot: bool = True,
    save_plot_dir: str = None,
):
    eos_pol, _MW_2, _MW_monomer, _rho_pol_am = NE_SAFT.get_mixture_info(sol=None, pol=pol)
    MW2 = MW2 if MW2 != None else _MW_2

    # Read data
    try:
        # Read exp file
        path = os.path.join(os.path.dirname(__file__), "litdata")
        path = r"\\?\%s" % path  # extended path (for very long path length)
        refpath = path + "\\references.xlsx"
        databasepath = path + "\\pol_PVT.xlsx"
        reffile = pd.ExcelFile(refpath, engine="openpyxl")
        datafile = pd.ExcelFile(databasepath, engine="openpyxl")
        data_sheet = f"{pol}_{pol_state}"
        df = pd.read_excel(datafile, data_sheet)
        print(df)
        df.dropna(subset=["P (Pa)"], inplace=True)  # drop NaN rows
    except Exception as e:
        print("")
        print("Error - importing exp data failed:")
        print("e")
    else:
        p_unq_list = sorted(list(set(df["P (Pa)"].values.tolist())))
        print(p_unq_list)
        rho_EQ_calc = [None for i in p_unq_list]
        rho_exp = [None for i in p_unq_list]
        T_exp = [None for i in p_unq_list]
        for i, p in enumerate(p_unq_list):
            df1 = df[(df["P (Pa)"] == p)]
            rho_exp[i] = df1["rho_pol (g/cm3)"]
            T_exp[i] = df1["T (K)"].values.tolist()
            rho_EQ_calc[i] = [NE_SAFT.get_pol_prop_EQ(T=_T_, p=p, pol=pol, MW2=MW2)[0] for _T_ in T_exp[i]]  # [g/cm^3]

        # Calculate upper limits of x and y axis
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')
        for i, p in enumerate(p_unq_list):
            x_min = min(x_min, min(T_exp[i])-273)
            x_max = max(x_max, max(T_exp[i])-273)            
            y_min = min(y_min, min(rho_exp[i].values.tolist() + rho_EQ_calc[i]))    # tolist() converts pandas series to list for joining two lists
            y_max = max(y_max, max(rho_exp[i].values.tolist() + rho_EQ_calc[i]))    
            
        # Plotting
        if fig_size == None:
            fig = plt.figure(figsize=(4.8, 3.5))    # Default            
        elif isinstance(fig_size, tuple):
            fig = plt.figure(figsize=fig_size)      # Big plot        
        else:
            fig = plt.figure(figsize=(4.8, 3.5))    # Default
        
        ax = fig.add_subplot(111)
        colours = list(Color("silver").range_to(Color("maroon"), len(p_unq_list)))  # colour gradient
        for i, p in enumerate(p_unq_list):
            # EQ Calc data
            ax.plot(
                [T - 273 for T in T_exp[i]],
                rho_EQ_calc[i],
                color='%s' % colours[i],
                marker='None',
                linestyle='dashdot',
                label='{:.0f} MPa'.format(p * 1e-6),
            )
            # Exp data
            ax.plot(
                [T - 273 for T in T_exp[i]],
                rho_exp[i],
                color='%s' % colours[i],
                marker='x',
                linestyle='None',
                # label= '{:.0f} MPa'.format(p*1e-6),
            )

        ax.set_xlabel(f'T $/$ °C')
        ax.set_ylabel(r'$\rho_{pol}^{0}$ $/$ $g \; cm^{-3}$')
        # ax.set_title(f'{pol} pure density')
        
        # Update ticks to cover all data points
        update_subplot_ticks(ax,  x_lo=x_lo, x_up=x_up, y_lo=y_lo, y_up=y_up)
        
        # Set legend outside the plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left").set_visible(True)
        
        if save_plot_dir != None:
            plt.savefig(save_plot_dir, dpi=1200)
            print(f"Plot saved: {save_plot_dir}")
            print("")
        
        if display_plot == True:
            plt.show()

def plot_pol_pvt_EosPrediction_custom2(
    pol: str,
    MW2: float = None,
    pol_state: str = "rubbery",
    x_lo: float = None,
    x_up: float = None,
    y_lo: float = None,
    y_up: float = None,
    fig_size: tuple = None,
    display_plot: bool = True,
    save_plot_dir: str = None,
):
    eos_pol, _MW_2, _MW_monomer, _rho_pol_am = NE_SAFT.get_mixture_info(sol=None, pol=pol)
    MW2 = MW2 if MW2 != None else _MW_2

    # Read data
    try:
        # Read exp file
        path = os.path.join(os.path.dirname(__file__), "litdata")
        path = r"\\?\%s" % path  # extended path (for very long path length)
        refpath = path + "\\references.xlsx"
        databasepath = path + "\\pol_PVT.xlsx"
        reffile = pd.ExcelFile(refpath, engine="openpyxl")
        datafile = pd.ExcelFile(databasepath, engine="openpyxl")
        data_sheet = f"{pol}_{pol_state}"
        df = pd.read_excel(datafile, data_sheet)
        print(df)
        df.dropna(subset=["P (Pa)"], inplace=True)  # drop NaN rows
    except Exception as e:
        print("")
        print("Error - importing exp data failed:")
        print("e")
    else:
        p_unq_list = sorted(list(set(df["P (Pa)"].values.tolist())))
        print(p_unq_list)
        rho_EQ_calc = [None for i in p_unq_list]
        rho_exp = [None for i in p_unq_list]
        T_exp = [None for i in p_unq_list]
        for i, p in enumerate(p_unq_list):
            df1 = df[(df["P (Pa)"] == p)]
            rho_exp[i] = df1["rho_pol (g/cm3)"]
            T_exp[i] = df1["T (K)"].values.tolist()
            rho_EQ_calc[i] = [NE_SAFT.get_pol_prop_EQ(T=_T_, p=p, pol=pol, MW2=MW2)[0] for _T_ in T_exp[i]]  # [g/cm^3]

        # Calculate upper limits of x and y axis
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')
        for i, p in enumerate(p_unq_list):
            x_min = min(x_min, min(T_exp[i])-273)
            x_max = max(x_max, max(T_exp[i])-273)            
            y_min = min(y_min, min(rho_exp[i].values.tolist() + rho_EQ_calc[i]))    # tolist() converts pandas series to list for joining two lists
            y_max = max(y_max, max(rho_exp[i].values.tolist() + rho_EQ_calc[i]))    
            
        # Plotting
        if fig_size == None:
            fig = plt.figure(figsize=(4.8, 3.5))    # Default            
        elif isinstance(fig_size, tuple):
            fig = plt.figure(figsize=fig_size)      # Big plot        
        else:
            fig = plt.figure(figsize=(4.8, 3.5))    # Default
        
        ax = fig.add_subplot(111)
        colours = list(Color("silver").range_to(Color("maroon"), len(p_unq_list)))  # colour gradient
        for i, p in enumerate(p_unq_list):
            # EQ Calc data
            ax.plot(
                [T - 273 for T in T_exp[i]],
                rho_EQ_calc[i],
                color='%s' % colours[i],
                marker='None',
                linestyle='dashdot',
                label='{:.0f} MPa'.format(p * 1e-6),
            )
            # Exp data
            ax.plot(
                [T - 273 for T in T_exp[i]],
                rho_exp[i],
                color='%s' % colours[i],
                marker='x',
                markersize=5,
                linestyle='None',
                # label= '{:.0f} MPa'.format(p*1e-6),
            )

        ax.set_xlabel(f'T $/$ °C')
        ax.set_ylabel(r'$\rho_{pol}^{0}$ $/$ $g \; cm^{-3}$')
        # ax.set_title(f'{pol} pure density')
        
        # Update ticks to cover all data points
        update_subplot_ticks(ax,  x_lo=x_lo, x_up=x_up, y_lo=y_lo, y_up=y_up)
        
        # Set legend outside the plot
        # ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left").set_visible(True)
        
        if save_plot_dir != None:
            plt.savefig(save_plot_dir, dpi=1200)
            print(f"Plot saved: {save_plot_dir}")
            print("")
        
        if display_plot == True:
            plt.show()

def plot_pol_pvt_EosPrediction_custom3(
    pol: str,
    MW2: float = None,
    pol_state: str = "rubbery",
    x_lo: float = None,
    x_up: float = None,
    y_lo: float = None,
    y_up: float = None,
    fig_size: tuple = None,
    display_plot: bool = True,
    save_plot_dir: str = None,
):
    eos_pol, _MW_2, _MW_monomer, _rho_pol_am = NE_SAFT.get_mixture_info(sol=None, pol=pol)
    MW2 = MW2 if MW2 != None else _MW_2

    # Read data
    try:
        # Read exp file
        path = os.path.join(os.path.dirname(__file__), "litdata")
        path = r"\\?\%s" % path  # extended path (for very long path length)
        refpath = path + "\\references.xlsx"
        databasepath = path + "\\pol_PVT.xlsx"
        reffile = pd.ExcelFile(refpath, engine="openpyxl")
        datafile = pd.ExcelFile(databasepath, engine="openpyxl")
        data_sheet = f"{pol}_{pol_state}"
        df = pd.read_excel(datafile, data_sheet)
        print(df)
        df.dropna(subset=["P (Pa)"], inplace=True)  # drop NaN rows
    except Exception as e:
        print("")
        print("Error - importing exp data failed:")
        print("e")
    else:
        p_unq_list = sorted(list(set(df["P (Pa)"].values.tolist())))
        print(p_unq_list)
        rho_EQ_calc = [None for i in p_unq_list]
        rho_exp = [None for i in p_unq_list]
        T_exp = [None for i in p_unq_list]
        for i, p in enumerate(p_unq_list):
            df1 = df[(df["P (Pa)"] == p)]
            rho_exp[i] = df1["rho_pol (g/cm3)"]
            T_exp[i] = df1["T (K)"].values.tolist()
            rho_EQ_calc[i] = [NE_SAFT.get_pol_prop_EQ(T=_T_, p=p, pol=pol, MW2=MW2)[0] for _T_ in T_exp[i]]  # [g/cm^3]

        # Calculate upper limits of x and y axis
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')
        for i, p in enumerate(p_unq_list):
            x_min = min(x_min, min(T_exp[i])-273)
            x_max = max(x_max, max(T_exp[i])-273)            
            y_min = min(y_min, min(rho_exp[i].values.tolist() + rho_EQ_calc[i]))    # tolist() converts pandas series to list for joining two lists
            y_max = max(y_max, max(rho_exp[i].values.tolist() + rho_EQ_calc[i]))    
            
        # Plotting
        if fig_size == None:
            fig = plt.figure(figsize=(4.8, 3.5))    # Default            
        elif isinstance(fig_size, tuple):
            fig = plt.figure(figsize=fig_size)      # Big plot        
        else:
            fig = plt.figure(figsize=(4.8, 3.5))    # Default
        
        ax = fig.add_subplot(111)
        colours = list(Color("silver").range_to(Color("maroon"), len(p_unq_list)))  # colour gradient
        for i, p in enumerate(p_unq_list):
            # EQ Calc data
            ax.plot(
                [T - 273 for T in T_exp[i]],
                rho_EQ_calc[i],
                color='%s' % colours[i],
                marker='None',
                linestyle='solid',
                label='{:.0f} MPa'.format(p * 1e-6),
            )
            # Exp data
            ax.plot(
                [T - 273 for T in T_exp[i]],
                rho_exp[i],
                color='%s' % colours[i],
                marker='x',
                markersize=5,
                linestyle='None',
                # label= '{:.0f} MPa'.format(p*1e-6),
            )

        ax.set_xlabel(f'T $/$ °C')
        ax.set_ylabel(r'$\rho_{pol}^{0}$ $/$ $g \; cm^{-3}$')
        # ax.set_title(f'{pol} pure density')
        
        # Update ticks to cover all data points
        update_subplot_ticks(ax,  x_lo=x_lo, x_up=x_up, y_lo=y_lo, y_up=y_up)
        
        # Set legend outside the plot
        # ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left").set_visible(True)
        
        if save_plot_dir != None:
            plt.savefig(save_plot_dir, dpi=1200)
            print(f"Plot saved: {save_plot_dir}")
            print("")
        
        if display_plot == True:
            plt.show()


def plot_pol_pvt_exp(
    pol: str,
    MW2: float = None,
    pol_state: str = "all",
    display_plot: bool = True,
    save_plot_dir: str = None,
):
    eos_pol, _MW_2, _MW_monomer, _rho_pol_am = NE_SAFT.get_mixture_info(sol=None, pol=pol)
    MW2 = MW2 if MW2 != None else _MW_2

    # Read data
    try:
        # Read exp file
        path = os.path.join(os.path.dirname(__file__), "litdata")
        path = r"\\?\%s" % path  # extended path (for very long path length)
        refpath = path + "\\references.xlsx"
        databasepath = path + "\\pol_PVT.xlsx"
        reffile = pd.ExcelFile(refpath, engine="openpyxl")
        datafile = pd.ExcelFile(databasepath, engine="openpyxl")
        data_sheet = f"{pol}_{pol_state}"
        df = pd.read_excel(datafile, data_sheet)
        print(df)
        df.dropna(subset=["P (Pa)"], inplace=True)  # drop NaN rows
    except Exception as e:
        print("")
        print("Error - importing exp data failed:", e)
        print("")
    else:
        p_unq_list = sorted(list(set(df["P (Pa)"].values.tolist())))
        print(p_unq_list)
        rho_exp = [None for i in p_unq_list]
        T_exp = [None for i in p_unq_list]
        for i, p in enumerate(p_unq_list):
            df1 = df[(df["P (Pa)"] == p)]
            rho_exp[i] = df1["rho_pol (g/cm3)"]
            T_exp[i] = df1["T (K)"].values.tolist()

        # Plotting
        fig = plt.figure(figsize=(4.8, 3.5))
        ax = fig.add_subplot(111)
        colours = list(Color("silver").range_to(Color("maroon"), len(p_unq_list)))  # colour gradient
        for i, p in enumerate(p_unq_list):            
            # Exp data
            ax.plot(
                [T - 273 for T in T_exp[i]],
                rho_exp[i],
                color="%s" % colours[i],
                marker="x",
                linestyle="None",
                label= "{:.0f} MPa".format(p*1e-6),
            )

        ax.set_xlabel("T (°C)")
        ax.set_ylabel(r"$\rho_{pol}^{0}$ ($g/cm^{3}$)")
        ax.set_title(f"{pol} pure density")
        
        ax.set_xlim(left=0, right=300)
        ax.tick_params(direction="in")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left").set_visible(True)
        if save_plot_dir != None:
            plt.savefig(save_plot_dir, dpi=1200, transparent=True)
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
    # figname = f"PS_PVT_default_{time_ID}.png"
    # savedir = result_folder_dir + f"\\{figname}"
    state = 'rubbery'
    eos_parameter_type = 'default'
    
    polymer = 'PS'
    # plot_pol_pvt_EosPrediction_custom(pol=polymer,
    #                            pol_state=state,
    #                            y_lo=0.9, y_up=1.15,
    #                            fig_size=(3.0, 2.5),
    #                            display_plot=True,
    #                         #    save_plot_dir=f'Anals/Paper plots/{polymer}density_{state}_{eos_parameter_type}EoSparameters.svg',
    #                            )
    # plot_pol_pvt_EosPrediction_custom2(pol=polymer,
    #                            pol_state=state,
    #                            x_lo=100, x_up=275,
    #                            y_lo=0.9, y_up=1.15,
    #                            fig_size=(3.0, 2.5),
    #                            display_plot=True,
    #                            save_plot_dir=f'Anals/Paper plots/{polymer}density_{state}_{eos_parameter_type}EoSparameters.svg',
    #                            )
    # plot_pol_pvt_EosPrediction_custom3(pol=polymer,
    #                            pol_state=state,
    #                            x_lo=100, x_up=275,
    #                            y_lo=0.9, y_up=1.15,
    #                            fig_size=(3.0, 2.5),
    #                            display_plot=True,
    #                            save_plot_dir=f'Anals/Paper plots/{polymer}density_{state}_{eos_parameter_type}EoSparameters.svg',
    #                            )
    polymer = 'PMMA'
    # plot_pol_pvt_EosPrediction_custom(pol=polymer,
    #                            pol_state=state,
    #                            display_plot=True,
    #                            x_lo=90,
    #                            save_plot_dir=f'Anals/Paper plots/{polymer}density_{state}_{eos_parameter_type}EoSparameters.svg',
    #                            )
    plot_pol_pvt_EosPrediction_custom2(pol=polymer,
                               pol_state=state,
                                x_lo=100, x_up=250,
                                y_lo=0.8, y_up=1.3,
                                fig_size=(3.0, 2.5),
                                display_plot=False,
                                save_plot_dir=f'Anals/Paper plots/{polymer}density_{state}_{eos_parameter_type}EoSparameters.svg',
                                )
    # plot_pol_pvt_EosPrediction_custom2(pol=polymer,
    #                            pol_state=state,
    #                            x_lo=90,
    #                            fig_size=(3.0, 2.5),
    #                            display_plot=False,
    #                            save_plot_dir=f'Anals/Paper plots/{polymer}density_{state}_{eos_parameter_type}EoSparameters.svg',
    #                            )
    # plot_pol_pvt_EosPrediction_custom3(pol=polymer,
    #                            pol_state=state,
    #                            x_lo=100, x_up=250,
    #                            y_lo=1.05, y_up=1.25,
    #                            fig_size=(3.0, 2.5),
    #                            display_plot=False,
    #                            save_plot_dir=f'Anals/Paper plots/{polymer}density_{state}_{eos_parameter_type}EoSparameters.svg',
    #                            )

    # plot_pol_PVT_exp(pol="PS",
    #                  display_plrubotherot=True, 
    #                  save_plot_dir=savedir,
    #                  )
    
    # NE_SAFT.fit_polPVT_multiTait(xlxs_sheet='PMMA_rubbery', 
    #                              display_plot=True, 
    #                              save_plot_dir=result_folder_dir + f"\\PMMA_rubbery_multiTait_{time_ID}.png")