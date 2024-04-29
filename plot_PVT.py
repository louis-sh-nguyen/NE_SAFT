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
import shutil
from colour import Color

def plot_pol_Ldensity_isobar_EQ(
    pol: str,
    MW2: float = None,
    pol_state: str = "rubbery",
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

        # Plotting
        fig = plt.figure(figsize=(4.8, 3.5))
        ax = fig.add_subplot(111)
        colours = list(Color("silver").range_to(Color("maroon"), len(p_unq_list)))  # colour gradient
        for i, p in enumerate(p_unq_list):
            # EQ Calc data
            ax.plot(
                [T - 273 for T in T_exp[i]],
                rho_EQ_calc[i],
                color="%s" % colours[i],
                marker="None",
                linestyle="solid",
                label="{:.0f} MPa".format(p * 1e-6),
            )
            # Exp data
            ax.plot(
                [T - 273 for T in T_exp[i]],
                rho_exp[i],
                color="%s" % colours[i],
                marker="x",
                linestyle="None",
                # label= "{:.0f} MPa".format(p*1e-6),
            )

        ax.set_xlabel("T (°C)")
        ax.set_ylabel(r"$\rho_{pol}^{0}$ ($g/cm^{3}$)")
        ax.set_title(f"{pol} pure density")
        
        ax.set_xlim(left=0, right=300)
        # ax.set_ylim(bottom=0.90, top=1.20)  # PS
        ax.set_ylim(bottom=1.0, top=1.30)  # PMMA
        ax.tick_params(direction="in")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left").set_visible(True)
        if save_plot_dir != None:
            plt.savefig(save_plot_dir, dpi=1200, transparent=True)
            print(f"Plot saved: {save_plot_dir}")
            print("")
        if display_plot == True:
            plt.show()


def plot_pol_PVT_exp(
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
    result_folder_dir = src_dir 
    figname = f"PMMA_PVT_default_{time_ID}.png"
    savedir = result_folder_dir + f"\\{figname}"
                    
    plot_pol_Ldensity_isobar_EQ(pol="PMMA",
                                pol_state="all",
                                display_plot=True, 
                                save_plot_dir=savedir,
                                )

    # plot_pol_PVT_exp(pol="PS",
    #                  display_plot=True, 
    #                  save_plot_dir=savedir,
    #                  )