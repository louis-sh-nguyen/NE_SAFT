import math
import os
import time
from datetime import datetime
# import addcopyfighandler
import matplotlib.pyplot as plt
import pandas as pd
# import NET_SAFTgMie_master as NE_SAFT   #* Default
import NET_SAFTgMie_sensitivity_aux as NE_SAFT   # Test
import numpy as np
import re
import shutil
from colour import Color
import matplotlib
from plot_isotherm_main import update_subplot_ticks

# matplotlib parameters
# matplotlib.rcParams["figure.figsize"] = [4.0, 3.5]  # in inches
matplotlib.rcParams["mathtext.default"] = "regular"  # same as regular text
matplotlib.rcParams["font.family"] = "DejaVu Sans"  # alternative: "serif"
matplotlib.rcParams["font.size"] = 7.0
matplotlib.rcParams["axes.titlesize"] = 7  
matplotlib.rcParams["axes.labelsize"] = 7  
matplotlib.rcParams["xtick.labelsize"] = 7 
matplotlib.rcParams["ytick.labelsize"] = 7 
matplotlib.rcParams["legend.fontsize"] = 7 
matplotlib.rcParams["legend.frameon"] = False
matplotlib.rcParams["grid.linestyle"] = "-."
matplotlib.rcParams["grid.linewidth"] = 0.15  # in point units
matplotlib.rcParams["figure.autolayout"] = True

def plot_noRepeatingUnit_sensitivity_EQ(
    no_l: float,
    no_u: float,
    no_of_points: int,
    T: float,
    p: float,
    sol: str,
    pol: str,
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
    ) = NE_SAFT.get_mixture_info(sol, pol)

    no_repeating_unit = np.linspace(no_l, no_u, no_of_points)
    MW2_list = no_repeating_unit * _MW_monomer  # [g/mol]
    solubility_EQ = [NE_SAFT.solve_solubility_EQ(T, p, sol, pol, MW_2_) for MW_2_ in MW2_list]
    print('no repeating unit = ', no_repeating_unit)
    print("solubility_EQ = ", solubility_EQ)
    
    # Calculate upper limits of x and y axis
    solubility_EQ_NoneRemoved = [s for s in solubility_EQ if s is not None]
    x_min = min(no_repeating_unit)
    x_max = max(no_repeating_unit)
    y_min = min(solubility_EQ_NoneRemoved)
    y_max = max(solubility_EQ_NoneRemoved)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # EQ
    ax.plot(
        no_repeating_unit,
        solubility_EQ,
        color='black',
        marker='None',
        linestyle='solid',
        label="EQ",
    )
    # labelling
    ax.set_xlabel('Number of repating units')
    ax.set_ylabel(r"Solubility $/$ $g_{s} \; g^{-1}_{p}$")
    # ax.set_title(r"$CO_{2}$-%s "%(pol) + f"at %s °C, %.1f MPa" % (T - 273, p * 1e-6))    
    # ax.set_yscale('log')
    
    # Update ticks
    update_subplot_ticks(ax, x_lo=0., y_lo=0.)
    
    ax.legend().set_visible(True)
    
    if save_plot_dir != None:
        plt.savefig(save_plot_dir, dpi=1200)
        print(f"Plot saved: {save_plot_dir}")
        print("")
    if display_plot == True:
        plt.show()


def plot_noRepeatingUnit_sensitivity_EQ_custom(
    no_l: float,
    no_u: float,
    no_of_points: int,
    T: float,
    p: float,
    sol: str,
    pol: str,
    fig_size: tuple = None,
    x_lo: float = None,
    x_up: float = None,
    y_lo: float = None,
    y_up: float = None,
    label: str = None,
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
    ) = NE_SAFT.get_mixture_info(sol, pol)

    no_repeating_unit = np.linspace(no_l, no_u, no_of_points)
    MW2_list = no_repeating_unit * _MW_monomer  # [g/mol]
    solubility_EQ = [NE_SAFT.solve_solubility_EQ(T, p, sol, pol, MW_2_) for MW_2_ in MW2_list]
    print('no repeating unit = ', no_repeating_unit)
    print("solubility_EQ = ", solubility_EQ)
    
    # Plotting
    if fig_size == None:
        fig = plt.figure()  # Default        
    elif isinstance(fig_size, tuple):
        fig = plt.figure(figsize=fig_size)  # Big plot        
    else:
        fig = plt.figure()  # Default
    
    ax = fig.add_subplot(111)
    
    # EQ
    ax.plot(
        no_repeating_unit,
        solubility_EQ,
        color='black',
        marker='None',
        markerfacecolor='None',
        linestyle='dashdot',
        label="EQ",
    )
    
    # Labelling
    ax.set_xlabel('Number of repating units')
    ax.set_ylabel(r"Solubility $/$ $g_{s} \; g^{-1}_{p}$")
    # ax.set_title(r"$CO_{2}$-%s "%(pol) + f"at %s °C, %.1f MPa" % (T - 273, p * 1e-6))    
    # ax.set_yscale('log')
    
    # Update ticks to cover all data points
    update_subplot_ticks(ax,  x_lo=x_lo, x_up=x_up, y_lo=y_lo, y_up=y_up)
    
    # Set legend location and handlelength
    # ax.legend(handlelength=2.6).set_visible(True)
    
    # Add label
    if label != None:
        fig.text(0., 0.98, f'{label}', ha='left', va='top', transform=fig.transFigure)
    
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
    time_ID = now.strftime("%y%m%d-%H%M")  # YYMMDD_HHMM
    
    # Create new directory to store results
    result_folder_dir = f'{src_dir}\\Anals\\Paper plots'
    
    # Number of repeating unit sensitivity
    # plot_noRepeatingUnit_sensitivity_EQ(no_l=1, no_u=1.5e3, no_of_points=20, 
    #                                     T=100+273, p=1e6,
    #                                     sol='CO2', pol='PS',
    #                                     display_plot=True,
    #                                     save_plot_dir=result_folder_dir + f"\\CO2-PS_100C_1MPa_SolubilityEQ_noRepeatingUnitSensitivity_{time_ID}.png"
    #                                     )
    # plot_noRepeatingUnit_sensitivity_EQ(no_l=1, no_u=1.5e3, no_of_points=20, 
    #                                     T=100+273, p=1e6, 
    #                                     sol='CO2', pol='PMMA',
    #                                     display_plot=True, 
    #                                     save_plot_dir=result_folder_dir + f"\\CO2-PMMA_100C_1MPa_SolubilityEQ_noRepeatingUnitSensitivity_{time_ID}.png"
    #                                     )
    
    # Number of repeating unit sensitivity
    plot_noRepeatingUnit_sensitivity_EQ_custom(no_l=1, no_u=300, no_of_points=100, 
                                        T=100+273, p=1.0e6,
                                        sol='CO2', pol='PS',
                                        x_lo=0, y_lo=0, 
                                        fig_size=(3.0, 2.5),
                                        label='a)',
                                        display_plot=False, 
                                        save_plot_dir=f'Anals/Paper plots/CO2-PS_100C_1MPa_SolubilityEQ_noRepeatingUnitSensitivity.png'
                                        )
    plot_noRepeatingUnit_sensitivity_EQ_custom(no_l=1, no_u=300, no_of_points=100, 
                                    T=100+273, p=1.0e6,
                                    sol='CO2', pol='PMMA',
                                    x_lo=0, y_lo=0, 
                                    fig_size=(3.0, 2.5),
                                    label='b)',
                                    display_plot=False,                                        
                                    save_plot_dir=f'Anals/Paper plots/CO2-PMMA_100C_1MPa_SolubilityEQ_noRepeatingUnitSensitivity.png'
                                    )