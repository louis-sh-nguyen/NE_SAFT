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
    ax.set_ylabel(r"Solubility ($g_{sol} \; / \; g_{pol}$)")
    ax.set_title(r"$CO_{2}$-%s "%(pol) + f"at %s °C, %.1f MPa" % (T - 273, p * 1e-6))
    
    ax.set_yscale('log')
    
    # Get the length of major ticks on the x-axis
    x_major_tick_length = ax.get_xticks()[1] - ax.get_xticks()[0]
    
    # Get the length of major ticks on the y-axis
    y_major_tick_length = ax.get_yticks()[1] - ax.get_yticks()[0]
    
    ax.set_xlim(left=0, right=x_max + x_major_tick_length)
    ax.set_ylim(bottom=0, top=y_max + y_major_tick_length)
    ax.tick_params(direction="in")
    ax.legend().set_visible(True)
    
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
    
    # Number of repeating unit sensitivity
    plot_noRepeatingUnit_sensitivity_EQ(no_l=1, no_u=1.5e3, no_of_points=20, 
                                        T=100+273, p=1e6,
                                        sol='CO2', pol='PS',
                                        display_plot=True,
                                        save_plot_dir=result_folder_dir + f"\\CO2-PS_100C_1MPa_SolubilityEQ_noRepeatingUnitSensitivity_{time_ID}.png"
                                        )
    plot_noRepeatingUnit_sensitivity_EQ(no_l=1, no_u=1.5e3, no_of_points=20, 
                                        T=100+273, p=1e6, 
                                        sol='CO2', pol='PMMA',
                                        display_plot=True, 
                                        save_plot_dir=result_folder_dir + f"\\CO2-PMMA_100C_1MPa_SolubilityEQ_noRepeatingUnitSensitivity_{time_ID}.png"
                                        )