"""
Louis Nguyen
Department of Cheimcal Engineering, Imperial College London
sn621@ic.ac.uk
"""

# Turn off numba warning
import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)
warnings.filterwarnings("ignore")

import autosave_NETSAFTgMie as save_py
import NELF_aux as NELF
from sgtpy_NETGP import component, mixture, saftgammamie

# from sgtpy import component, mixture, saftgammamie

import math
import os
import time
from datetime import datetime
import addcopyfighandler
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
from colour import Color
from numpy import *
import pandas as pd
from scipy.optimize import curve_fit, fsolve
import NET_SAFTgMie_master as NE_SAFT


# Plotting master configuration
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

# dictionary for common plotting point styles
exp_style = {
    "color": "black",
    "marker": "o",
    "linestyle": "None",
    "markerfacecolor": "None",
}
calc_style = {"color": "black", "marker": "None", "linestyle": "solid"}

custom_colours = [
    "black",
    "green",
    "blue",
    "red",
    "purple",
    "orange",
    "brown",
    "plum",
    "indigo",
    "olive",
    "grey",
]
custom_markers = ["o", "x", "^", "*", "s", "D"]
custom_linestyles = ["solid", "dashed", "dotted", "dashdot", (0, (1, 10))]  # descending priority

def get_mixture_info(sol: str, pol: str, MW2: float = None):
    MW1_ref = {"CO2": 44, "CH4": 16, "C2H4": 28, "H2O": 18}  # [g/mol]

    MWmonomer_ref = {
        "PMMA": 100,
        "PS": 104,
    }  # [g/mol]
    MW2_ref = {  # n=1000 repeating units
        "PMMA": 100 * 1000,
        "PS": 104 * 1000,
    }  # [g/mol]
    rho_pol_am_ref = {
        "PMMA": 1.181,  # * REAL
        "PS": 1.056,  # * REAL
    }  # [g-pol-am/cm^3-pol-am]
    if pol != None:
        MW_2 = MW2_ref[pol] if MW2 == None else MW2  # [g/mol]
        MW_monomer = MWmonomer_ref[pol]  # [g/mol]
        rho_pol_am = rho_pol_am_ref[pol]  # [g-pol-am/cm^3-pol-am]
        # polymer properties
        if pol == "PS":
            k_sw_ref = 0.0060  # CO2-PS [MPa^-1]
            n = math.ceil(MW_2 / MW_monomer)  # round up
            # polymer_obj = component(GC={"aCH": 5 * n,"aCCH": 1 * n,"CH2": 1 * n})   #* Default
            polymer_obj = component(GC={"aCH_PS": 5 * n, "aCCH": 1 * n, "CH2": 1 * n})  # *Optimised

        elif pol == "PMMA":
            k_sw_ref = 0.027  # CO2-PMMA [MPa^-1]
            n = math.ceil(MW_2 / MW_monomer)  # round up
            # polymer_obj = component(GC={"CH2": 1 * n, "C": 1 * n, "CH3": 2 * n, "COO": 1 * n})  # * Default
            polymer_obj = component(GC={"CH2": 1 * n, "C": 1 * n, "CH3": 2 * n, "COO_PMMA": 1 * n})  # * Optimised

        # Create SAFT-g Mie EOS object of pure polymer
        polymer_obj.saftgammamie()
        eos_pol = saftgammamie(polymer_obj, compute_critical=False)

    if sol != None:
        MW_1 = MW1_ref[sol]  # [g/mol]
        if sol == "CO2":
            sol_obj = component(GC={"CO2": 1})
        # Create Create SAFT-g Mie EOS object of pure solute
        sol_obj.saftgammamie()
        eos_sol = saftgammamie(sol_obj)

    if sol != None and pol == None:
        return eos_sol, MW_1
    if pol != None and sol == None:
        return eos_pol, MW_2, MW_monomer, rho_pol_am
    if sol != None and pol != None:
        # mixture object
        mix_obj = mixture(sol_obj, polymer_obj)
        # Create SAFT-g Mie EOS object of Mixture
        mix_obj.saftgammamie()
        eos_mix = saftgammamie(mix_obj, compute_critical=False)
        return eos_mix, eos_sol, MW_1, MW_2, MW_monomer, rho_pol_am, k_sw_ref


def solve_solubility_NE(
    T: float,
    p: float,
    sol: str,
    pol: str,
    MW2: float,
    ksw: float,
    rho20: float,
    display_result: bool = False,
    return_extended: bool = False,
):
    """Solve solubility of solute in pol using NE-SAFT-g Mie.

    Args:
        T (float): Temperature [K]
        P (float): Pressure (Pa)
        MW_2 (float): MW of amorphous polymer [g/mol]
        k_sw (float): swelling coefficient [MPa^-1]
        rho_2_dry (float): polymer denisty [g-pol/cm^3-mix]
    Returns:
        float: solubility [g-sol/g-pol-am]
    """
    p_MPa = p * 1e-6  # [MPa]

    # get mixture properties
    eos_mix, eos_sol, MW_1, _MW_2, _MW_monomer, _rho_2_am_dry, _k_sw = get_mixture_info(sol, pol, MW2)
    rho20 = rho20 if rho20 != None else _rho_2_am_dry
    ksw = ksw if ksw != None else _k_sw
    psat, vlsat, vvsat = eos_sol.psat(T)
    # Saturation Pressure (Pa), saturated liquid volume (m3/mol), saturated vapor volume (m3/mol).
    if p >= psat:  # L phase
        state_ext = 'L'
        # print("external phase: liquid")
    else:  # V phase
        state_ext = 'V'
        # print("external phase: vapour")
    
    rho_ext = eos_sol.density(T, p, state_ext)  # [mol/m^3]
    # print("rho_1 = ", rho_1)

    # Calculate chemical potential of external phase
    muad_ext = eos_sol.muad(rho_ext, T) / (8.314 * T)  # dimensionless
    
    # Calculate fugacity of external phase
    lnFugCoeff_ext = eos_sol.logfug(T, p, state_ext, 1/rho_ext)[0]
    fugCoeff_ext = exp(lnFugCoeff_ext)
    fug_ext = fugCoeff_ext * p      # [Pa]
    fug_ext_MPa = fug_ext * 1e-6    # [MPa]
    # print("fugacity of external phase = ", fug_ext_MPa)

    # NE polymer mixture
    def func(x_1_):
        """
        Args:
            x_1_ (_type_): mol_sol/mol_mix

        """
        # Convert mole fraction [mol/mol] to mass fraction [g/g]
        omg_1_ = (x_1_ * MW_1) / (x_1_ * MW_1 + (1 - x_1_) * MW2)  # [g_sol/g_mix]
        
        # Converting density from g/cm^3 to mol/m^3
        rhol_0_ = 1e6 * (rho20 / MW2) * 1 / (1 - x_1_)  # dry density [mol-mix/m^3-mix] #* Default

        #* density-pressure relation
        #* OLD relation
        # rhol_ = rhol_0_ * (1 - ksw * p_MPa)  # [mol-mix/m^3-mix]
        #* NEW relation, Default
        # rhol_ = rhol_0_ / (1 + ksw * p_MPa)  # [mol-mix/m^3-mix]
        
        #* OLD relation, using fugacity
        # rhol_ = rhol_0_ * (1 - ksw * fug_ext_MPa)  # [mol-mix/m^3-mix]
        #* NEW relation, using fugacity
        rhol_ = rhol_0_ / (1 + ksw * fug_ext_MPa)  # [mol-mix/m^3-mix]
        
        #* TEST relation #1
        # rhol_ = rhol_0_ * (1 - ksw * rho_1)  # [mol-mix/m^3-mix]
        #* TEST relation #2
        # rhol_ = rhol_0_ * (1 - ksw * omg_1_)  # [mol-mix/m^3-mix]
        #* TEST relation #3
        # rhol_ = rhol_0_ / (1 + ksw * omg_1_)  # [mol-mix/m^3-mix]        

        x_ = hstack([x_1_, 1 - x_1_])  # [mol/mol-mix]
        rho_i_ = x_ * rhol_  # [mol/m^3-mix]
        muad_S = eos_mix.muad(rho_i_, T)  # adimensional [mu/RT]

        return [muad_S[0] - muad_ext]

    # x0 = linspace(9.90e-1, 9.99e-1, 10)     #*original
    x0 = linspace(8.00e-1, 9.99e-1, 30)  # *test

    i = 0
    while i < (len(x0)):
        # print("i = ", i)
        try:
            solution = fsolve(func, x0=x0[i], xtol=1e-10)
            residue = func(x_1_=solution)
            # Check return of func() is 0
            # print("\tsolution = ", solution)
            # print("\tresidue = ", residue)
            residue_float = [float(i) for i in residue]
            if isclose(residue_float, [0.0]).all() == True:
                # print('Solution found')
                x_1 = solution[0]
                
                # Check if x_1 is within [0,1]
                if x_1 < 0 or x_1 > 1:
                    i += 1
                    continue
                
                x = hstack([x_1, 1 - x_1])  # [mol/mol-mix]
                # calculate muad_S                
                omega_1 = (x_1 * MW_1) / (x_1 * MW_1 + (1 - x_1) * MW2)  # [g_sol/g_mix]
                _rhol_ = 1e6 * (rho20 / MW2) * 1 / (1 - x_1) * (1 - ksw * p_MPa)
                _rho_i_ = x * _rhol_
                muad1 = eos_mix.muad(_rho_i_, T)

                # print("rho_L [mol_mix/m^3_mix] = ", _rhol_)
                # print("rho_i [mol/m^3_mix] = ", _rho_i_)
                # print("muad_S =", muad1[0])
                # print("muad_G = ", muad_G)

                # print("\nBackward calculation from muad_S")
                # print(
                #     "rho_L Topliss's method [mol_mix/m^3_mix] = ",
                #     eos_mix.density(x, T, p, "L"),
                # )
                # print(
                #     "rho_L Newton's method [mol_mix/m^3_mix] = ",
                #     eos_mix.density(x, T, p, "L", rho0=rho20),
                # )
                # print("rho_i [mol/m^3_mix] = ", eos_mix.density(x, T, p, "L") * x)

                # print("x_1 = %g [mol_CO2/mol_mix]" % x_1)
                # print("rho20 from input [g/cm^3]: ", rho20)
                # print(
                #     "rho20 from SAFT [g/cm^3]: ",
                #     eos_mix.density(x, T, p, "L") * 1e-6 * (1 - x_1) * MW2 / (1 - ksw * p_MPa),
                # )

                # calculate density
                rhol_0 = 1e6 * (rho20 / MW2) * 1 / (1 - x_1)  # dry density [mol-mix/m^3-mix]                
                # rhol_ = rhol_0 * (1 - ksw * rho_1)  #* TEST relation #1 [mol-mix/m^3-mix]                
                # rhol_ = rhol_0 * (1 - ksw * omega_1)  #* TEST relation #2 [mol-mix/m^3-mix]                
                rhol_ = rhol_0 * (1 + ksw * omega_1)  #* TEST relation #3 [mol-mix/m^3-mix]                
                rho_2 = rhol_ * (1 - x_1) * MW2 * 1e-6  # [g-pol/cm^3-mix]
                V2 = 1 / rho_2  # [cm^3-mix/g-pol]
                S_1 = omega_1 / (1 - omega_1)  # [g-sol/g-pol-am]

                if display_result == True:
                    print(
                        "(NE) Solution found ^-^"
                        + "\tT=%s°C, p=%g MPa, k_sw=%g MPa^-1, MW2=%g g/mol" % (T - 273, p_MPa, ksw, MW2)
                    )
                    print("")
                    print("\t(NE) Solubility:\t%g [mol_sol/mol_pol]\t%g [g_sol/g_pol_am]" % ((x_1 / (1 - x_1)), S_1))

                if return_extended == False:
                    return S_1  
                else:
                    return S_1, omega_1, muad1, rho_2, V2
            else:
                i += 1
        except Exception as e:
            i += 1
            if display_result == True:
                print("__Error:\t", e, "__")
                print(
                    "(T=°C, MW2=%g g/mol, ksw=%g MPa^-1, p=%g MPa)" % (T - 273, MW2, ksw, p * 1e-6)
                    + "\tMoving to i = %s/%s " % (i, len(x0))
                )
    if i >= (len(x0)):
        if display_result == True:
            print("(NE) No solution found for T=%s°C,MW2=%g, ksw=%g, p=%g MPa" % (T - 273, MW2, ksw, p * 1e-6))
            print("")


def solve_solubility_EQ(
    T: float,
    p: float,
    sol: str,
    pol: str,
    MW2: float = None,
    display_result: bool = False,
    return_extended: bool = False,
):

    # get mixture properties
    eos_mix, eos_sol, MW_1, _MW_2, _MW_monomer, _rho_2_am_dry, _k_sw = get_mixture_info(sol, pol, MW2)
    MW2 = MW2 if MW2 != None else _MW_2
    # EQ pure solute external gas phase (G)

    psat, vlsat, vvsat = eos_sol.psat(T)
    # Saturation Pressure (Pa), saturated liquid volume (m3/mol), saturated vapor volume (m3/mol).
    if p >= psat:  # L phase
        rho_1 = eos_sol.density(T, p, "L")  # [mol/m^3]
    else:  # V phase
        rho_1 = eos_sol.density(T, p, "V")  # [mol/m^3]

    muad_G = eos_sol.muad(rho_1, T) / (8.314 * T)  # dimensionless

    # EQ polymer mixture
    def func(x_1_):
        x_ = hstack([x_1_, 1 - x_1_])  # [mol/mol-mix]
        rhol_ = eos_mix.density(x_, T, p, "L")  # * Topliss's method
        # rhol_ = eos_mix.density(x_, T, p, "L",rho0=1e+6*_rho_2_inf/_MW_2)  # * Newton's method
        rho_i_ = x_ * rhol_  # [mol/m^3-mix]
        muad_S = eos_mix.muad(rho_i_, T)  # dimensionless [mu/RT]
        return [muad_S[0] - muad_G]

    x0 = linspace(9.90e-1, 9.99e-1, 10)   #* Default
    # x0 = linspace(9.1e-1, 9.99e-1, 10)
    # TODO add x0 finding mechanism
    i = 0

    while i < (len(x0)):
        try:
            solution = fsolve(func, x0=x0[i], xtol= 1e-10)            
            residue = func(x_1_=solution)
            residue_float = [float(j) for j in residue]
            if isclose(residue_float, [0.0]).all() == True:
                x_1 = solution[0]  # [mol-sol/mol-mix]
                omega_1 = (x_1 * MW_1) / (x_1 * MW_1 + (1 - x_1) * MW2)
                S_1 = omega_1 / (1 - omega_1)  # [g-sol/g-pol-am]

                x = hstack([x_1, 1 - x_1])  # [mol/mol-mix]
                rhol = eos_mix.density(x, T, p, "L")  # [mol-mix/m^3-mix]
                rho_2 = rhol * (1 - x_1) * MW2 * 1e-6  # [g-pol/cm^3_mix]
                V2 = 1 / rho_2  # [cm^3-mix/g-pol]
                
                rho_i = x * rhol  # [mol/m^3-mix]
                muad_S = eos_mix.muad(rho_i, T)  # dimensionless [mu/RT]

                if display_result == True:
                    print("(EQ) Solution found ^-^ " + "\tT=%s°C, p=%g MPa, MW2=%g g/mol" % (T - 273, p * 1e-6, MW2))
                    print("\t(EQ) Solubility:\t%g [mol_sol/mol_pol]\t%g [g_sol/g_pol_am]" % ((x_1 / (1 - x_1)), S_1))
                if return_extended == False:
                    return S_1
                else:
                    return S_1, x_1, rhol, rho_2, V2, muad_S[0]
            else:
                i += 1
        except Exception as e:
            i += 1
            if display_result == True:
                print("__Error:\t", e, "__")
                print(
                    "(T=%s°C, p=%g MPa, MW2=%g g/mol)" % (T - 273, p * 1e-6, MW2)
                    + "\tMoving to i = %s/%s " % (i, len(x0))
                )
    if i >= (len(x0)):
        if display_result == True:
            print("(EQ) No solution found for T=%g°C, p=%g MPa, MW2=%g g/mol" % (T - 273, p * 1e-6, MW2))
            print("")

def fit_ksw_NE(
    T: float,
    sol: str,
    pol: str,
    xlxs_sheet_refno: str,
    rho20: float = None,
    MW2: float = None,
    ksw_x0_list: list[float] = linspace(0.0, 0.01, 10),
    glassy_filter: bool = False,
    display_plot: bool = True,
    save_plot_dir: str = None,
) -> tuple[float, float]:
    """_summary_

    Args:
        T (float): _description_
        sol (str): _description_
        pol (str): _description_
        xlxs_sheet_refno (int): _description_
        rho20 (float, optional): _description_. Defaults to None.
        plot (bool, optional): _description_. Defaults to True.
        save_plot (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    (
        _eos_mix,
        _eos_sol,
        _MW_1,
        MW_2,
        _MW_monomer,
        rho_2_am_dry,
        _k_sw,
    ) = get_mixture_info(sol, pol)
    rho20 = rho20 if rho20 != None else rho_2_am_dry  # [g/cm^3]
    MW2 = MW2 if MW2 != None else MW_2  # [g/mol]
    # Read solubiltiy data
    ref_no = xlxs_sheet_refno
    xlxs_sheet = f"S_{T-273}C ({ref_no})"
    try:  # more specific spreadsheet
        path = os.path.join(os.path.dirname(__file__), "litdata")
        refpath = path + "/references.xlsx"
        databasepath = path + "/%s-%s.xlsx" % (sol, pol)
        datafile = pd.ExcelFile(databasepath, engine="openpyxl")
        df = pd.read_excel(datafile, xlxs_sheet)
        df.dropna(subset=["P [MPa]"], inplace=True)  # drop na rows

        # Glassy filter [_fit_ksw_NE?_]
        if glassy_filter == True:
            df1 = df[df["_fit_ksw_NE?_"] == "Yes"]
        else:
            df1 = df

        # Get reference ID
        reffile = pd.ExcelFile(refpath, engine="openpyxl")
        ref_df = pd.read_excel(reffile, "references")
        # get ref ID of matching ref no
        ref_ID = ref_df.loc[ref_df["# ref"] == f"[{ref_no}]", "refID"].item()
        hasExpData = True
    except Exception as e:
        print("")
        print("Error - importing exp data failed:")
        print(e)
        hasExpData = False

    # Continue if solubility is available
    if hasExpData == True:
        print('Solubility data available from sheet "%s".' % xlxs_sheet)
        print(f"exp ref ID: {ref_ID}")
        print("")
    else:
        print('Error: No solubility data available from sheet "%s".' % xlxs_sheet)
        print("")
        return None

    p_MPa_exp = asarray(df1["P [MPa]"])
    p_exp = p_MPa_exp * 1e6  # [Pa]
    solubility_exp = asarray(df1["Solubility [g-sol/g-pol-am]"])
    print("p_list = ", p_exp)
    print("solubility = ", solubility_exp)

    # fitting ksw
    def ksw_fitting_func(p_list_, ksw_fit):
        solubility_list = [solve_solubility_NE(T, p_, sol, pol, MW2, ksw_fit, rho20) for p_ in p_list_]
        return solubility_list

    x0_list = ksw_x0_list
    # start time
    start_time = time.time()
    for i, _ksw0_ in enumerate(x0_list):
        try:
            # print("ksw_0 = ", _ksw0_)
            # Fitting function
            result, cov = curve_fit(ksw_fitting_func, p_exp, solubility_exp, p0=(_ksw0_))

        except Exception as e:
            print("Fitting unsuccessful for ksw_0 = ", _ksw0_)
            if i == (len(x0_list) - 1):
                print("Error - curve_fit unsuccessful.")
            pass
        else:
            print("")
            print("Fitting done. Fitting time: \t %.0f seconds" % (time.time() - start_time))
            ksw_f = result
            print("k_sw = %g MPa^-1" % ksw_f)
            break

    # Fitting error
    solubility_calc_evaluation = ksw_fitting_func(p_exp, ksw_f)

    # Calculate solubility from new ksw
    p_MPa_full = asarray(df["P [MPa]"])
    p_calc = linspace(0.01, p_MPa_full[-1] * 1e6, 40)
    p_MPa_calc = p_calc * 1e-6
    solubility_calc = ksw_fitting_func(p_calc, ksw_f)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # fitting exp data
    ax.plot(
        p_MPa_exp,
        solubility_exp,
        color="black",
        marker=exp_style["marker"],
        linestyle=exp_style["linestyle"],
        # markerfacecolor=exp_style["markerfacecolor"],
        label=f"fitting exp data: {ref_ID} ({ref_no})",
    )
    # remaining exp data
    if len(df.index) != len(df1.index):
        ax.plot(
            df[df["_fit_ksw_NE?_"] != "Yes"]["P [MPa]"],
            df[df["_fit_ksw_NE?_"] != "Yes"]["Solubility [g-sol/g-pol-am]"],
            color="grey",
            marker=exp_style["marker"],
            linestyle=exp_style["linestyle"],
            markerfacecolor=exp_style["markerfacecolor"],
            label=f"unused exp data: {ref_ID} ({ref_no})",
        )
    # calculated from fitting values
    ax.plot(
        p_MPa_calc,
        solubility_calc,
        color=calc_style["color"],
        marker=calc_style["marker"],
        linestyle=calc_style["linestyle"],
        label=r"NE $k_{sw} = %.3g$" % (ksw_f),
    )
    # labelling
    ax.set_xlabel(r"p (MPa)")
    ax.set_ylabel(r"Solubility ($g_{sol} / g_{pol\;am}$)")
    ax.set_title(r"%s-%s at %s°C" % (sol, pol, T - 273))
    ax.annotate(
        r"NE $\rho_{20}$ = %.4f $g/cm^{-3}$" % rho20,
        xy=(1.0, -0.09),
        xycoords="axes fraction",
        ha="right",
        va="center",
        fontsize="xx-small",
    )
    # styling
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    # ax.grid(visible=True)
    ax.tick_params(direction="in")
    ax.legend().set_visible(True)

    if display_plot == True:
        plt.show()
    if save_plot_dir != None:
        plt.savefig(save_plot_dir, dpi=1200)
        print(f"Plot saved: {save_plot_dir}")
        print("")

    return ksw_f[0]


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
    ) = get_mixture_info(sol, pol, MW2)
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
            p_MPa_exp_list[i] = asarray(dict[sheet]["P [MPa]"])
            solubility_exp_list[i] = asarray(dict[sheet]["Solubility [g-sol/g-pol-am]"])
    
    # Get pressure range
    if p_l != None and p_u != None:
        p_calc = linspace(p_l, p_u, no_p_points)  # [Pa]
    else:
        if hasExpData == True:
            for i, sheet in enumerate(matched_sheets):
                current_max_p_MPa = p_MPa_exp_list[i].max()
                if i == 0:
                    max_p_MPa = current_max_p_MPa
                else:
                    max_p_MPa = max(current_max_p_MPa, max_p_MPa)
        
        max_p = max_p_MPa * 1e6  # [Pa]
        p_calc = linspace(1, max_p, no_p_points)    # [Pa]
        
    p_MPa_calc = p_calc * 1e-6
    print("p_cal = ", p_calc)

    # Create empty placeholders, return None in case calculation fails
    solubility_NE_calc_list = [None for i in range(len(ksw_list))]
    p_MPa_exp_list = [None for i in range(len(matched_sheets))]
    solubility_exp_list = [None for i in range(len(matched_sheets))]
    label_NE = [None for i in range(len(ksw_list))]
    print("p_cal = ", p_calc)
    # calculated NE solubility for each ksw
    for i, ksw_ in enumerate(ksw_list):
        solubility_NE_calc_list[i] = [solve_solubility_NE(T, _p_, sol, pol, MW2, ksw_list[i], rho20) for _p_ in p_calc]
        print(
            "solubility at ksw=%g =\t" % ksw_list[i], solubility_NE_calc_list[i]
        )  # calculated NE solubility with ksw != 0

    #* Calculate EQ solubility
    solubility_EQ_list = [solve_solubility_EQ(T, p_, sol, pol, MW2) for p_ in p_calc]
    print("\nsolubility_EQ = ", solubility_EQ_list)

    # Original label
    label_EQ = "EQ"
    label_NE = [r"NE $k_{sw} = %.3g \, MPa^{-1}$" % (ksw_) for ksw_ in ksw_list]

    # Importing exp data
    if hasExpData == True:
        for i, sheet in enumerate(matched_sheets):
            p_MPa_exp_list[i] = asarray(dict[sheet]["P [MPa]"])
            solubility_exp_list[i] = asarray(dict[sheet]["Solubility [g-sol/g-pol-am]"])
    
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Exp solubility
    if hasExpData == True:
        for i, sheet in enumerate(matched_sheets):
            ax.plot(
                p_MPa_exp_list[i],
                solubility_exp_list[i],
                color=exp_style["color"],
                marker=custom_markers[i],
                linestyle=exp_style["linestyle"],
                markerfacecolor=exp_style["markerfacecolor"],
                label=f"exp: {ref_ID[i]} ({ref_no[i]})",
            )

    #* EQ solubility
    ax.plot(
        p_MPa_calc,
        solubility_EQ_list,
        color=custom_colours[1],
        marker="None",
        linestyle="solid",
        label=label_EQ,
    )

    # NE solubility
    for i, ksw_ in enumerate(ksw_list):
        ax.plot(
            p_MPa_calc,
            solubility_NE_calc_list[i],
            color=custom_colours[2],
            marker=calc_style["marker"],
            linestyle=custom_linestyles[i],
            label=label_NE[i],
        )
    # labelling
    ax.set_xlabel(r"p (MPa)")
    ax.set_ylabel(r"Solubility ($g_{sol} / g_{pol}$)")
    ax.set_title("%s-%s at %.0f°C " % (sol, pol, T - 273))
        
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

def get_dV2eq_df0(T: float, sol: str, pol: str, p0=10, MW2: float = None, frc_step: float = 1e-4):
    """approximate by dVeq_dp at p=0 as f~p.

    Args:
        T (_type_): _description_
        sol (_type_): _description_
        pol (_type_): _description_
        p0 (int, optional): _description_. Defaults to 10.
        MW_2 (_type_, optional): _description_. Defaults to None.
        frc_step (_type_, optional): _description_. Defaults to 1e-3.

    Returns:
        _type_: _description_
    """
    # get mixture properties
    eos_mix, eos_sol, MW_1, _MW_2, _MW_monomer, rho_2_am_dry, _k_sw = get_mixture_info(sol, pol, MW2)
    MW2 = MW2 if MW2 != None else _MW_2
    
    def get_fug_sol(temp, pressure):
        # Saturation Pressure (Pa), saturated liquid volume (m3/mol), saturated vapor volume (m3/mol).
        psat, vlsat, vvsat = eos_sol.psat(temp)
        
        if pressure >= psat:  # L phase
            state_ext = 'L'
        else:  # V phase
            state_ext = 'V'

        rho_sol = eos_sol.density(temp, pressure, state_ext)  # [mol/m^3]
        lnFugCoeff_sol = eos_sol.logfug(temp, pressure, state_ext, 1/rho_sol)[0]
        fugCoeff_sol = exp(lnFugCoeff_sol)
        fug_sol = fugCoeff_sol * pressure      # [Pa]
        
        return fug_sol
        
    def get_V2eq(_p):  # with penetrant presence
        S_gg, x1, rhol = solve_solubility_EQ(T, _p, sol, pol, MW2, return_extended=True)[:3]
        rho_eq = rhol * (x1 * MW_1 + (1 - x1) * MW2) * 1e-6  # [g_mix/cm^3_mix]
        rho2_eq = rho_eq * (1 - S_gg / (S_gg + 1))  # [g_pol/cm^3_mix]
        V2_eq = 1 / rho2_eq  # [cm^3_mix/g_mix]   #* Default
        # V2_eq = 1 / rho_eq  # [cm^3_mix/g_mix]     #* Test
        return V2_eq

    def fx(_p0_): #dVeq/df
        p_l = _p0_ * (1 - frc_step)  # [Pa]
        p_u = _p0_ * (1 + frc_step)  # [Pa]
        f_l = get_fug_sol(T, p_l)  # [Pa]
        f_u = get_fug_sol(T, p_u)  # [Pa]
        V2eq_l = get_V2eq(p_l)  # [cm^3/g]
        V2eq_u = get_V2eq(p_u)  # [cm^3/g]
        return (V2eq_u - V2eq_l) / (f_u - f_l)

    p0_list = arange(1, 5, 1)  # [Pa]   #* Default
    # p0_list = arange(1e-3, 5e-3, 1e-3)  # [Pa]   #* Test
    dV2eq_df = [fx(_p0_) for _p0_ in p0_list]    
    dV2eq_dp_ave = average(dV2eq_df)
    isclose_checker = all(
        [isclose(x, dV2eq_dp_ave, dV2eq_dp_ave * 0.05) for x in dV2eq_df]
    )  # 5% tolerance arounge average

    if isclose_checker == True:
        return dV2eq_dp_ave
    else:
        print("Error: dV2eq_dp_ave numerical result unsatisfactory")
        return None


def predict_ksw_NE(
    T: float,
    sol: str,
    pol: str,
    rho20: float = None,
    MW2: float = None,
    unit: str = "MPa^-1",
) -> float:
    eos_pol, _MW_2, _MW_monomer, rho_pol_am = get_mixture_info(sol=None, pol=pol)
    rho20 = rho20 if rho20 != None else rho_pol_am
    MW2 = MW2 if MW2 != None else _MW_2

    # Step 1: rho_pol_0
    V_pol0 = 1 / rho20  # [cm^3/g]

    # Step 2: chi
    chi = NE_SAFT.get_chi(pol)  # [adim]

    # Step 3: dV2eq_df0 numercial result
    dV2eq_df0 = get_dV2eq_df0(T, sol, pol, MW2=MW2)
    
    # Step 4: ksw
    ksw_Pa = chi / V_pol0 * dV2eq_df0  # [Pa^-1]
    ksw_MPa = ksw_Pa * 1e6  # [MPa^-1]
    if unit == "MPa^-1":
        return ksw_MPa
    if unit == "Pa^-1":
        return ksw_Pa


def get_dlnrho2eq_df0(T: float, sol: str, pol: str, p0=10, MW2: float = None, frc_step: float = 1e-4):
    """approximate by drho2eq_dp at p=0 as f~p.

    Args:
        T (_type_): _description_
        sol (_type_): _description_
        pol (_type_): _description_
        p0 (int, optional): _description_. Defaults to 10.
        MW_2 (_type_, optional): _description_. Defaults to None.
        frc_step (_type_, optional): _description_. Defaults to 1e-3.

    Returns:
        _type_: _description_
    """
    # get mixture properties
    eos_mix, eos_sol, MW_1, _MW_2, _MW_monomer, rho_2_am_dry, _k_sw = get_mixture_info(sol, pol, MW2)
    eos_pol = get_mixture_info(sol=None, pol=pol)[0]    
    MW2 = MW2 if MW2 != None else _MW_2
    
    # Get rho_pol0_EQ
    rho_pol0_EQ = eos_pol.density(T, 1e-5, 'L')    # [mol/m^3]
    rho_pol0_EQ = rho_pol0_EQ * MW2 *1e-6       # [g/cm^3]
    
    def get_fug_sol(temp, pressure):
        # Saturation Pressure (Pa), saturated liquid volume (m3/mol), saturated vapor volume (m3/mol).
        psat, vlsat, vvsat = eos_sol.psat(temp)
        
        if pressure >= psat:  # L phase
            state_ext = 'L'
        else:  # V phase
            state_ext = 'V'

        rho_sol = eos_sol.density(temp, pressure, state_ext)  # [mol/m^3]
        lnFugCoeff_sol = eos_sol.logfug(temp, pressure, state_ext, 1/rho_sol)[0]
        fugCoeff_sol = exp(lnFugCoeff_sol)
        fug_sol = fugCoeff_sol * pressure      # [Pa]
        
        return fug_sol
        
    def get_rho2eq(_p):  # with penetrant presence
        S_gg, x1, rhol = solve_solubility_EQ(T, _p, sol, pol, MW2, return_extended=True)[:3]
        rho_eq = rhol * (x1 * MW_1 + (1 - x1) * MW2) * 1e-6  # [g_mix/cm^3_mix]
        rho2_eq = rho_eq * (1 - S_gg / (S_gg + 1))  # [g_pol/cm^3_mix]
        return rho2_eq

    def fx(_p0_, step=frc_step): #dlnrho2eq/df
        p_l = _p0_ * (1 - step)  # [Pa]
        p_u = _p0_ * (1 + step)  # [Pa]
        f_l = get_fug_sol(T, p_l)  # [Pa]
        f_u = get_fug_sol(T, p_u)  # [Pa]
        rho2eq_l = get_rho2eq(p_l)  # [g/cm^3]
        rho2eq_u = get_rho2eq(p_u)  # [g/cm^3]
        lnrho2eq_l = log(rho2eq_l)  # [g/cm^3]
        lnrho2eq_u = log(rho2eq_u)  # [g/cm^3]
        return (lnrho2eq_u - lnrho2eq_l) / (f_u - f_l)  #* Default 
        # return (lnrho2eq_u - log(rho_pol0_EQ)) / (f_u)  #* Test

    p0_list = arange(1, 5, 1)  # [Pa]   #* Default
    # p0_list = arange(0.04, 0.05, 0.002)  # [Pa]   #* Test
    dlnrho2eq_df = [fx(_p0_, step=1e-7) for _p0_ in p0_list]
    dlnrho2eq_df_ave = average(dlnrho2eq_df)
    isclose_checker = all(
        [isclose(x, dlnrho2eq_df, rtol=0.05) for x in dlnrho2eq_df]
    )  # 5% tolerance arounge average

    if isclose_checker == True:
        return dlnrho2eq_df_ave
    else:
        print("Error: drho2eq_dp_ave numerical result is unsatisfactory")
        return None

def predict_ksw_NE_TEST(
    T: float,
    sol: str,
    pol: str,
    rho20: float = None,
    MW2: float = None,
    unit: str = "MPa^-1",
) -> float:
    eos_pol, _MW_2, _MW_monomer, rho_pol_am = get_mixture_info(sol=None, pol=pol)
    rho20 = rho20 if rho20 != None else rho_pol_am
    MW2 = MW2 if MW2 != None else _MW_2

    # Step 1: rho_pol_0
    rho_pol0 = rho20  # [g/cm^3]

    # Step 2: chi
    chi = NE_SAFT.get_chi(pol)  # [adim]
    
    # Step 3: rho_pol0_EQ
    rho_pol0_EQ = eos_pol.density(T, 1, 'L')    # [mol/m^3]
    rho_pol0_EQ = rho_pol0_EQ * MW2 *1e-6       # [g/cm^3]

    # Step 4: dVlnrhoeq_df0 numercial result
    dlnrho2eq_df0 = get_dlnrho2eq_df0(T, sol, pol, MW2=MW2)
    
    # Step 5: ksw
    ksw_Pa = - chi * rho_pol0 / rho_pol0_EQ * dlnrho2eq_df0  # [Pa^-1]
    ksw_MPa = ksw_Pa * 1e6  # [MPa^-1]
    
    if unit == "MPa^-1":
        return ksw_MPa
    if unit == "Pa^-1":
        return ksw_Pa

def get_dV2eq_df(T: float, p: float, sol: str, pol: str, MW2: float = None, frc_step: float = 1e-4):
    """approximate by dVeq_dp at p=0 as f~p.

    Args:
        T (_type_): _description_
        sol (_type_): _description_
        pol (_type_): _description_
        p0 (int, optional): _description_. Defaults to 10.
        MW_2 (_type_, optional): _description_. Defaults to None.
        frc_step (_type_, optional): _description_. Defaults to 1e-3.

    Returns:
        _type_: _description_
    """
    # get mixture properties
    eos_mix, eos_sol, MW_1, _MW_2, _MW_monomer, rho_2_am_dry, _k_sw = get_mixture_info(sol, pol, MW2)
    eos_pol = get_mixture_info(sol=None, pol=pol)[0]    
    MW2 = MW2 if MW2 != None else _MW_2
    
    # Get rho_pol0_EQ
    rho_pol0_EQ = eos_pol.density(T, 1e-5, 'L') * MW2 *1e-6    # [g/cm^3]
    V_pol0_EQ = 1 / rho_pol0_EQ     # [cm^3/g]
    
    def get_fug_sol(temp, pressure):
        # Saturation Pressure (Pa), saturated liquid volume (m3/mol), saturated vapor volume (m3/mol).
        psat, vlsat, vvsat = eos_sol.psat(temp)
        
        if pressure >= psat:  # L phase
            state_ext = 'L'
        else:  # V phase
            state_ext = 'V'

        rho_sol = eos_sol.density(temp, pressure, state_ext)  # [mol/m^3]
        lnFugCoeff_sol = eos_sol.logfug(temp, pressure, state_ext, 1/rho_sol)[0]
        fugCoeff_sol = exp(lnFugCoeff_sol)
        fug_sol = fugCoeff_sol * pressure      # [Pa]
        
        return fug_sol
        
    def get_V2eq(_p):  # with penetrant presence
        S_gg, x1, rhol = solve_solubility_EQ(T, _p, sol, pol, MW2, return_extended=True)[:3]
        rho_eq = rhol * (x1 * MW_1 + (1 - x1) * MW2) * 1e-6  # [g_mix/cm^3_mix]
        rho2_eq = rho_eq * (1 - S_gg / (S_gg + 1))  # [g_pol/cm^3_mix]
        V_eq = 1 / rho_eq  # [cm^3_mix/g_mix]
        V2_eq = 1 / rho2_eq  # [cm^3_mix/g_pol]
        return V2_eq    #* Default
        # return V_eq    #* Test

    def fx(_p0_): #dVeq/df
        p_l = _p0_ * (1 - frc_step)  # [Pa]
        p_u = _p0_ * (1 + frc_step)  # [Pa]
        f = get_fug_sol(T, _p0_)  # [Pa]
        f_l = get_fug_sol(T, p_l)  # [Pa]
        f_u = get_fug_sol(T, p_u)  # [Pa]
        V2eq_l = get_V2eq(p_l)  # [cm^3/g]
        V2eq_u = get_V2eq(p_u)  # [cm^3/g]
        return (V2eq_u - V2eq_l) / (f_u - f_l)  #* Default
        # return (V2eq_u - V_pol0_EQ) / (f_u)  #* Test

    if p < 10:
        p_list = arange(1, 5, 1)    # [Pa]
    else:
        p_list = arange(p-2, p+2, 1)  # [Pa]
        
    dV2eq_df = [fx(_p) for _p in p_list]    
    dV2eq_dp_ave = average(dV2eq_df)
    isclose_checker = all(
        [isclose(x, dV2eq_dp_ave, rtol=0.005) for x in dV2eq_df]
    )  # 5% tolerance arounge average

    if isclose_checker == True:
        return dV2eq_dp_ave
    else:
        print("Error: dV2eq_dp_ave numerical result unsatisfactory")
        return None

def predict_ksw_NE_NEW(
    T: float,
    p: float,
    sol: str,
    pol: str,
    rho20: float = None,
    MW2: float = None,
    unit: str = "MPa^-1",
) -> float:
    eos_pol, _MW_2, _MW_monomer, rho_pol_am = get_mixture_info(sol=None, pol=pol)
    rho20 = rho20 if rho20 != None else rho_pol_am
    MW2 = MW2 if MW2 != None else _MW_2

    # Step 1: rho_pol_0
    V_pol0 = 1 / rho20  # [cm^3/g]

    # Step 2: chi
    chi = NE_SAFT.get_chi(pol)  # [adim]

    # Step 3: dV2eq_df0 numercial result
    dV2eq_df0 = get_dV2eq_df(T, p, sol, pol, MW2=MW2)
    
    # Step 4: ksw
    ksw_Pa = chi / V_pol0 * dV2eq_df0  # [Pa^-1]
    ksw_MPa = ksw_Pa * 1e6  # [MPa^-1]
    if unit == "MPa^-1":
        return ksw_MPa
    if unit == "Pa^-1":
        return ksw_Pa
    
if __name__ == "__main__":
    src_dir = os.path.dirname(__file__)
    src_dir = r"\\?\%s" % src_dir  # extended path (for very long path length)

    start_time = time.time()
    now = datetime.now()  # current time
    time_ID = now.strftime("%y%m%d_%H%M")  # YYMMDD_HHMM
    
    # Create new directory to store results
    result_folder_dir = f'{src_dir}\\Anals'
    
    #* rho_pol_0 values
    # From PVT
    # PS
    # rho_35C = 1.042
    # rho_51C = 1.037
    # rho_81C = 1.030    
    # PMMA
    # rho_35C = 1.178
    # rho_51C = 1.174
    # rho_81C = 1.165  
    
    # From fitting
    # PS
    # rho_35C = 1.019
    # rho_51C = 1.030
    # rho_81C = 1.010
    # PMMA
    # rho_35C = 1.139
    # rho_51C = 1.195
    # rho_81C = 1.154
    
    #* Fit ksw
    # polymer = "PS"
    # rho_35C = 1.042
    # rho_51C = 1.037
    # rho_81C = 1.030
    
    # polymer = "PMMA"
    # rho_35C = 1.178
    # rho_51C = 1.174
    # rho_81C = 1.165  
    # try:
    #     ksw_35C = fit_ksw_NE(T=35+273, sol="CO2", pol=polymer, xlxs_sheet_refno="8", rho20=rho_35C, 
    #             ksw_x0_list=linspace(1e-2, 10, 5),
    #             display_plot=False, 
    #             save_plot_dir=result_folder_dir + f'\\CO2-{polymer}_35C_optimised_NETGP_kswFit_newKswFormFugacity_{time_ID}.png'
    #             )
    # except:
    #     pass
    
    # try:
    #     ksw_51C = fit_ksw_NE(T=51+273, sol="CO2", pol=polymer, xlxs_sheet_refno="8", rho20=rho_51C, 
    #             ksw_x0_list=linspace(1e-2, 10, 5),
    #             display_plot=False, 
    #             save_plot_dir=result_folder_dir + f'\\CO2-{polymer}_51C_optimised_NETGP_kswFit_newKswFormFugacity_{time_ID}.png'
    #             )
    # except:
    #     pass
    
    # try:
    #     ksw_81C = fit_ksw_NE(T=81+273, sol="CO2", pol=polymer, xlxs_sheet_refno="8", rho20=rho_81C, 
    #             ksw_x0_list=linspace(1e-2, 10, 5),
    #             display_plot=False,
    #             save_plot_dir=result_folder_dir + f'\\CO2-{polymer}_81C_optimised_NETGP_kswFit_newKswFormFugacity_{time_ID}.png'
    #             )
    # except:
    #     pass
    
    # #* Plot ksw sensitivity
    # Form 1
    # ksw_35C = 7.09e-6
    # ksw_51C = 6.82e-6
    # ksw_81C = 7.47e-6
    
    # Form 2
    # ksw_35C = 0.88868
    # ksw_51C = 0.874
    # ksw_81C = 0.942
    
    # 35 °C
    # try:
    #     plot_isotherm_EQvNE(
    #         # p_l=1,
    #         # p_u=25e6,
    #         no_p_points=40,
    #         sol="CO2",
    #         pol="PMMA",
    #         T=35+273,
    #         rho20=rho_35C,
    #         ksw_list=[ksw_35C, ksw_35C*0.90, ksw_35C*0.95, ksw_35C*1.05, ksw_35C*1.10],
    #         xlxs_sheet_refno_list=["8"],
    #         display_plot=False,
    #         save_plot_dir=result_folder_dir + f'\\CO2-PMMA_35C_optimised_NETGP_newKswForm2_kswSensitivity_{time_ID}.png',
    #     )
    # except:
    #     pass
    
    # 51 °C
    # try:
    #     plot_isotherm_EQvNE(
    #         # p_l=1,
    #         # p_u=25e6,
    #         no_p_points=40,
    #         sol="CO2",
    #         pol="PMMA",
    #         T=51+273,
    #         rho20=rho_51C,
    #         ksw_list=[ksw_51C, ksw_51C*0.90, ksw_51C*0.95, ksw_51C*1.05, ksw_51C*1.10],
    #         xlxs_sheet_refno_list=["8"],
    #         display_plot=False,
    #         save_plot_dir=result_folder_dir + f'\\CO2-PMMA_51C_optimised_NETGP_newKswForm2_kswSensitivity_{time_ID}.png',
    #     )
    # except:
    #     pass
    
    # 81 °C
    # try:
    #     plot_isotherm_EQvNE(
    #         # p_l=1,
    #         # p_u=25e6,
    #         no_p_points=40,
    #         sol="CO2",
    #         pol="PMMA",
    #         T=81+273,
    #         rho20=rho_81C,
    #         ksw_list=[ksw_81C, ksw_81C*0.90, ksw_81C*0.95, ksw_81C*1.05, ksw_81C*1.10],
    #         xlxs_sheet_refno_list=["8"],
    #         display_plot=False,
    #         save_plot_dir=result_folder_dir + f'\\CO2-PMMA_81C_optimised_NETGP_newKswForm2_kswSensitivity_{time_ID}.png',
    #     )
    # except:
    #     pass
    
    # #* Plot EoS vs NET-GP results
    # PS from PVT
    polymer = 'PMMA'
    rho_35C = 1.178
    rho_51C = 1.174
    rho_81C = 1.165
    ksw_35C = predict_ksw_NE(T=35+273, sol='CO2', pol=polymer, rho20=rho_35C)
    ksw_51C = predict_ksw_NE(T=51+273, sol='CO2', pol=polymer, rho20=rho_51C)
    ksw_81C = predict_ksw_NE(T=81+273, sol='CO2', pol=polymer, rho20=rho_81C)
    # 35 °C
    try:
        plot_isotherm_EQvNE(
            # p_l=1,
            # p_u=25e6,
            no_p_points=40,
            sol='CO2',
            pol=polymer,
            xlxs_sheet_refno_list=['8'],
            display_plot=False,
            T=35+273, ksw_list=[ksw_35C], rho20=rho_35C, save_plot_dir=result_folder_dir + f'\\CO2-{polymer}_35C_optimised_EoSvsNETGP_newKswFormFugacity_{time_ID}.png',
        )
    except:
        pass
    
    # 51 °C
    try:
        plot_isotherm_EQvNE(
            # p_l=1,
            # p_u=25e6,
            no_p_points=40,
            sol='CO2',
            pol=polymer,
            xlxs_sheet_refno_list=['8'],
            display_plot=False,
            T=51+273, ksw_list=[ksw_51C], rho20=rho_51C, save_plot_dir=result_folder_dir + f'\\CO2-{polymer}_51C_optimised_EoSvsNETGP_newKswFormFugacity_{time_ID}.png',
        )
    except:
        pass
    
    # 81 °C
    try:
        plot_isotherm_EQvNE(
            # p_l=1,
            # p_u=25e6,
            no_p_points=40,
            sol='CO2',
            pol=polymer,
            xlxs_sheet_refno_list=['8'],
            display_plot=False,
            T=81+273, ksw_list=[ksw_81C], rho20=rho_81C, save_plot_dir=result_folder_dir + f'\\CO2-{polymer}_81C_optimised_EoSvsNETGP_newKswFormFugacity_{time_ID}.png',
        )
    except:
        pass
    
    #* dV/df
    # PS
    # rho20 = 1.042    
    # print("new dV_df = ", get_dV2eq_df0(T=35+273, sol="CO2", pol="PS"))
    # print("old dV_df = ", NE_SAFT.get_dV2eq_df0(T=35+273, sol="CO2", pol="PS"))
    # print("new drho2eq_df = ", get_dlnrho2eq_df0(T=35+273, sol="CO2", pol="PS"))
    
    #* ksw new method
    # ksw_old = predict_ksw_NE(T=35+273, sol="CO2", pol="PS", rho20=1.042)
    # print(f'ksw old = {ksw_old} MPa^-1')
    # ksw_new = predict_ksw_NE_NEW(T=35+273, p=1, sol="CO2", pol="PS", rho20=1.042)
    # print(f'ksw new = {ksw_new} MPa^-1')
    # ksw_test = predict_ksw_NE_TEST(T=35+273, sol="CO2", pol="PS", rho20=1.042)
    # print(f'ksw test = {ksw_test} MPa^-1')
    
    # values = [get_dV2eq_df(35+273, _p, "CO2", "PS") for _p in [1, 100, 1000, 1e5]]
    # print(values)
    # ksw = []
    # for _p in [1, 100, 1000, 1e4, 1e5, 1e6]:
    #     _ksw = predict_ksw_NE_NEW(T=35+273, p=_p, sol="CO2", pol="PS", rho20=1.042)
    #     print(f'p = {_p} Pa,\tksw = {_ksw} MPa^-1')
    #     ksw.append(_ksw)
    # print('ksw = ', ksw)
    # rhopol0 = 1.042
    # ksw = predict_ksw_NE_NEW(T=35+273, p=1, sol="CO2", pol="PS", rho20=rhopol0)
    # print('ksw = ', ksw)
    
    # plot_isotherm_EQvNE(
    #     # p_l=1,
    #     # p_u=25e6,
    #     no_p_points=20,
    #     sol='CO2',
    #     pol='PS',
    #     T=35+273, rho20=rhopol0, ksw_list=[ksw], 
    #     xlxs_sheet_refno_list=["8"],
    #     display_plot=True,
    #     # save_plot_dir=result_folder_dir + f'\\CO2-PMMA_35C_optimised_EoSvsNETGP_newKswForm2_{time_ID}.png',
    # )