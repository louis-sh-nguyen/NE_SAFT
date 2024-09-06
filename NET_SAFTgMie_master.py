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
from scipy.integrate import quad
import external_phase as ext


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


def find_closest(lst, K):
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - K))]


def get_mixture_info(sol: str, pol: str, MW2: float = None):
    MW1_ref = {"CO2": 44, "CH4": 16, "C2H4": 28, "H2O": 18}  # [g/mol]

    MWmonomer_ref = {
        "PMMA": 100,
        "PEMA": 114,
        "PET": 192,
        "PS": 104,
        "HDPE": 28,
        "PLA": 72,
    }  # [g/mol]
    MW2_ref = {  # n=1000 repeating units
        "PMMA": 100 * 1000,
        "PEMA": 114 * 1000,
        "PET": 192 * 1000,
        "PS": 104 * 1000,
        "HDPE": 28 * 1000,
        "PLA": 72 * 1000,
    }  # [g/mol]
    rho_pol_am_ref = {
        # "PMMA": 1.181,  # * REAL
        "PMMA": 0.94,  # *TEST
        # "PEMA": 1.124,  #* REAL
        "PEMA": 0.94,  # *TEST
        "PET": 1.331,
        "PS": 1.056,  # * REAL
        # "PS": 1.041,  # *TEST
        "HDPE": 0.94,
        "PLA": 1.27,
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

        elif pol == "PEMA":
            k_sw_ref = 0.0  # CO2-PMMA [MPa^-1]
            n = math.ceil(MW_2 / MW_monomer)  # round up
            polymer_obj = component(GC={"CH2": 2 * n, "C": 1 * n, "CH3": 2 * n, "COO": 1 * n})

        elif pol == "HDPE":
            k_sw_ref = 0.00  # unknown
            n = math.ceil(MW_2 / MW_monomer)  # round up
            polymer_obj = component(
                GC={
                    "CH2": 2 * n,
                }
            )
        elif pol == "PLA":
            k_sw_ref = 0.00  # unknown
            n = math.ceil(MW_2 / MW_monomer)  # round up
            polymer_obj = component(
                GC={
                    "COO": 1 * n,
                    "CH": 1 * n,
                    "CH3": 1 * n,
                }
            )
        # Create SAFT-g Mie EOS object of pure polymer
        polymer_obj.saftgammamie()
        eos_pol = saftgammamie(polymer_obj, compute_critical=False)

    if sol != None:
        MW_1 = MW1_ref[sol]  # [g/mol]
        if sol == "CO2":
            sol_obj = component(GC={"CO2": 1})
        elif sol == "CH4":
            sol_obj = component(GC={"CH4": 1})
        elif sol == "C2H4":
            sol_obj = component(GC={"CH2": 2})
        elif sol == "H2O":
            sol_obj = component(GC={"H2O": 1})
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
        rho_1 = eos_sol.density(T, p, "L")  # [mol/m^3]
        # print("external phase: liquid")
    else:  # V phase
        rho_1 = eos_sol.density(T, p, "V")  # [mol/m^3]
        # print("external phase: vapour")

    # Calculate chemical potential of external phase
    # muad from SL EOS
    # muad_G_SLEOS = ext.muad_SLEOS("CO2", T, p_MPa)   # SL EOS
    # muad_G_PCSAFT = ext.muad_PCSAFT("CO2", T, p_MPa)  # PC SAFT
    # muad from SGTpy
    muad_G = eos_sol.muad(rho_1, T) / (8.314 * T)  # dimensionless
    # print("muad_G = ", muad_G)

    # muad_G = mu_G/(8.314*T)  # adimensional
    # print("T = ",T)
    # print("P = ",p)
    # print("muad_G_SAFTgMie :" ,muad_G)
    # print("muad_G_SLEOS :" ,muad_G_SLEOS)
    # print("muad_G_PCSAFT :" ,muad_G_PCSAFT)
    # print("mu_G :" ,muad_G*(8.314 * T))
    # testing
    # print("SGTpy = ",muad_G)
    # print("SLEOS = ",muad_G_SLEOS)
    # print("PCSAFT = ",muad_G_PCSAFT)

    # NE polymer mixture
    def func(x_1_):
        """
        Args:
            x_1_ (_type_): mol_sol/mol_mix

        """
        # Converting density from g/cm^3 to mol/m^3
        rhol_0_ = 1e6 * (rho20 / MW2) * 1 / (1 - x_1_)  # dry density [mol-mix/m^3-mix]

        #* density-pressure relation
        #* OLD relation
        # rhol_ = rhol_0_ * (1 - ksw * p_MPa)  # [mol-mix/m^3-mix]
        #* NEW relation  
        rhol_ = rhol_0_ / (1 + ksw * p_MPa)  # [mol-mix/m^3-mix]        

        x_ = hstack([x_1_, 1 - x_1_])  # [mol/mol-mix]
        rho_i_ = x_ * rhol_  # [mol/m^3-mix]
        muad_S = eos_mix.muad(rho_i_, T)  # adimensional [mu/RT]

        return [muad_S[0] - muad_G]

    # x0 = linspace(9.90e-1, 9.99e-1, 10)     #*original
    x0 = linspace(8.00e-1, 9.99e-1, 30)  # *test

    # TODO add x0 finding and "remembering" mechanism
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
                x_1 = solution[0]
                x = hstack([x_1, 1 - x_1])  # [mol/mol-mix]
                # calculate muad_S
                # print("Forward calculation from rho20")
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
                rhol = rhol_0 * (1 - ksw * p_MPa)  # [mol-mix/m^3-mix]
                rho_2 = rhol * (1 - x_1) * MW2 * 1e-6  # [g-pol/cm^3-mix]
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
                    return S_1  # TODO change S_1 to omg1_omg2
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
    p_MPa = p * 1e-6  # [MPa]

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
            solution = fsolve(func, x0=x0[i])            
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

def get_mu_from_x(T: float,
    p: float,
    sol: str,
    pol: str,
    x_sol: float,  #[mol/mol]
    MW2: float = None,
):
    # get mixture properties
    eos_mix, eos_sol, MW_1, _MW_2, _MW_monomer, _rho_2_am_dry, _k_sw = get_mixture_info(sol, pol, MW2)
    MW2 = MW2 if MW2 != None else _MW_2
    
    # Calculate chemical potential of external phase from provided solute composition x_sol
    x = hstack([x_sol, 1 - x_sol])  # [mol/mol-mix]
    rhol = eos_mix.density(x, T, p, "L")  # [mol-mix/m^3-mix]
    rho_i = x * rhol  # [mol/m^3-mix]
    muad_S = eos_mix.muad(rho_i, T)  # dimensionless [mu/RT]
    mu_S = muad_S * 8.314 * T  # [J/mol]
    
    return mu_S[0]
    
        
def get_pol_prop_EQ(T: float, p: float, pol: str, MW2: float = None):
    eos_pol, _MW_2, _MW_monomer, _rho_pol_am = get_mixture_info(sol=None, pol=pol, MW2=MW2)
    MW2 = MW2 if MW2 != None else _MW_2
    # get pol density and volume
    rho_molm3 = eos_pol.density(T, p, "L")  # [mol/m^3]
    rho_gcm3 = rho_molm3 * MW2 * 1e-6  # [g/cm^3]
    V_cm3g = 1 / rho_molm3  # [cm^3/g]
    return rho_gcm3, V_cm3g


def get_p_intersect_EQvNE(
    T: float,
    sol: str,
    pol: str,
    ksw: float,
    rho20: float = None,
    MW2: float = None,
    display_result: bool = False,
):
    # get mixture properties
    eos_mix, eos_sol, MW_1, _MW_2, _MW_monomer, rho_2_am_dry, _k_sw = get_mixture_info(sol, pol, MW2)
    MW2 = MW2 if MW2 != None else _MW_2
    rho20 = rho20 if rho20 != None else rho_2_am_dry

    p_intersect_list = []  # [Pa]

    # finding intersection of NE and EQ model (T=Tg)
    def func(variable):
        """
        Args:
            _p_ (float): glass-transition pressure [Pa]
        """
        _p_ = variable[0]  # [Pa]
        V2_NE = solve_solubility_NE(T, _p_, sol, pol, MW2, ksw, rho20, return_extended=True)[4]  # [cm^3/g]
        V2_EQ = solve_solubility_EQ(T, _p_, sol, pol, MW2, return_extended=True)[4]  # [cm^3/g]
        return [V2_NE - V2_EQ]

    x0 = linspace(1e3, 10e6, 5)  # * [Pa]
    i = 0
    while i < (len(x0)):
        try:
            solution = fsolve(func, x0=float(x0[i]))
            # print("solution = ", solution)
            residue = func(variable=solution)
            residue_float = [float(i) for i in residue]
            if isclose(residue_float, [0.0]).all() == True:
                pg = solution[0]
                if display_result == True:
                    print("\nSolution found ^-^")
                    print("NE and EQ intersect at pg = %g MPa" % (pg * 1e-6))
                if i == 0:  # for the first item
                    p_intersect_list.append(pg)
                else:  # check for duplicates for subsequent item
                    if isclose(pg, p_intersect_list[-1], rtol=pg * 0.01) == True:  # 1% tolerance
                        pass
                    else:
                        p_intersect_list.append(pg)

                i += 1
            else:
                i += 1
        except Exception as e:
            print(e)
            i += 1
            print("\tMoving to i = ", i)
    if i >= (len(x0)):
        p_intersect_unqlist = list(set(p_intersect_list))
        if len(p_intersect_unqlist) == 0:
            print("\nNo pg solution found for T=%g°C" % (T - 273))
    return p_intersect_unqlist


def plot_MW2_sensitivity_NE(
    MW2_l: float,
    MW2_u: float,
    no_of_points: int,
    T: float,
    p: float,
    sol: str,
    pol: str,
    ksw: float = None,
    rho20: float = None,
):
    (
        _eos_mix,
        _eos_sol,
        _MW_1,
        _MW_2,
        _MW_monomer,
        rho_2_am_dry,
        _k_sw,
    ) = get_mixture_info(sol, pol)
    rho20 = rho20 if rho20 != None else rho_2_am_dry
    MW2_list = linspace(MW2_l, MW2_u, no_of_points)

    solubility_list = [solve_solubility_NE(T, p, sol, pol, MW_2_, ksw, rho20) for MW_2_ in MW2_list]
    print("\nMW2_list = ", MW2_list)
    print("solubility_list = ", solubility_list)

    # Plotting
    y_scale = 1e-3
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # calculated solubility
    ax.plot(
        MW2_list * y_scale,
        solubility_list,
        color=calc_style["color"],
        marker=calc_style["marker"],
        linestyle=calc_style["linestyle"],
    )
    ax.text(
        MW2_list[math.ceil(len(MW2_list) / 2)] * y_scale,
        solubility_list[math.ceil(len(solubility_list) / 2) - 1],
        r"$k_{sw}=%s\;MPa^{-1}$" % ksw,
        fontsize="smaller",
    )

    # labelling
    ax.set_xlabel(r"$%g \: MW_{2}$ ($g_{pol} / mol_{pol}$)" % (y_scale))
    ax.set_ylabel(r"Solubility ($g_{CO_{2}} / g_{pol\:am}$)")
    ax.set_title(r"$CO_{2}$-%s at %s°C %s, MPa" % (pol, T - 273, p * 1e-6))
    # styling
    ax.set_ylim(bottom=0)
    # ax.grid(visible=True)
    ax.tick_params(direction="in")
    plt.show()


def plot_MW2_sensitivity_EQ(
    MW2_l: float,
    MW2_u: float,
    no_of_points: int,
    T: float,
    p: float,
    sol: str,
    pol: str,
):
    (
        _eos_mix,
        _eos_sol,
        _MW_1,
        _MW_2,
        _MW_monomer,
        rho_2_am_dry,
        _k_sw,
    ) = get_mixture_info(sol, pol)

    MW2_list = linspace(MW2_l, MW2_u, no_of_points)
    solubility_EQ = [solve_solubility_EQ(T, p, sol, pol, MW_2_) for MW_2_ in MW2_list]
    print("\nMW2_list = ", MW2_list)
    print("solubility_EQ = ", solubility_EQ)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # EQ
    ax.plot(
        MW2_list,
        solubility_EQ,
        color=calc_style["color"],
        marker="None",
        linestyle="dashed",
        label="EQ",
    )
    # labelling
    ax.set_xlabel(r"$MW_{2}$ (g/mol)")
    ax.set_ylabel(r"Solubility ($g_{CO_{2}} / g_{pol\:am}$)")
    ax.set_title(r"$CO_{2}$-%s at %s°C, %s MPa" % (pol, T - 273, p * 1e-6))
    # styling
    # ax.set_ylim(bottom=0)
    # ax.grid(visible=True)
    ax.tick_params(direction="in")
    ax.legend().set_visible(True)
    plt.show()


def plot_MW2_sensitivity_EQvNE(
    MW2_l: float,
    MW2_u: float,
    no_of_points: int,
    T: float,
    p: float,
    sol: str,
    pol: str,
    ksw: float = None,
    rho20: float = None,
):
    (
        _eos_mix,
        _eos_sol,
        _MW_1,
        _MW_2,
        _MW_monomer,
        rho_2_am_dry,
        _k_sw,
    ) = get_mixture_info(sol, pol)
    rho20 = rho20 if rho20 != None else rho_2_am_dry

    MW2_list = linspace(MW2_l, MW2_u, no_of_points)

    solubility_NE = [solve_solubility_NE(T, p, sol, pol, MW_2_, ksw, rho20) for MW_2_ in MW2_list]
    solubility_EQ = [solve_solubility_EQ(T, p, sol, pol, MW_2_) for MW_2_ in MW2_list]
    print("\nMW2_list = ", MW2_list)
    print("solubility_NE = ", solubility_NE)
    print("solubility_EQ = ", solubility_EQ)
    # Plotting

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # NE
    ax.plot(
        MW2_list,
        solubility_NE,
        color=calc_style["color"],
        marker=calc_style["marker"],
        linestyle=calc_style["linestyle"],
        label=r"NE $k_{sw}$ = %.3g $MPa^{-1}$" % ksw,
    )
    # EQ
    ax.plot(
        MW2_list,
        solubility_EQ,
        color=calc_style["color"],
        marker="None",
        linestyle="dashed",
        label="EQ",
    )

    # labelling
    ax.set_xlabel(r"$MW_{2}$ (g/mol)")
    ax.set_ylabel(r"Solubility ($g_{CO_{2}} / g_{pol\:am}$)")
    ax.set_title(r"$CO_{2}$-%s at %s°C, %s MPa" % (pol, T - 273, p * 1e-6))
    # styling
    # ax.set_ylim(bottom=0)
    # ax.grid(visible=True)
    ax.tick_params(direction="in")
    ax.legend().set_visible(True)
    plt.show()


def plot_ksw_sensitivity_EQvNE(
    ksw_l: float,
    ksw_u: float,
    no_of_points: int,
    T: float,
    p: list[float],
    sol: str,
    pol: str,
    MW2: float = None,
    rho20: float = None,
):
    """Plot solubility vs. k_sw for both NE- and EQ- EoS, for multiple pressure values.

    Args:
        ksw_l (float): lower k_sw [MPa^-1]
        ksw_u (float): upper k_sw [MPa^-1]
        no_of_points (int): number of points between ksw_l and ksw_u
        T (float): temperature [K]
        p (list[float]): pressure [Pa]
        pol (str): polymer
        MW2 (float, optional): MW of polymer [g/mol]. Defaults to None.
    """
    (
        _eos_mix,
        _eos_sol,
        _MW_1,
        _MW_2,
        _MW_monomer,
        rho_2_am_dry,
        _k_sw,
    ) = get_mixture_info(sol, pol, MW2)
    MW2 = MW2 if MW2 != None else _MW_2
    rho20 = rho20 if rho20 != None else rho_2_am_dry
    ksw_list = linspace(ksw_l, ksw_u, no_of_points)
    # storing the solubility at all pressure in 1 array
    solubility_NE = [None for i in range(len(p))]  # empty array with same size as input p
    solubility_EQ = [None for i in range(len(p))]
    i = 0
    while i < len(p):
        solubility_NE[i] = [solve_solubility_NE(T, p[i], sol, pol, MW2, k_sw_, rho20) for k_sw_ in ksw_list]
        solubility_EQ[i] = solve_solubility_EQ(T, p[i], sol, pol, MW2)
        i += 1

    print("\np = ", p)
    print("ksw_list = ", ksw_list)
    print("solubility_NE = ", solubility_NE)
    print("solubility_EQ = ", solubility_EQ)
    # Plotting

    fig = plt.figure()
    ax = fig.add_subplot(111)
    i = 0
    while i < len(p):
        # NE
        ax.plot(
            ksw_list,
            solubility_NE[i],
            color=custom_colours[i],
            marker=calc_style["marker"],
            linestyle=calc_style["linestyle"],
            label=r"NE %s MPa" % (p[i] * 1e-6),
        )
        # EQ
        ax.hlines(
            y=solubility_EQ[i],
            xmin=ksw_list[0],
            xmax=ksw_list[-1],
            color=custom_colours[i],
            linestyle="dashed",
            label=r"EQ %s MPa" % (p[i] * 1e-6),
        )
        i += 1
    # labelling
    ax.set_xlabel(r"$k_{sw}$ ($MPa^{-1}$)")
    ax.set_ylabel(r"Solubility ($g_{CO_{2}} / g_{pol\:am}$)")
    ax.set_title(r"$CO_{2}$-%s at %s°C" % (pol, T - 273))
    txt = r"$MW_{2}$ = %g g/mol" % MW2
    fig.text(0.05, 0.05, txt, ha="left", fontstyle="italic", fontsize="x-small")  # captioning
    # styling
    ax.set_ylim(bottom=0)
    # ax.grid(visible=True)
    ax.tick_params(direction="in")
    ax.legend(fontsize="x-small").set_visible(True)
    plt.show()


def plot_isotherm_NE(
    sol: str,
    pol: str,
    T: float,
    p_l: float,
    p_u: float,
    no_of_points: int,
    ksw_list: list[float],
    MW2: float = None,
    rho20: float = None,
    xlxs_sheet_refno_list: list[str] = None,
    display_plot: bool = True,
    save_plot_dir: str = None,
):
    """Plot solubility vs. k_sw for NE version of EoS, for ONE pressure value.

    Args:
        ksw_l (flt): lower k_sw [MPa^-1]
        ksw_u (flt): upper k_sw [MPa^-1]
        no_of_points (int): number of points between ksw_l abd ksw_u
        T (flt): temperature [K]
        p (flt): pressure [Pa]
        pol (str): polymer
        S_1_ref (_type_, optional): _description_. Defaults to None.
    """
    (
        _eos_mix,
        _eos_sol,
        _MW_1,
        _MW_2,
        _MW_monomer,
        rho_2_am_dry,
        _k_sw,
    ) = get_mixture_info(sol, pol)
    MW2 = MW2 if MW2 != None else _MW_2
    rho20 = rho20 if rho20 != None else rho_2_am_dry
    # import exp data

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
            search_pattern = f"^S_{T-273}C (.*)"  # strat with {T}C
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

    # calculating solubility
    MW2 = MW2 if MW2 != None else _MW_2
    p_calc = linspace(p_l, p_u, no_of_points)
    p_MPa_calc = p_calc * 1e-6  # [MPa]
    solubility_calc_list = [None for i in range(len(ksw_list))]
    p_MPa_exp_list = [None for i in range(len(matched_sheets))]
    solubility_exp_list = [None for i in range(len(matched_sheets))]

    for i, ksw_ in enumerate(ksw_list):
        solubility_calc_list[i] = [solve_solubility_NE(T, _p_, sol, pol, MW2, ksw_list[i], rho20) for _p_ in p_calc]
        print("solubility at ksw=%g =\t" % ksw_list[i], solubility_calc_list[i])

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # calculated solubility
    for i, ksw_ in enumerate(ksw_list):
        ax.plot(
            p_MPa_calc,
            solubility_calc_list[i],
            color=custom_colours[i],
            marker=calc_style["marker"],
            linestyle=calc_style["linestyle"],
            label=r"$k_{sw}$ = %.3g $MPa^{-1}$" % (ksw_list[i]),
        )

    # Experimental data
    if hasExpData == True:
        for i, sheet in enumerate(matched_sheets):
            p_MPa_exp_list[i] = asarray(dict[sheet]["P [MPa]"])
            solubility_exp_list[i] = asarray(dict[sheet]["Solubility [g-sol/g-pol-am]"])
            ax.plot(
                p_MPa_exp_list[i],
                solubility_exp_list[i],
                color=exp_style["color"],
                marker=custom_markers[i],
                linestyle=exp_style["linestyle"],
                markerfacecolor=exp_style["markerfacecolor"],
                label=ref_ID[i],
            )
    # labelling
    ax.set_xlabel(r"p (MPa)")
    ax.set_ylabel(r"Solubility ($g_{sol} / g_{pol\:am}$)")
    ax.set_title("{:s}-{:s} at {:.0f}°C".format(sol, pol, T - 273))
    ax.annotate(
        r"$\rho_{20}$ = %.3f $g/cm^{-3}$" % rho20,
        xy=(1.0, -0.13),
        xycoords="axes fraction",
        ha="right",
        va="center",
        fontsize="x-small",
    )
    # styling
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    # ax.grid(visible=True)
    ax.tick_params(direction="in")
    ax.legend().set_visible(True)
    if display_plot == True:
        plt.show()
    if save_plot_dir != None:
        plt.savefig(save_plot_dir, dpi=1200, transparent=True)
        print(f"Plot saved: {save_plot_dir}")
        print("")


def plot_p_sensitivity_NE(
    p_l: float,
    p_u: float,
    no_of_points: int,
    T: float,
    sol: str,
    pol: str,
    ksw: float,
    MW2: float = None,
    rho20: float = None,
):
    (
        _eos_mix,
        _eos_sol,
        _MW_1,
        MW_2,
        _MW_monomer,
        rho_2_am_dry,
        _k_sw,
    ) = get_mixture_info(sol, pol)
    MW2 = MW2 if MW2 != None else MW_2
    rho20 = rho20 if rho20 != None else rho_2_am_dry
    p_list = linspace(p_l, p_u, no_of_points)
    solubility_list = [solve_solubility_NE(T, p_, sol, pol, MW2, ksw, rho20) for p_ in p_list]
    # print("\np_list = ", p_list)
    # print("solubility_list = ", solubility_list)

    print("")
    print("MW2 = %g g/mol" % MW_2)
    print("k_sw = %g MPa^-1" % ksw)
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # calculated solubility
    p_MPa_list = p_list * 1e-6
    ax.plot(
        p_MPa_list,
        solubility_list,
        color=calc_style["color"],
        marker=calc_style["marker"],
        linestyle=calc_style["linestyle"],
        label=r"NE-SAFT-$\gamma$ Mie" + "\n$k_{sw}$ = %s $MPa^{-1}$" % ksw,
    )

    # referencee solubility
    # TODO add try blco and hasExpData boolean
    databasepath = os.path.join(os.path.dirname(__file__), "litdata")
    databasepath += "/%s-%s.xlsx" % (sol, pol)
    file = pd.ExcelFile(databasepath, engine="openpyxl")
    df = pd.read_excel(file, "%sC" % (T - 273))
    p_MPa_lit = df["P [MPa]"]
    solubility_ref = df["Solubility [g-sol/g-pol-am]"]
    ax.plot(
        p_MPa_lit,
        solubility_ref,
        color="red",
        marker="o",
        markerfacecolor="None",
        linestyle="None",
        label=r"exp data",
    )
    # labelling
    ax.set_xlabel(r"p (MPa)")
    ax.set_ylabel(r"Solubility ($g_{CO_{2}} / g_{pol\:am}$)")
    ax.set_title(r"$CO_{2}$-%s at %s°C" % (pol, T - 273))
    # styling
    ax.set_ylim(bottom=0)
    # ax.grid(visible=True)
    ax.tick_params(direction="in")
    ax.legend().set_visible(True)
    plt.show()


def get_fitting_AAD(Y_exp: list[float], Y_calc: list[float]) -> float:  # TODO use *arg for input list
    if len(Y_exp) != len(Y_calc):
        print("Error in %AAD calculation: different input dimension. ")
        return None
    n = len(Y_exp)
    AAD_cumulative = 0

    for i in range(len(Y_exp)):
        AAD_cumulative += abs((Y_exp[i] - Y_calc[i]) / Y_exp[i])
    AAD = AAD_cumulative / n
    return AAD


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
    try:
        AAD_percent = get_fitting_AAD(solubility_exp, solubility_calc_evaluation) * 100  # [%]
    except:
        AAD_percent = 0
    print("Fitting error: AAD%% = %.1f%%" % (AAD_percent))
    print("")

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
        label=r"NE $k_{sw} = %.3g \, MPa^{-1}$ (AAD%%=%.1f%%)" % (ksw_f, AAD_percent),
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

    return ksw_f[0], AAD_percent


def fit_rho20_NE(
    T: float,
    sol: str,
    pol: str,
    xlxs_sheet_refno: str,
    MW2: float = None,
    rho20_x0_list: list[float] = linspace(0.7, 1.1, 10),
    display_plot: bool = True,
    save_plot_dir: str = None,
) -> tuple[float, float]:
    """_summary_

    Args:
        T (_type_): _description_
        sol (_type_): _description_
        pol (_type_): _description_
        xlxs_sheet_refno (int): _description_
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
    MW2 = MW2 if MW2 != None else MW_2  # [g/mol]

    # Read solubility data
    ref_no = xlxs_sheet_refno
    xlxs_sheet = f"S_{T-273}C ({ref_no})"
    try:  # more specific spreadsheet
        path = os.path.join(os.path.dirname(__file__), "litdata")
        refpath = path + "/references.xlsx"
        databasepath = path + "/%s-%s.xlsx" % (sol, pol)
        datafile = pd.ExcelFile(databasepath, engine="openpyxl")
        df = pd.read_excel(datafile, xlxs_sheet)
        df.dropna(subset=["P [MPa]"], inplace=True)

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

    # Data from 2 smallest positive pressure for fitting
    df1 = df.nsmallest(2, "P [MPa]")
    print("Pressure used (MPa): ", df1["P [MPa]"].values.tolist())
    print("")
    p_MPa_exp_trimmed = asarray(df1["P [MPa]"])
    p_exp_trimmed = p_MPa_exp_trimmed * 1e6  # [Pa]
    solubility_exp_trimmed = asarray(df1["Solubility [g-sol/g-pol-am]"])
    print("p_list = ", p_exp_trimmed)
    print("solubility = ", solubility_exp_trimmed)

    # Fitting rho20
    def rho20_fitting_func(p_list_, rho20_fit):  # use ksw=0
        solubility_list = [solve_solubility_NE(T, p_, sol, pol, MW2, 0, rho20_fit) for p_ in p_list_]  # [g/g]
        return solubility_list

    x0_list = rho20_x0_list
    # start time
    start_time = time.time()
    for i, _rho20_0_ in enumerate(x0_list):
        try:
            # print("rho20_0 = ", _rho20_0_)
            # Fitting function
            result, cov = curve_fit(
                rho20_fitting_func,
                p_exp_trimmed,
                solubility_exp_trimmed,
                p0=(_rho20_0_),
            )

        except Exception as e:
            print("Fitting unsuccessful for rho20_0 = ", _rho20_0_)
            if i == (len(x0_list) - 1):
                print("Error - curve_fit unsuccessful.")
            pass
        else:
            print("")
            print("Fitting done. Fitting time: \t %.0f seconds" % (time.time() - start_time))
            rho20_f = result  # [g/cm^3]
            print("rho20 = %g g/cm^3" % rho20_f)
            break

    # fitting error
    solubility_calc_evaluation = rho20_fitting_func(p_exp_trimmed, rho20_f)
    try:
        AAD_percent = get_fitting_AAD(solubility_exp_trimmed, solubility_calc_evaluation) * 100  # [%]
    except:
        AAD_percent = 0
    print("Fitting error: AAD%% = %.1f%%" % (AAD_percent))
    print("")

    # Calculate solubility from new rho20
    p_MPa_exp_full = asarray(df["P [MPa]"])
    p_exp_full = p_MPa_exp_full * 1e6  # [Pa]
    p_calc_full = linspace(0.01, p_exp_full[-1], 40)
    p_MPa_calc_full = p_calc_full * 1e-6  # [MPa]
    solubility_exp_full = asarray(df["Solubility [g-sol/g-pol-am]"])
    solubility_calc_full = rho20_fitting_func(p_calc_full, rho20_f)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # trimmed exp data (fitted)
    ax.plot(
        p_MPa_exp_trimmed,
        solubility_exp_trimmed,
        color="black",
        marker=exp_style["marker"],
        linestyle=exp_style["linestyle"],
        # markerfacecolor=exp_style["markerfacecolor"],
        label=f"fitting exp data: {ref_ID} ({ref_no})",
    )
    # remaining exp data
    ax.plot(
        p_MPa_exp_full[2:],  # skip first 2 used in trimmed data
        solubility_exp_full[2:],  # skip first 2 used in trimmed data
        color="grey",
        marker=exp_style["marker"],
        linestyle=exp_style["linestyle"],
        markerfacecolor=exp_style["markerfacecolor"],
        label=f"unused exp data: {ref_ID} ({ref_no})",
    )

    # calculated from fited rho20
    ax.plot(
        p_MPa_calc_full,
        solubility_calc_full,
        color="black",
        marker=calc_style["marker"],
        linestyle=calc_style["linestyle"],
        # label=r"NE $\rho_{20} = %.5g g/cm^{3}$"%(rho20))
        label=r"NE $\rho_{20} = %.4f \, g/cm^{3}$ (AAD%%=%.1f%%)" % (rho20_f, AAD_percent),
    )
    # labelling
    ax.set_xlabel(r"p (MPa)")
    ax.set_ylabel(r"Solubility ($g_{sol} / g_{pol\;am}$)")
    ax.set_title(r"%s-%s at %s°C" % (sol, pol, T - 273))

    # styling
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    # ax.grid(visible=True)
    ax.tick_params(direction="in")
    ax.legend().set_visible(True)

    if display_plot == True:
        plt.show()
    if save_plot_dir != None:
        plt.savefig(save_plot_dir, dpi=1200, transparent=True)
        print(f"Plot saved: {save_plot_dir}")
        print("")
    return rho20_f[0], AAD_percent


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

    # calculating solubility
    p_calc = linspace(p_l, p_u, no_of_points)  # [Pa]
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
        solubility_NE_calc_list[i] = [solve_solubility_NE(T, _p_, sol, pol, MW2, ksw_list[i], rho20) for _p_ in p_calc]
        print(
            "solubility at ksw=%g =\t" % ksw_list[i], solubility_NE_calc_list[i]
        )  # calculated NE solubility with ksw != 0

    # calculated EQ solubility
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
        # calculate AAD for NE and EQ when there is only 1 exp data sheet
        if len(matched_sheets) == 1:
            p_exp_list = p_MPa_exp_list[0] * 1e6  # [Pa]
            sol_exp_list = solubility_exp_list[0]
            solubility_calc_evaluation_EQ = [solve_solubility_EQ(T, p_, sol, pol, MW2) for p_ in p_exp_list]
            try:
                AAD_percent_EQ = get_fitting_AAD(sol_exp_list, solubility_calc_evaluation_EQ) * 100  # [%]
            except:
                AAD_percent_EQ = 0
            print("AAD%% for EQ: AAD%% = %.1f%%" % (AAD_percent_EQ))
            label_EQ += " (AAD%%=%.1f%%)" % AAD_percent_EQ
            for i, ksw_ in enumerate(ksw_list):
                solubility_calc_evaluation_NE_list[i] = [
                    solve_solubility_NE(T, _p_, sol, pol, MW2, ksw_list[i], rho20) for _p_ in p_exp_list
                ]
                try:
                    AAD_percent_NE[i] = (
                        get_fitting_AAD(sol_exp_list, solubility_calc_evaluation_NE_list[i]) * 100
                    )  # [%]
                except:
                    AAD_percent_NE[i] = 0
                print("AAD%% for NE ksw=%g: AAD%% = %.1f%%" % (ksw_, AAD_percent_NE[i]))
                label_NE[i] += " (AAD%%=%.1f%%)" % AAD_percent_NE[i]

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

    # EQ solubility
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
    ax.set_ylabel(r"Solubility ($g_{sol} / g_{pol\:am}$)")
    ax.set_title("%s-%s at %.0f°C " % (sol, pol, T - 273))
    ax.annotate(
        r"NE $\rho_{20}$ = %.4f $g/cm^{-3}$" % rho20,
        xy=(1.0, -0.09),
        xycoords="axes fraction",
        ha="right",
        va="center",
        fontsize="xx-small",
    )
    # styling
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    # ax.grid(visible=True)
    ax.tick_params(direction="in")
    legend_ncol = 1 if (len(matched_sheets) + len(ksw_list)) < 5 else 2
    ax.legend(ncol=legend_ncol).set_visible(True)
    if display_plot == True:
        plt.show()
    if save_plot_dir != None:
        plt.savefig(save_plot_dir, dpi=1200, transparent=True)
        print(f"Plot saved: {save_plot_dir}")
        print("")


def plot_isobar_EQvsNE(
    T_l: float,
    T_u: float,
    no_of_points: int,
    p: float,
    sol: str,
    pol: str,
    ksw: float,
    MW2: float = None,
    rho20: float = None,
):
    (
        _eos_mix,
        _eos_sol,
        _MW_1,
        _MW_2,
        _MW_monomer,
        rho_2_am_dry,
        _k_sw,
    ) = get_mixture_info(sol, pol, MW2)
    # calculating solubility
    MW2 = MW2 if MW2 != None else _MW_2
    rho20 = rho20 if rho20 != None else rho_2_am_dry

    T_calc = linspace(T_l, T_u, no_of_points)
    # calculated NE solubility with ksw != 0
    solubility_NE_w_ksw_list = [solve_solubility_NE(T_, p, sol, pol, MW2, ksw, rho20) for T_ in T_calc]
    # calculated EQ solubility
    solubility_EQ_list = [solve_solubility_EQ(T_, p, sol, pol, MW2) for T_ in T_calc]
    print("T_cal = ", T_calc)
    print("\nsolubility_NE_w_ksw = ", solubility_NE_w_ksw_list)
    print("\nsolubility_EQ = ", solubility_EQ_list)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # NE ksw != 0
    ax.plot(
        T_calc,
        solubility_NE_w_ksw_list,
        color=calc_style["color"],
        marker=calc_style["marker"],
        linestyle=calc_style["linestyle"],
        label=r"NE $k_{sw}$ = %.3g $MPa^{-1}$" % ksw,
    )
    # EQ
    ax.plot(
        T_calc,
        solubility_EQ_list,
        color=calc_style["color"],
        marker="None",
        linestyle="dashed",
        label="EQ",
    )
    # labelling
    ax.set_xlabel(r"T (K)")
    ax.set_ylabel(r"Solubility ($g_sol / g_{pol\;am}$)")
    ax.set_title(r"%s-%s at %g MPa" % (sol, pol, p * 1e-6))
    # styling
    ax.set_ylim(bottom=0)
    # ax.set_xlim(left=0)
    # ax.grid(visible=True)
    ax.tick_params(direction="in")
    ax.legend().set_visible(True)
    plt.show()


def plot_isotherm_EQ(
    p_l: float,
    p_u: float,
    no_of_points: int,
    T_list: list[float],
    sol: str,
    pol: str,
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
    ) = get_mixture_info(sol, pol, MW2)
    MW2 = MW2 if MW2 != None else _MW_2

    p_calc = linspace(p_l, p_u, no_of_points)
    p_MPa_calc = p_calc * 1e-6
    print("p_cal = ", p_calc)

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
                    search_pattern = f"^S_{T-273}C.*({j})"
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

    for i, T in enumerate(T_list):
        # calculated EQ solubility
        solubility_EQ[i] = [solve_solubility_EQ(T, p_, sol, pol, MW2) for p_ in p_calc]
        print("\nsolubility_EQ at %s°C = " % (T - 273), solubility_EQ[i])

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    nlegendcount = 0
    for i, T in enumerate(T_list):
        # EQ calc
        ax.plot(
            p_MPa_calc,
            solubility_EQ[i],
            color=custom_colours[i],
            marker="None",
            linestyle="solid",
            label=f"EQ model {T-273}°C",
        )
        nlegendcount += 1
        # exp data
        if hasExpData[i] == True:
            for j, sheet in enumerate(matched_sheets[i]):
                ax.plot(
                    dict[sheet]["P [MPa]"],
                    dict[sheet]["Solubility [g-sol/g-pol-am]"],
                    color=custom_colours[i],
                    marker=custom_markers[j],
                    linestyle="None",
                    markerfacecolor="None",
                    label=f"exp {T-273}°C: {ref_ID[i][j]} ({ref_no[i][j]})",
                )
                nlegendcount += 1

    # labelling
    ax.set_xlabel(r"p (MPa)")
    ax.set_ylabel(r"Solubility ($g_{sol} / g_{pol\;am}$)")
    ax.set_title(r"%s-%s" % (sol, pol))
    # styling
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    # ax.grid(visible=True)
    ax.tick_params(direction="in")
    legend_ncol = 1 if (nlegendcount) < 5 else 2
    print("legend columns = ", nlegendcount)
    ax.legend(ncol=legend_ncol).set_visible(True)
    ax.legend(fontsize="xx-small").set_visible(True)
    if display_plot == True:
        plt.show()
    if save_plot_dir != None:
        plt.savefig(save_plot_dir, dpi=1200, transparent=True)
        print(f"Plot saved: {save_plot_dir}")
        print("")


def plot_isotherm_EQ_molFraction(
    p_l: float,
    p_u: float,
    no_of_points: int,
    T_list: list[float],
    sol: str,
    pol: str,
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
    ) = get_mixture_info(sol, pol, MW2)
    MW2 = MW2 if MW2 != None else _MW_2

    p_calc = linspace(p_l, p_u, no_of_points)
    p_MPa_calc = p_calc * 1e-6
    print("p_cal = ", p_calc)

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
                    search_pattern = f"^S_{T-273}C.*({j})"
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

    for i, T in enumerate(T_list):
        # calculated EQ solubility (mol fraction)
        solubility_EQ[i] = [
            solve_solubility_EQ(T, p_, sol, pol, MW2, return_extended=True)[1] for p_ in p_calc
        ]  # [mol/mol]
        print("\nsolubility_EQ at %s°C = " % (T - 273), solubility_EQ[i])

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    nlegendcount = 0
    for i, T in enumerate(T_list):
        # EQ calc
        ax.plot(
            p_MPa_calc,
            solubility_EQ[i],
            color=custom_colours[i],
            marker="None",
            linestyle="solid",
            label=f"EQ model {T-273}°C",
        )
        nlegendcount += 1

    # labelling
    ax.set_xlabel(r"p (MPa)")
    ax.set_ylabel(r"Mol fraction ($mol_{sol} / mol_{mix}$)")
    ax.set_title(r"%s-%s" % (sol, pol))
    # styling
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    # ax.grid(visible=True)
    ax.tick_params(direction="in")
    legend_ncol = 1 if (nlegendcount) < 5 else 2
    print("legend columns = ", nlegendcount)
    ax.legend(ncol=legend_ncol).set_visible(True)
    ax.legend(fontsize="xx-small").set_visible(True)
    if display_plot == True:
        plt.show()
    if save_plot_dir != None:
        plt.savefig(save_plot_dir, dpi=1200, transparent=True)
        print(f"Plot saved: {save_plot_dir}")
        print("")


def plot_isotherm_EQ_rhoL(
    p_l: float,
    p_u: float,
    no_of_points: int,
    T_list: list[float],
    sol: str,
    pol: str,
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
    ) = get_mixture_info(sol, pol, MW2)
    MW2 = MW2 if MW2 != None else _MW_2

    p_calc = linspace(p_l, p_u, no_of_points)
    p_MPa_calc = p_calc * 1e-6
    print("p_cal = ", p_calc)

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
                    search_pattern = f"^S_{T-273}C.*({j})"
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

    for i, T in enumerate(T_list):
        # calculated EQ solubility (mol-mix/m^3-mix)
        solubility_EQ[i] = [
            solve_solubility_EQ(T, p_, sol, pol, MW2, return_extended=True)[2] for p_ in p_calc
        ]  # [mol/m^3]
        print("\nsolubility_EQ at %s°C = " % (T - 273), solubility_EQ[i])

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    nlegendcount = 0
    for i, T in enumerate(T_list):
        # EQ calc
        ax.plot(
            p_MPa_calc,
            solubility_EQ[i],
            color=custom_colours[i],
            marker="None",
            linestyle="solid",
            label=f"EQ model {T-273}°C",
        )
        nlegendcount += 1

    # labelling
    ax.set_xlabel(r"p (MPa)")
    ax.set_ylabel(r"Mixture density ($mol_{mix} / m^{3}_{mix}$)")
    ax.set_title(r"%s-%s" % (sol, pol))
    # styling
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    # ax.grid(visible=True)
    ax.tick_params(direction="in")
    legend_ncol = 1 if (nlegendcount) < 5 else 2
    print("legend columns = ", nlegendcount)
    ax.legend(ncol=legend_ncol).set_visible(True)
    ax.legend(fontsize="xx-small").set_visible(True)
    if display_plot == True:
        plt.show()
    if save_plot_dir != None:
        plt.savefig(save_plot_dir, dpi=1200, transparent=True)
        print(f"Plot saved: {save_plot_dir}")
        print("")


def plot_isotherm_NELF(
    p_l: float,
    p_u: float,
    no_of_points: int,
    T: float,
    sol: str,
    pol: str,
    kij: float,
    ksw: float,
):
    try:
        # import exp data
        databasepath = os.path.join(os.path.dirname(__file__), "litdata")
        databasepath += "/%s-%s.xlsx" % (sol, pol)
        file = pd.ExcelFile(databasepath, engine="openpyxl")
        # exp data
        df = pd.read_excel(file, "%sC" % (T - 273))
        p_MPa_exp = asarray(df["P [MPa]"])
        solubility_exp = asarray(df["Solubility [g-sol/g-pol-am]"])
        hasExpData = True
    except Exception as e:
        hasExpData = False
        print("")
        print("Error - importing exp data failed:")
        print(e)
    print("hasExpData = ", hasExpData)
    # calculate solubility
    p_calc = linspace(p_l, p_u, no_of_points)
    p_MPa_calc = p_calc * 1e-6
    # calculated EQ solubility
    solubility_NELF = [NELF.solve_solubility_NELF(sol, pol, T, p_, 1.262, kij, ksw) for p_ in p_calc]
    print("p_cal = ", p_calc)
    print("\nsolubility_NELF = ", solubility_NELF)
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # EQ
    ax.plot(
        p_MPa_calc,
        solubility_NELF,
        color=calc_style["color"],
        marker="None",
        linestyle="dashed",
        label="EQ",
    )
    # exp data
    if hasExpData == True:
        ax.plot(
            p_MPa_exp,
            solubility_exp,
            color=exp_style["color"],
            marker=exp_style["marker"],
            linestyle=exp_style["linestyle"],
            markerfacecolor=exp_style["markerfacecolor"],
            label="exp data",
        )
    # labelling
    ax.set_xlabel(r"p (MPa)")
    ax.set_ylabel(r"Solubility ($g_{sol} / g_{pol\;am}$)")
    ax.set_title(r"%s-%s at %s°C (NELF)" % (sol, pol, T - 273))
    # styling
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    # ax.grid(visible=True)
    ax.tick_params(direction="in")
    ax.legend().set_visible(True)
    plt.show()


def plot_isotherm_NELFvNESAFT(
    p_l: float,
    p_u: float,
    no_of_points: int,
    T: float,
    sol: str,
    pol: str,
    rho20_NELF: float,
    kij_NELF: float,
    ksw_NELF: float,
    rho20_NESAFT: float,
    ksw_NESAFT: list[float],
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
    ) = get_mixture_info(sol, pol)
    MW2 = MW2 if MW2 != None else _MW_2
    # import exp data
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
            search_pattern = f"^S_{T-273}C (.*)"  # strat with {T}C
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
    p_MPa_exp_list = [None for i in range(len(matched_sheets))]
    solubility_exp_list = [None for i in range(len(matched_sheets))]

    # calculate solubility
    p_calc = linspace(p_l, p_u, no_of_points)
    p_MPa_calc = p_calc * 1e-6
    print("p_cal = ", p_calc)
    # calculated NELF solubility
    solubility_NELF = [NELF.solve_solubility_NELF(sol, pol, T, p_, rho20_NELF, kij_NELF, ksw_NELF) for p_ in p_calc]
    print("\nsolubility_NELF = ", solubility_NELF)

    # calculate NE-SAFT solubility
    solubility_NESAFT = [solve_solubility_NE(T, _p_, sol, pol, MW2, ksw_NESAFT, rho20_NESAFT) for _p_ in p_calc]
    print("\nsolubility_NE-SAFT = ", solubility_NESAFT)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # NELF
    ax.plot(
        p_MPa_calc,
        solubility_NELF,
        color=custom_colours[0],
        marker="None",
        linestyle="solid",
        label=r"NELF" + "\n" + r"($k_{sw}$ = %.3g $MPa^{-1}$, $\rho_{20}$ = %.3f $g/cm^{3}$)" % (ksw_NELF, rho20_NELF),
    )
    # NE-SAFT
    ax.plot(
        p_MPa_calc,
        solubility_NESAFT,
        color=custom_colours[1],
        marker="None",
        linestyle="solid",
        label=r"NE-SAFT"
        + "\n"
        + r"($k_{sw}$ = %.3g $MPa^{-1}$, $\rho_{20}$ = %.3f $g/cm^{3}$)" % (ksw_NESAFT, rho20_NESAFT),
    )
    # Experimental data
    if hasExpData == True:
        for i, sheet in enumerate(matched_sheets):
            p_MPa_exp_list[i] = asarray(dict[sheet]["P [MPa]"])
            solubility_exp_list[i] = asarray(dict[sheet]["Solubility [g-sol/g-pol-am]"])
            ax.plot(
                p_MPa_exp_list[i],
                solubility_exp_list[i],
                color=exp_style["color"],
                marker=custom_markers[i],
                linestyle=exp_style["linestyle"],
                markerfacecolor=exp_style["markerfacecolor"],
                label=f"exp data: {ref_ID[i]} ({ref_no[i]})",
            )
    # labelling
    ax.set_xlabel(r"p (MPa)")
    ax.set_ylabel(r"Solubility ($g_{sol} / g_{pol\;am}$)")
    ax.set_title(r"%s-%s at %s°C" % (sol, pol, T - 273))
    # styling
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.tick_params(direction="in")
    ax.legend().set_visible(True)

    if display_plot == True:
        plt.show()
    if save_plot_dir != None:
        plt.savefig(save_plot_dir, dpi=1200, transparent=True)
        print(f"Plot saved: {save_plot_dir}")
        print("")


def plot_density_isotherm_EQvNE(
    p_l: float,
    p_u: float,
    no_of_points: int,
    T: float,
    sol: str,
    pol: str,
    ksw_list: list[float],
    MW2: float = None,
    rho20: float = None,
):
    (
        _eos_mix,
        _eos_sol,
        _MW_1,
        _MW_2,
        _MW_monomer,
        rho_2_am_dry,
        _k_sw,
    ) = get_mixture_info(sol, pol, MW2)
    MW2 = MW2 if MW2 != None else _MW_2
    rho20 = rho20 if rho20 != None else rho_2_am_dry

    p_calc = linspace(p_l, p_u, no_of_points)  # [Pa]
    print("p_cal = ", p_calc)
    p_MPa_calc = p_calc * 1e-6
    # EQ density
    rho2_EQ_list = [solve_solubility_EQ(T, p_, sol, pol, MW2, return_extended=True)[3] for p_ in p_calc]
    print("rho2_EQ = ", rho2_EQ_list)

    # NE density with ksw
    rho2_NE_list = [None for i in range(len(ksw_list))]
    for i, _ksw in enumerate(ksw_list):
        rho2_NE_list_i = [None for i in range(len(p_calc))]
        for j, p_ in enumerate(p_calc):
            try:
                rho2_NE_list_i[j] = solve_solubility_NE(T, p_, sol, pol, MW2, _ksw, rho20, return_extended=True)[3]
            except:
                rho2_NE_list_i[j] = None
        rho2_NE_list[i] = rho2_NE_list_i
        print(f"rho2_NE at {_ksw} = ", rho2_NE_list[i])
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # EQ
    ax.plot(
        p_MPa_calc,
        rho2_EQ_list,
        color=calc_style["color"],
        marker="None",
        linestyle="dashed",
        label="EQ",
    )
    # NE
    for i, _rho2_ in enumerate(rho2_NE_list):
        ax.plot(
            p_MPa_calc,
            rho2_NE_list[i],
            color=custom_colours[i],
            marker=calc_style["marker"],
            linestyle=calc_style["linestyle"],
            label=r"NE $k_{sw}$ = %.3g $MPa^{-1}$" % (ksw_list[i]),
        )
    # labelling
    ax.set_xlabel(r"p (MPa)")
    ax.set_ylabel(r"$\rho_{2}$ ($g/cm^{-3}$)")
    ax.set_title(r"%s-%s at %s°C" % (sol, pol, T - 273))
    ax.annotate(
        r"$\rho_{20}$ = %.3f $g/cm^{-3}$" % rho20,
        xy=(1.0, -0.13),
        xycoords="axes fraction",
        ha="right",
        va="center",
        fontsize="x-small",
    )
    # styling
    # ax.set_ylim(bottom=0)
    # ax.set_xlim(left=0)
    ax.tick_params(direction="in")
    ax.legend(fontsize="x-small").set_visible(True)
    fig.tight_layout()
    plt.show()


def plot_V_isotherm_EQvNE(
    p_l: float,
    p_u: float,
    no_of_points: int,
    T: float,
    sol: str,
    pol: str,
    ksw_list: list[float],
    rho20: float = None,
    MW2: float = None,
    display_plot: bool = True,
    save_plot_dir: str = None,
) -> list[float]:
    (
        _eos_mix,
        _eos_sol,
        _MW_1,
        _MW_2,
        _MW_monomer,
        rho_2_am_dry,
        _k_sw,
    ) = get_mixture_info(sol, pol, MW2)
    MW2 = MW2 if MW2 != None else _MW_2
    rho20 = rho20 if rho20 != None else rho_2_am_dry

    p_calc = linspace(p_l, p_u, no_of_points)  # [Pa]
    print("p_cal = ", p_calc)
    p_MPa_calc = p_calc * 1e-6
    # EQ volume
    V2_EQ_list = [solve_solubility_EQ(T, p_, sol, pol, MW2, return_extended=True)[4] for p_ in p_calc]
    print("V2_EQ = ", V2_EQ_list)

    # NE volume with ksw
    V2_NE_list = [None for i in range(len(ksw_list))]
    for i, _ksw in enumerate(ksw_list):
        V2_NE_list_i = [None for i in range(len(p_calc))]
        for j, p_ in enumerate(p_calc):
            try:
                V2_NE_list_i[j] = solve_solubility_NE(T, p_, sol, pol, MW2, _ksw, rho20, return_extended=True)[4]
            except:
                V2_NE_list_i[j] = None
        V2_NE_list[i] = V2_NE_list_i
        print(f"V2_NE at {_ksw} = ", V2_NE_list[i])

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # EQ
    ax.plot(
        p_MPa_calc,
        V2_EQ_list,
        color=custom_colours[1],
        marker="None",
        linestyle=custom_linestyles[1],
        label="EQ",
    )
    # NE
    for i, _V2_ in enumerate(V2_NE_list):
        ax.plot(
            p_MPa_calc,
            V2_NE_list[i],
            color=custom_colours[2],
            marker="None",
            linestyle=custom_linestyles[i + 1],
            label=r"NE $k_{sw}$ = %.3g $MPa^{-1}$" % (ksw_list[i]),
        )

    # labelling
    ax.set_xlabel(r"p (MPa)")
    ax.set_ylabel(r"$\hat{V}_{2}$ ($cm^{3}/g$)")
    ax.set_title(r"%s-%s at %s°C" % (sol, pol, T - 273))
    ax.annotate(
        r"NE $\rho_{20}$ = %.3f $g/cm^{-3}$" % rho20,
        xy=(1.0, -0.09),
        xycoords="axes fraction",
        ha="right",
        va="center",
        fontsize="xx-small",
    )
    # styling
    # ax.set_ylim(bottom=0)
    # ax.set_xlim(left=0)
    ax.tick_params(direction="in")
    ax.legend().set_visible(True)
    fig.tight_layout()
    if display_plot == True:
        plt.show()
    if save_plot_dir != None:
        plt.savefig(save_plot_dir, dpi=1200, transparent=True)
        print(f"Plot saved: {save_plot_dir}")
        print("")


def plot_V_isotherm_EQvNEvCombined(
    p_l: float,
    p_u: float,
    no_of_points: int,
    T: float,
    sol: str,
    pol: str,
    ksw_list: list[float],
    rho20: float = None,
    MW2: float = None,
    display_plot: bool = True,
    save_plot_dir: str = None,
) -> list[float]:
    (
        _eos_mix,
        _eos_sol,
        _MW_1,
        _MW_2,
        _MW_monomer,
        rho_2_am_dry,
        _k_sw,
    ) = get_mixture_info(sol, pol, MW2)
    MW2 = MW2 if MW2 != None else _MW_2
    rho20 = rho20 if rho20 != None else rho_2_am_dry

    p_calc = linspace(p_l, p_u, no_of_points)  # [Pa]
    print("p_cal = ", p_calc)
    p_MPa_calc = p_calc * 1e-6
    # EQ volume
    V2_EQ_list = [solve_solubility_EQ(T, p_, sol, pol, MW2, return_extended=True)[4] for p_ in p_calc]
    print("V2_EQ = ", V2_EQ_list)

    # NE volume with ksw
    V2_NE_list = [None for i in range(len(ksw_list))]
    for i, _ksw in enumerate(ksw_list):
        V2_NE_list_i = [None for i in range(len(p_calc))]
        for j, p_ in enumerate(p_calc):
            try:
                V2_NE_list_i[j] = solve_solubility_NE(T, p_, sol, pol, MW2, _ksw, rho20, return_extended=True)[4]
            except:
                V2_NE_list_i[j] = None
        V2_NE_list[i] = V2_NE_list_i
        print(f"V2_NE at {_ksw} = ", V2_NE_list[i])

    # Combined results from EQ and NE using p interect
    V2_combined_list = [None for i in ksw_list]  # placeholder
    has_combined_data = [None for i in ksw_list]  # placeholder
    p_intersect_list = [None for i in ksw_list]  # placeholder
    S_NE_intersect_list = [None for i in ksw_list]  # placeholder
    S_EQ_intersect_list = [None for i in ksw_list]  # placeholder
    Sg_intersect_list = [None for i in ksw_list]  # placeholder
    try:
        p_intersect_list = [
            min(get_p_intersect_EQvNE(T, sol, pol, ksw=ksw_, rho20=rho20, MW2=MW2)) for ksw_ in ksw_list
        ]
        print("pg = ", p_intersect_list)
    except:
        print("Error: no pg found.")
        has_combined_data[i] = False

    else:
        # NE item in combined list
        for i, _ksw_ in enumerate(ksw_list):
            V2_combined_list_i = []
            for j, _p_ in enumerate(p_calc):
                if _p_ < p_intersect_list[i]:
                    V2_combined_list_i.append(V2_NE_list[i][j])
                else:
                    V2_combined_list_i.append(V2_EQ_list[j])
            V2_combined_list[i] = V2_combined_list_i
            has_combined_data[i] = True
            S_NE_intersect_list[i] = solve_solubility_NE(T, p_intersect_list[i], sol, pol, MW2, _ksw_, rho20)
            S_EQ_intersect_list[i] = solve_solubility_EQ(T, p_intersect_list[i], sol, pol, MW2)
            # check if EQ and NE solubility is equal (i.e. within 1% difference)
            if isclose(
                S_NE_intersect_list[i],
                S_EQ_intersect_list[i],
                S_NE_intersect_list[i] * 0.01,
            ):
                Sg_intersect_list[i] = (S_NE_intersect_list[i] + S_EQ_intersect_list[i]) / 2

        print("V2 combined = ", V2_combined_list)
        print("Sg  = ", Sg_intersect_list)
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # EQ
    ax.plot(
        p_MPa_calc,
        V2_EQ_list,
        color=custom_colours[1],
        marker="None",
        linestyle=custom_linestyles[1],
        label="EQ",
    )
    # NE
    for i, _V2_ in enumerate(V2_NE_list):
        ax.plot(
            p_MPa_calc,
            V2_NE_list[i],
            color=custom_colours[2],
            marker="None",
            linestyle=custom_linestyles[i + 1],
            label=r"NE $k_{sw}$ = %.3g $MPa^{-1}$" % (ksw_list[i]),
        )
    # combined EQ and NE
    for i, _V2_ in enumerate(V2_combined_list):
        if has_combined_data[i] == True:
            ax.plot(
                p_MPa_calc,
                V2_combined_list[i],
                color=custom_colours[0],
                marker="None",
                linestyle=custom_linestyles[i],
                label=r"combined ($k_{sw}$ = %.3g $MPa^{-1}$)" % (ksw_list[i]),
            )
    # labelling
    ax.set_xlabel(r"p (MPa)")
    ax.set_ylabel(r"$\hat{V}_{2}$ ($cm^{3}/g$)")
    ax.set_title(r"%s-%s at %s°C" % (sol, pol, T - 273))
    ax.annotate(
        r"NE $\rho_{20}$ = %.3f $g/cm^{-3}$" % rho20,
        xy=(1.0, -0.09),
        xycoords="axes fraction",
        ha="right",
        va="center",
        fontsize="xx-small",
    )
    # styling
    # ax.set_ylim(bottom=0)
    # ax.set_xlim(left=0)
    ax.tick_params(direction="in")
    ax.legend().set_visible(True)
    fig.tight_layout()
    if display_plot == True:
        plt.show()
    if save_plot_dir != None:
        plt.savefig(save_plot_dir, dpi=1200, transparent=True)
        print(f"Plot saved: {save_plot_dir}")
        print("")
    return p_intersect_list, Sg_intersect_list


def plot_dV_V0_isotherm_EQvNE(
    p_l: float,
    p_u: float,
    no_of_points: int,
    T: float,
    sol: str,
    pol: str,
    ksw_list: list[float],
    MW2: float = None,
    rho20: float = None,
):  # TODO check this
    (
        _eos_mix,
        _eos_sol,
        _MW_1,
        _MW_2,
        _MW_monomer,
        rho_2_am_dry,
        _k_sw,
    ) = get_mixture_info(sol, pol, MW2)
    MW2 = MW2 if MW2 != None else _MW_2
    rho20 = rho20 if rho20 != None else rho_2_am_dry

    p_calc = linspace(p_l, p_u, no_of_points)  # [Pa]
    print("p_cal = ", p_calc)
    p_MPa_calc = p_calc * 1e-6
    # EQ density
    V2_EQ_list = [solve_solubility_EQ(T, p_, sol, pol, MW2, return_extended=True)[4] for p_ in p_calc]
    dV_V0_EQ_list = [None for i in range(len(V2_EQ_list))]
    for i, V in enumerate(V2_EQ_list):
        try:
            dV_V0_EQ_list[i] = (V - V2_EQ_list[0]) / V2_EQ_list[0]
        except:
            dV_V0_EQ_list[i] = None
    print("V2_EQ = ", V2_EQ_list)

    # NE density with ksw
    V2_NE_list = [None for i in range(len(ksw_list))]
    V20_NE_list = [None for i in range(len(ksw_list))]
    dV_V0_NE_list = [None for i in range(len(ksw_list))]
    for i, _ksw in enumerate(ksw_list):
        V2_NE_list_i = [None for i in range(len(p_calc))]
        dV_V0_NE_list_i = [None for i in range(len(p_calc))]
        for j, p_ in enumerate(p_calc):
            try:
                V2_NE_list_i[j] = solve_solubility_NE(T, p_, sol, pol, MW2, _ksw, rho20, return_extended=True)[4]
                dV_V0_NE_list_i[j] = (V2_NE_list_i[j] - V2_NE_list_i[0]) / V2_NE_list_i[0]  # [volume fraction]

            except:
                V2_NE_list_i[j] = None
                dV_V0_NE_list_i[j] = None
        V2_NE_list[i] = V2_NE_list_i
        V20_NE_list[i] = V2_NE_list[i][0]
        dV_V0_NE_list[i] = dV_V0_NE_list_i
        print(f"V2_NE at {_ksw} = ", V2_NE_list[i])
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # EQ
    ax.plot(
        p_MPa_calc,
        dV_V0_EQ_list,
        color=custom_colours[1],
        marker="None",
        linestyle="dashed",
        label="EQ",
    )
    # NE
    for i, _V2_ in enumerate(V2_NE_list):
        ax.plot(
            p_MPa_calc,
            dV_V0_NE_list[i],
            color=custom_colours[2],
            marker="None",
            linestyle=custom_linestyles[i + 1],
            label=r"NE $k_{sw}$ = %.3g $MPa^{-1}$" % (ksw_list[i]),
        )
    # labelling
    ax.set_xlabel(r"p (MPa)")
    ax.set_ylabel(r"$\Delta V_{2}/V_{20}$")
    ax.set_title(r"%s-%s at %s°C" % (sol, pol, T - 273))
    ax.annotate(
        r"NE $\rho_{20}$ = %.3f $g/cm^{-3}$" % rho20,
        xy=(1.0, -0.13),
        xycoords="axes fraction",
        ha="right",
        va="center",
        fontsize="x-small",
    )
    # styling
    # ax.set_ylim(bottom=0)
    # ax.set_xlim(left=0)
    ax.tick_params(direction="in")
    ax.legend(fontsize="x-small").set_visible(True)
    fig.tight_layout()
    plt.show()


def get_therm_factor(
    T: float,
    p: float,
    sol: str,
    pol: str,
    ksw: float,
    frc_step: float = 1e-3,
    rho20: float = None,
    MW2: float = None,
):
    """solve d(mu1/RT)/d(omega_1) numerically within **polymer-rich** phase only, and NOT solution to phase EQ.

    Args:
        T (_type_): _description_
        p (_type_): _description_
        sol (_type_): _description_
        pol (_type_): _description_
        frc_step : step length as fraction of ln(omega).
        k_sw (_type_, optional): _description_. Defaults to None.
        MW2 (_type_, optional): _description_. Defaults to None.
    """
    (
        eos_mix,
        _eos_sol,
        MW1,
        _MW_2,
        _MW_monomer,
        rho_2_am_dry,
        _k_sw,
    ) = get_mixture_info(sol, pol, MW2)
    MW2 = MW2 if MW2 != None else _MW_2
    rho20 = rho20 if rho20 != None else rho_2_am_dry

    # get S1 [g-sol/g-pol] and muad at phase equlibria
    (
        S1,
        omega_1,
        muad_1,
    ) = solve_solubility_NE(
        T, p, sol, pol, MW2, ksw, rho20, return_extended=True
    )[:3]

    def get_muad_NE(lnomega1):  # calculates muad from known ln(omega1) using SAFT-g Mie
        omega1 = exp(lnomega1)
        # calculate muad1 from known omega
        x_1 = MW2 / (MW1 / omega1 - MW1 + MW2)
        x = hstack([x_1, 1 - x_1])  # [mol/mol-mix]
        rhol = eos_mix.density(x, T, p, "L")  # * Topliss's method
        rho_i = x * rhol  # [mol/m^3-mix]
        muad1 = eos_mix.muad(rho_i, T)  # adim
        return muad1[0]

    # frc_step = 0.001  # step length as fraction of ln(omega)
    # print("\nomega1 = ", omega_1)
    # (l)ower muad and ln(omega1)
    lnomega1_l = log(omega_1) * (1 - frc_step)
    muad1_l = get_muad_NE(lnomega1_l)
    # print("ln(omega1)_l = ", lnomega1_l)
    # print("muad_l = ", muad1_l)
    # (u)pper muad and ln(omega1)
    lnomega1_u = log(omega_1) * (1 + frc_step)
    muad1_u = get_muad_NE(lnomega1_u)
    # print("ln(omega1)_u = ", lnomega1_u)
    # print("muad_u = ", muad1_u)

    dmuad1_dlnomega1 = (muad1_u - muad1_l) / (lnomega1_u - lnomega1_l)
    return dmuad1_dlnomega1


def get_FFV(pol: str, rho20: float):
    # Step 1: calculate occupied volume of polymer chains based on Bondi's method
    Vwi = {
        "CH2": 10.23,
        "CH(C6H5)": 52.6,
        "COO": 15.2,
        "CH(CH3)": 20.45,
        "C(CH3)(COOCH3)": 46.7,
    }  # group vdw volume [cm^3/mol]
    Mi = {
        "CH2": 14.03,
        "CH(C6H5)": 90.12,
        "COO": 44.01,
        "CH(CH3)": 28.05,
        "C(CH3)(COOCH3)": 86.05,
    }  # group MW (g/mol)
    if pol == "PS":  # total cm^3/mol / total g/cm^3
        V_w = (Vwi["CH2"] + Vwi["CH(C6H5)"]) / (Mi["CH2"] + Mi["CH(C6H5)"])
    elif pol == "PLA":
        V_w = (Vwi["COO"] + Vwi["CH(CH3)"]) / (Mi["COO"] + Mi["CH(CH3)"])
    elif pol == "PMMA":
        V_w = (Vwi["CH2"] + Vwi["C(CH3)(COOCH3)"]) / (Mi["CH2"] + Mi["C(CH3)(COOCH3)"])
    V_occ = 1.3 * V_w  # [g/cm^3]
    # Step 2:  get FFV of dry polymer
    FFV = ((1 / rho20) - V_occ) / (1 / rho20)
    return FFV


def get_S10_EQ(T: float, sol: str, pol: str, MW2: float = None):
    (
        eos_mix,
        _eos_sol,
        MW_1,
        _MW_2,
        _MW_monomer,
        rho_2_am_dry,
        _k_sw,
    ) = get_mixture_info(sol, pol, MW2)
    MW2 = MW2 if MW2 != None else _MW_2

    p0_list = arange(1, 5, 1)  # [Pa]
    p0_MPa = p0_list * 1e-6  # [MPa]
    S0eq_gg = [solve_solubility_EQ(T, _p0_, sol, pol, MW2) for _p0_ in p0_list]  # [g-sol/g-pol]
    omega1_0eq = [S0eq_gg_ / (1 + S0eq_gg_) for S0eq_gg_ in S0eq_gg]  # [g-sol/g-mix]
    S10eq = [None for i in range(len(p0_list))]
    for i in arange(len(p0_list)):
        # Solubility coeff at low P [MPa^-1]
        S10eq[i] = omega1_0eq[i] / p0_MPa[i]
    # print(S10eq)
    S10eq_ave = average(S10eq)  # average value
    isclose_checker = all([isclose(x, S10eq_ave, S10eq_ave * 0.05) for x in S10eq])  # 5% tolerance arounge average
    if isclose_checker == True:
        return S10eq_ave  # [MPa^-1]
    else:
        print("Error: S10_eq numerical result unsatisfactory")
        return None


def get_S10_NE(
    T: float,
    sol: str,
    pol: str,
    ksw: float,
    rho20: float = None,
    MW2: float = None,
):
    (
        eos_mix,
        _eos_sol,
        _MW_1,
        _MW_2,
        _MW_monomer,
        rho_2_am_dry,
        _k_sw,
    ) = get_mixture_info(sol, pol, MW2)
    MW2 = MW2 if MW2 != None else _MW_2
    rho20 = rho20 if rho20 != None else rho_2_am_dry

    p0_list = arange(1, 5, 1)  # [Pa]   #*change if needed
    p0_MPa = p0_list * 1e-6  # [MPa]
    omega_10 = [solve_solubility_NE(T, _p0_, sol, pol, MW2, ksw, rho20, return_extended=True)[1] for _p0_ in p0_list]

    S10 = [None for i in arange(len(p0_list))]
    for i in range(len(p0_list)):
        S10[i] = omega_10[i] / p0_MPa[i]  # [MPa^-1]

    # print(S10)
    S10_ave = average(S10)
    isclose_checker = all([isclose(x, S10_ave, S10_ave * 0.05) for x in S10])  # 5% tolerance arounge average
    if isclose_checker == True:
        return S10_ave  # [MPa^-1]
    else:
        print("Error: S10_NE numerical result unsatisfactory")
        return None


def get_S1_NE(
    T: float,
    p: float,
    sol: str,
    pol: str,
    ksw: float,
    rho20: float = None,
    MW2: float = None,
    omg1: float = None,
):
    (
        eos_mix,
        _eos_sol,
        _MW_1,
        _MW_2,
        _MW_monomer,
        rho_2_am_dry,
        _k_sw,
    ) = get_mixture_info(sol, pol, MW2)
    MW2 = MW2 if MW2 != None else _MW_2
    rho20 = rho20 if rho20 != None else rho_2_am_dry

    # Step 1: Get solubility value
    if omg1 == None:  # calculates
        (
            _S1,
            omega_1,
            _muad_1,
        ) = solve_solubility_NE(
            T, p, sol, pol, MW2, ksw, rho20, return_extended=True
        )[:3]
    else:  # use manual input
        omega_1 = omg1

    p_MPa = p * 1e-6  # [MPa]
    # Calculate S10 (infinitely dilute)
    S1 = omega_1 / p_MPa  # [MPa^-1]
    return S1


def get_L10(sol: str, pol: str, rho20: float):
    # Step 1: calculate FFV of dry polymer
    FFV = get_FFV(pol, rho20)
    # Step 2: caluclate L0 for sol
    A = {"CO2": 1.03e-3}  # [cm^2/s]
    B = {"CO2": 1.7}  # adim
    L10 = A[sol] * exp(-B[sol] / FFV)
    return L10


def get_L1(
    T: float,
    p: float,
    sol: str,
    pol: str,
    ksw: float,
    rho20: float = None,
    MW2: float = None,
    omg1: float = None,
):
    (
        eos_mix,
        _eos_sol,
        MW_1,
        _MW_2,
        _MW_monomer,
        rho_2_am_dry,
        _k_sw,
    ) = get_mixture_info(sol, pol, MW2)
    MW2 = MW2 if MW2 != None else _MW_2
    rho20 = rho20 if rho20 != None else rho_2_am_dry
    # Solution from NETGP model
    if omg1 == None:  # calculates
        (
            _S1,
            omega_1,
            _muad_1,
        ) = solve_solubility_NE(
            T, p, sol, pol, MW2, ksw, rho20, return_extended=True
        )[:3]
    else:  # use manual input
        omega_1 = omg1

    # Step 1: get FFV of dry polymer
    FFV = get_FFV(pol, rho20)
    # Step 2: calculate beta
    # omega_1 = omega1_0EQ  # use EQ value
    K = {"CO2": 0.858}  # * only at 35°C

    S10eq = get_S10_EQ(T, sol, pol)  # Solubility coeff at low P estimated with EQ [MPa^-1]

    # S10 = get_S10_NE(
    #     T, sol, pol, k_sw=k_sw, MW2=MW_2
    # )  # Solubility coeff at low P with NETGP [MPa^-1]

    # S0eq = (omega1_0EQ/MW_1*22.4e3*rho_2_am_dry) / p0_MPa  # Solubility coeff at low P [cm^3STP.cm^-3.MPa^-1]
    beta = K[sol] * (ksw / S10eq) / (FFV**2)  # * correlation at 35°C only

    # Step 3: calculate L1
    L10 = get_L10(sol, pol, rho20)
    L1 = L10 * exp(beta * omega_1)  # [cm^2/s]

    # print("1/FFV = %.3g" % (1 / FFV))
    # print("L10 = %.3g\tcm^2/s" % L10)
    # print("ksw = %g\tMPa^-1" % k_sw)
    # print("omega_1 = %.3g g-sol/g-mix" % omega_1)
    # print("S0eq = %.3g MPa^-1" % S10eq)
    # print("S10 = %.3g MPa^-1" % S10)
    # print("S0eq = %g cm^3(STP).cm^-3.MPa^-1" %S0eq)
    # print("")
    # print("ksw/S0eq/FFV^2 = %.3g" % ((k_sw / S10eq) / (FFV**2)))
    # print("beta = %.2g" % beta)
    return L1


def get_P10(
    T: float,
    sol: str,
    pol: str,
    ksw: float,
    rho20: float = None,
    MW2: float = None,
    unit=None,
):
    (
        eos_mix,
        _eos_sol,
        MW1,
        _MW_2,
        _MW_monomer,
        rho_2_am_dry,
        _k_sw,
    ) = get_mixture_info(sol, pol, MW2)
    MW2 = MW2 if MW2 != None else _MW_2
    rho20 = rho20 if rho20 != None else rho_2_am_dry

    L10 = get_L10(sol, pol, rho20)  # [cm^2/s]
    S10 = get_S10_NE(T, sol, pol, ksw=ksw, rho20=rho20, MW2=MW2)  # [MPa^-1]

    P10 = L10 * (rho20 / MW1) * S10  # [mol/cm.s.MPa]
    P10_SI = P10 * (1e2 * 1e-6)  # [mol/m.s.Pa]
    P10_barrer = P10_SI / (3.348e-16)  # [barrer]
    # print("L10 = %.3g\tcm^2/s" % L10)
    # print("S10 = %.3g MPa^-1" % S10)
    if unit == None:
        return P10
    elif unit == "SI":
        return P10_SI
    elif unit == "barrer":
        return P10_barrer


def get_Z1(T: float, p: float, sol: str):
    eos_sol, MW_1 = get_mixture_info(sol, pol=None, MW2=None)
    # Calculate p_sat [Pa]
    psat, vlsat, vvsat = eos_sol.psat(T)
    # Identify phase of external solute
    if p >= psat:  # L phase
        rho_1 = eos_sol.density(T, p, "L")  # [mol/m^3]
    else:  # V phase
        rho_1 = eos_sol.density(T, p, "V")  # [mol/m^3]
    v1 = 1 / rho_1  # molar volume [m^3/mol]
    # Calculate compresibility factor
    Z1 = p * v1 / (8.314 * T)  # Compressibility factor [adim]
    return Z1


def get_P1(
    T: float,
    p_u: float,
    p_d: float,
    sol: str,
    pol: str,
    ksw: float,
    rho20: float = None,
    MW2=None,
    unit=None,
):
    from scipy.integrate import trapz

    (
        eos_mix,
        _eos_sol,
        MW_1,
        _MW_2,
        _MW_monomer,
        rho_2_am_dry,
        _k_sw,
    ) = get_mixture_info(sol, pol, MW2)
    MW2 = MW2 if MW2 != None else _MW_2
    rho20 = rho20 if rho20 != None else rho_2_am_dry

    def f(_p_):  # integral term
        # print("\tp = %g MPa" % (_p_ * 1e-6))
        # Step 1: calculate solubility
        (
            _S1,
            omega_1,
            _muad_1,
        ) = solve_solubility_NE(
            T, _p_, sol, pol, MW2, ksw, rho20, return_extended=True
        )[:3]
        # Step 2: calculates Z1, S1, L1 and rho2
        Z1 = get_Z1(T, _p_, sol)  # compressibility factor [adim]
        S1_MPa = get_S1_NE(T, _p_, sol, pol, ksw, rho20, MW2, omega_1)  # solubility coeff [MPa^-1]
        S1 = S1_MPa * 1e-6  # [Pa^-1]
        L1 = get_L1(T, _p_, sol, pol, ksw, rho20, MW2, omega_1)  # [cm^2/s]
        rho2 = rho20 * (1 - ksw * (_p_ * 1e-6))  # [g/cm^3]
        # rho2 = rho_2_am_dry / (1 + k_sw * (_p_ * 1e-6))  # adjusted [g-mix/cm^3-mix]  # TEST
        return rho2 * L1 * S1 * Z1

    p = linspace(p_d, p_u, 10)  # *more bins is more accurate, 10 is opimum for speed and accuracy
    y = [f(_p) for _p in p]
    fx_res = trapz(y, p)
    # print("p = ", p)
    # print(y)
    # print("f(x) = %.3g" % fx_res)
    # print("")

    P1 = 1 / (MW_1 * (p_u - p_d)) * fx_res  # [mol/cm.s.Pa]
    P1_SI = P1 * (1e2)  # [mol/m.s.Pa]
    P1_barrer = P1_SI / (3.348e-16)  # [barrer]
    if unit == None:
        return P1
    elif unit == "SI":
        return P1_SI
    elif unit == "barrer":
        return P1_barrer


def plot_permeability_isotherm_NE(
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
    ) = get_mixture_info(sol, pol, MW2)
    rho20 = rho20 if rho20 != None else rho_2_am_dry
    MW2 = MW2 if MW2 != None else _MW_2

    # import exp data
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
            search_pattern = f"^P_{T-273}C (.*)"  # strat with P_{T-273}C
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
            print(dict[sheet])

        print(ref_no)
        ref_ID = []
        ref_df = pd.read_excel(reffile, "references")
        # print(ref_df)
        for i, no in enumerate(ref_no):
            ref_ID.append(ref_df.loc[ref_df["# ref"] == f"[{no}]", "refID"].item())
        print(ref_ID)
    except Exception as e:
        print("")
        print("Error - importing P exp data failed:")
        print(e)

    hasExpData = True if len(matched_sheets) > 0 else False

    p_calc = linspace(p_l, p_u, no_of_points)
    p_MPa_calc = p_calc * 1e-6  # [MPa]
    print("p_cal = ", p_calc)

    # Create empty placeholders, return None in case calculation fails
    permeability_NE_calc_list = [None for i in range(len(ksw_list))]
    p_MPa_exp_list = [None for i in range(len(matched_sheets))]
    permeability_exp_list = [None for i in range(len(matched_sheets))]
    permeability_calc_evaluation_NE_list = [None for i in range(len(ksw_list))]
    AAD_percent_NE = [None for i in range(len(ksw_list))]
    label_NE = [None for i in range(len(ksw_list))]

    # calculated NE permeability for each ksw
    for i, ksw_ in enumerate(ksw_list):
        permeability_NE_calc_list[i] = [
            get_P1(
                T,
                pu_,
                1,
                sol,
                pol,
                ksw=ksw_list[i],
                rho20=rho20,
                MW2=MW2,
                unit="barrer",
            )
            for pu_ in p_calc
        ]
        print("Permeability at ksw=%g =\t" % ksw_list[i], permeability_NE_calc_list[i])

    # Original label
    label_NE = [r"NE $k_{sw} = %.3g \, MPa^{-1}$" % (ksw_) for ksw_ in ksw_list]

    # Importing exp data
    if hasExpData == True:
        for i, sheet in enumerate(matched_sheets):
            p_MPa_exp_list[i] = asarray(dict[sheet]["P [MPa]"])
            permeability_exp_list[i] = asarray(dict[sheet]["Permeability [barrer]"])

        # calculate AAD for NE when there is only 1 exp data sheet
        if len(matched_sheets) == 1:
            p_exp_list = p_MPa_exp_list[0] * 1e6  # [Pa]
            perm_exp_evaluation_list = permeability_exp_list[0]
            for i, ksw_ in enumerate(ksw_list):
                permeability_calc_evaluation_NE_list[i] = [
                    get_P1(
                        T,
                        _p_,
                        1,
                        sol,
                        pol,
                        ksw=ksw_list[i],
                        rho20=rho20,
                        MW2=MW2,
                        unit="barrer",
                    )
                    for _p_ in p_exp_list
                ]
                print(permeability_calc_evaluation_NE_list[i])
                try:
                    AAD_percent_NE[i] = (
                        get_fitting_AAD(
                            perm_exp_evaluation_list,
                            permeability_calc_evaluation_NE_list[i],
                        )
                        * 100
                    )  # [%]
                except:
                    AAD_percent_NE[i] = 0
                print(AAD_percent_NE[i])
                print("AAD%% for NE ksw=%g: AAD%% = %.1f%%" % (ksw_, AAD_percent_NE[i]))
                label_NE[i] += " (AAD%%=%.1f%%)" % AAD_percent_NE[i]
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # exp data
    if hasExpData == True:
        for i, sheet in enumerate(matched_sheets):
            ax.plot(
                p_MPa_exp_list[i],
                permeability_exp_list[i],
                color=exp_style["color"],
                marker=custom_markers[i],
                linestyle=exp_style["linestyle"],
                markerfacecolor=exp_style["markerfacecolor"],
                label=f"exp: {ref_ID[i]} ({ref_no[i]})",
            )
    # NE model
    for i, ksw_ in enumerate(ksw_list):
        ax.plot(
            p_MPa_calc,
            permeability_NE_calc_list[i],
            color=custom_colours[2],
            marker=calc_style["marker"],
            linestyle=custom_linestyles[i],
            label=label_NE[i],
        )

    # labelling
    ax.set_xlabel(r"p (MPa)")
    ax.set_ylabel(r"Permeability (barrer)")
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
    # ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    # ax.grid(visible=True)
    ax.tick_params(direction="in")
    legend_ncol = 1 if (len(matched_sheets) + len(ksw_list)) < 5 else 2
    ax.legend(ncol=legend_ncol).set_visible(True)
    if display_plot == True:
        plt.show()
    if save_plot_dir != None:
        plt.savefig(save_plot_dir, dpi=1200, transparent=True)
        print(f"Plot saved: {save_plot_dir}")
        print("")


def import_MultiTait_parameters(pol: str):
    try:
        databasepath = os.path.join(os.path.dirname(__file__), "litdata")
        databasepath += "/pol-sol_parameters_150223.xlsx"
        file = pd.ExcelFile(databasepath, engine="openpyxl")
        df = pd.read_excel(file, sheet_name="multiTait")
        df.set_index("Polymer", inplace=True)

    except Exception as e:
        print("Error: ")
        print(e)
    else:
        try:
            df_filter = df.loc[pol]
        except Exception as e:
            print("Error: ")
            print(e)
        else:
            chi = df_filter["chi"].item()  # [adim]
            Tg_C = df_filter["Tg (°C)"].item()  # [°C]
            B0_g = df_filter["B0_g (MPa)"].item()  # [MPa]
            B0_r = df_filter["B0_r (MPa)"].item()  # [MPa]
            B1_g = df_filter["B1_g (°C^-1)"].item()  # [°C^-1]
            B1_r = df_filter["B1_r (°C^-1)"].item()  # [°C^-1]
            return Tg_C, B0_g, B0_r, B1_g, B1_r, chi


def fit_polPVT_multiTait(xlxs_sheet: str, display_plot: bool = True, save_plot_dir: str = None):
    try:
        databasepath = os.path.join(os.path.dirname(__file__), "litdata")
        databasepath += "/pol_PVT.xlsx"
        file = pd.ExcelFile(databasepath, engine="openpyxl")
        df = pd.read_excel(file, sheet_name=xlxs_sheet)

    except Exception as e:
        print("Error: ")
        print(e)
        return None
    else:

        def V_func(X, a0, a1, a2, B0, B1):
            T_C, p_MPa = X
            C = 0.0894
            B = B0 * exp(-B1 * T_C)
            V0 = a0 + a1 * T_C + a2 * T_C**2
            V = V0 * (1 - C * log(1 + p_MPa / B))
            return V  # [cm^3/g]

        T_C_list = df["T (°C)"].values.tolist()  # [°C]
        p_MPa_list = df["P (MPa)"].values.tolist()  # [MPa]
        V_exp = df["V_pol (cm3/g)"].values.tolist()  # [cm3/g]
        result, cov = curve_fit(V_func, (T_C_list, p_MPa_list), V_exp, p0=(0.9, 2e-4, 1e-7, 100, 3e-3))
        a0, a1, a2, B0, B1 = result

        # unique pressure values
        pMPa_unq_list = list(set(p_MPa_list))  # [MPa]
        pMPa_unq_list.sort()
        T_C_unq_list = list(set(T_C_list))  # [°C]
        T_C_unq_list.sort()

        T_C_upper = T_C_unq_list[-1]  # [°C]
        T_C_lower = T_C_unq_list[0]  # [°C]
        
        print(f"T range: {T_C_lower}°C - {T_C_upper}°C")
        print("a0 = %.3g cm^3/g" % a0)
        print("a1 = %.3g cm^3/g°C" % a1)
        print("a2 = %.3g cm^3/g°C^2" % a2)
        print("B0 = %.3g MPa" % B0)
        print("B1 = %.3g °C^-1" % B1)
        print("")
        
        # Temperature values for each pressure
        T_C = [None for i in range(len(pMPa_unq_list))]
        for i, p in enumerate(pMPa_unq_list):
            T_C[i] = df[(df["P (MPa)"] == p)]["T (°C)"].values.tolist()  # [MPa]
        
        # V values for each pressure
        V_exp = [None for i in range(len(pMPa_unq_list))]
        V_multiTait = [None for i in range(len(pMPa_unq_list))]
        for i, p in enumerate(pMPa_unq_list):
            V_exp[i] = df[(df["P (MPa)"] == p)]["V_pol (cm3/g)"].values.tolist()
            V_multiTait[i] = [V_func((_T_C, p), a0, a1, a2, B0, B1) for _T_C in T_C[i]]
        # Calculate upper limits of x and y axis
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')
        for i, p in enumerate(pMPa_unq_list):
            x_min = min(x_min, min(T_C[i]))
            x_max = max(x_max, max(T_C[i]))            
            y_min = min(y_min, min(V_exp[i] + V_multiTait[i]))
            y_max = max(y_max, max(V_exp[i] + V_multiTait[i]))
        
        title_dict = {'PS_rubbery': 'PS (rubbery)', 'PS_glassy': 'PS (glassy)', 
                      'PMMA_rubbery': 'PMMA (rubbery)', 'PMMA_glassy': 'PMMA (glassy)'}
        # Plotting
        fig = plt.figure(figsize=(4.8, 3.5))
        ax = fig.add_subplot(111)
        colours = list(Color("silver").range_to(Color("maroon"), len(pMPa_unq_list)))  # colour gradient

        for i, p in enumerate(pMPa_unq_list):
            # Multi-Tait model
            # _TC = df[(df["P (MPa)"] == p)]["T (°C)"].values.tolist()  # [MPa]
            # _V_ = [V_func((_T_, p), a0, a1, a2, B0, B1) for _T_ in _TC]
            ax.plot(
                T_C[i],
                V_multiTait[i],
                color="%s" % colours[i],
                marker="None",
                linestyle="solid",
                label="{:.0f} MPa".format(p),
            )
            # exp data
            ax.scatter(
                # df[(df["P (MPa)"] == p)]["T (°C)"],
                T_C[i],
                # df[(df["P (MPa)"] == p)]["V_pol (cm3/g)"],
                V_multiTait[i],
                color="%s" % colours[i],
                marker="x",
                linestyle="None",
            )

        ax.set_xlabel("T (°C)")
        ax.set_ylabel(r"$\hat{V}_{pol}$ ($cm^{3}/g$)")
        ax.set_title(title_dict[xlxs_sheet])
        
        # Get the length of major ticks on the x-axis
        x_major_tick_length = ax.get_xticks()[1] - ax.get_xticks()[0]
        
        # Get the length of major ticks on the y-axis
        y_major_tick_length = ax.get_yticks()[1] - ax.get_yticks()[0]
        
        # Set adjust x and y tick to cover all data
        ax.set_xlim(left=x_min - x_major_tick_length, right=x_max + x_major_tick_length)    # Default
        ax.set_ylim(bottom=y_min - y_major_tick_length, top=y_max + y_major_tick_length)    # Default
        # ax.set_xlim(left=x_min - x_major_tick_length, right=x_max + 2*x_major_tick_length)    # For paper
        # ax.set_ylim(bottom=y_min - y_major_tick_length, top=y_max + y_major_tick_length)    # For paper
        
        ax.tick_params(direction="in")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left").set_visible(True)
        if save_plot_dir != None:
            plt.savefig(save_plot_dir, dpi=1200)
            print(f"Plot saved: {save_plot_dir}")
            print("")
        if display_plot == True:
            plt.show()

        return a0, a1, a2, B0, B1, T_C_lower, T_C_upper, df


def get_V20_multiTait(T: float, p: float, pol: str):
    try:
        (
            a0_g,
            a1_g,
            a2_g,
            B0_g,
            B1_g,
            T_C_lower_g,
            T_C_upper_g,
            df_g,
        ) = fit_polPVT_multiTait(f"{pol}_glassy", display_plot=False)
        (
            a0_r,
            a1_r,
            a2_r,
            B0_r,
            B1_r,
            T_C_lower_r,
            T_C_upper_r,
            df_r,
        ) = fit_polPVT_multiTait(f"{pol}_rubbery", display_plot=False)
    except Exception as e:
        print("Error: ")
        print(e)
        return None
    else:
        # find closest T value at given p in df
        pMPa_unq_list = list(set(df_g["P (MPa)"].values.tolist() + df_r["P (MPa)"].values.tolist()))  # [MPa]
        pMPa_unq_list.sort()
        pMPa_closest = find_closest(pMPa_unq_list, p * 1e-6)
        print(f"closest pressure = {pMPa_closest} MPa")

        def V_func(X, a0, a1, a2, B0, B1):
            T_C, p_MPa = X
            C = 0.0894
            B = B0 * exp(-B1 * T_C)
            V0 = a0 + a1 * T_C + a2 * T_C**2
            V = V0 * (1 - C * log(1 + p_MPa / B))
            return V

        V20_g = V_func((T - 273, p * 1e-6), a0_g, a1_g, a2_g, B0_g, B1_g)  # [cm^3/g]
        V20_r = V_func((T - 273, p * 1e-6), a0_r, a1_r, a2_r, B0_r, B1_r)  # [cm^3/g]

        if T <= (df_g[(df_g["P (MPa)"] == pMPa_closest)]["T (°C)"].max() + 273):
            print("Glassy state")
            V20 = V20_g
        elif T >= (df_r[(df_r["P (MPa)"] == pMPa_closest)]["T (°C)"].min() + 273):
            print("Rubbery state")
            V20 = V20_r
        else:
            print("average of Rubbery and Glassy")
            V20 = (V20_g + V20_r) / 2

        return V20


def get_chi(pol: str):
    Tg_C, B0_g, B0_r, B1_g, B1_r, chi = import_MultiTait_parameters(pol)
    if math.isnan(chi):  # check if chi is provided
        try:
            C = 0.0894
            kappa_g = C / (B0_g * exp(-B1_g * Tg_C))  # [MPa^-1]
            kappa_r = C / (B0_r * exp(-B1_r * Tg_C))  # [MPa^-1]

            chi = kappa_g / kappa_r  # [adim]
        except Exception as e:
            print("Error: chi cannot be calculated:")
            print(e)
    else:
        chi = chi
    return chi


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

    def get_V2eq(_p_):  # with penetrant presence
        S_gg, x1, rhol = solve_solubility_EQ(T, _p_, sol, pol, MW2, return_extended=True)[:3]
        rho_eq = rhol * (x1 * MW_1 + (1 - x1) * MW2) * 1e-6  # [g_mix/cm^3_mix]
        rho2_eq = rho_eq * (1 - S_gg / (S_gg + 1))  # [g_pol/cm^3_mix]
        V2_eq = 1 / rho2_eq  # [cm^3_mix/g_mix]
        return V2_eq

    def fx(_p0_):
        p_l = _p0_ * (1 - frc_step)  # [Pa]
        p_u = _p0_ * (1 + frc_step)  # [Pa]
        V2eq_l = get_V2eq(p_l)  # [cm^3/g]
        V2eq_u = get_V2eq(p_u)  # [cm^3/g]
        return (V2eq_u - V2eq_l) / (p_u - p_l)

    p0_list = arange(1, 5, 1)  # [Pa]
    dV2eq_dp = [fx(_p0_) for _p0_ in p0_list]
    dV2eq_dp_ave = average(dV2eq_dp)
    isclose_checker = all(
        [isclose(x, dV2eq_dp_ave, dV2eq_dp_ave * 0.05) for x in dV2eq_dp]
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
    chi = get_chi(pol)  # [adim]

    # Step 3: dV2eq_df0 numercial result
    dV2eq_df0 = get_dV2eq_df0(T, sol, pol, MW2=MW2)
    # Step 4: ksw
    ksw_Pa = chi / V_pol0 * dV2eq_df0  # [Pa^-1]
    ksw_MPa = ksw_Pa * 1e6  # [MPa^-1]
    if unit == "MPa^-1":
        return ksw_MPa
    if unit == "Pa^-1":
        return ksw_Pa


def plot_ksw_parity(
    xlxs_sheet: str,
    sol_list: list[str] = None,
    display_plot: bool = True,
    save_plot_dir: str = None,
) -> None:
    try:
        databasepath = os.path.join(os.path.dirname(__file__), "litdata")
        databasepath += "/ksw_Tg.xlsx"
        file = pd.ExcelFile(databasepath, engine="openpyxl")
        df = pd.read_excel(file, xlxs_sheet)

        ksw_predicted = df["predicted ksw (MPa^-1)"].values.tolist()
        ksw_ref = df["fitted ksw (MPa^-1)"].values.tolist()
    except Exception as e:
        print("")
        print("Error - importing ksw data failed:")
        print(e)
    else:
        # Getting unique list
        df_sol_list = sorted(list(set(df["sol"].values.tolist())))
        df_pol_list = sorted(list(set(df["pol"].values.tolist())))
        df_sol_list.reverse()  # reverse to get CO2 first
        sol_colour = ["red", "green", "blue", "purple", "orange", "brown"]
        pol_marker = ["o", "^", "s"]
        # use all solutes available if sol_list == None
        if sol_list == None:
            sol_list = df_sol_list
        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111)
        z = 0
        # y=x line
        ksw_all = ksw_predicted + ksw_ref
        x = linspace(min(ksw_all), max(ksw_all), 10)
        z += 1
        ax.plot(x, x, linestyle="solid", color="black", zorder=z)
        # ksw data
        for id_i, sol_ in enumerate(sol_list):
            for id_j, pol_ in enumerate(df_pol_list):
                try:
                    a = df[(df["sol"] == sol_) & (df["pol"] == pol_) & (df["Use in plot?"] == "Yes")]

                except:
                    continue
                else:
                    z += 1
                    mappable = ax.scatter(
                        a["fitted ksw (MPa^-1)"].values.tolist(),
                        a["predicted ksw (MPa^-1)"].values.tolist(),
                        c=a["T (C)"].values.tolist(),
                        cmap="coolwarm",
                        marker=pol_marker[id_j],
                        label=f"{sol_}-{pol_}",
                        zorder=z,
                    )

        # labelling
        ax.set_xlabel(r"fitted $k_{sw}$ ($MPa^{-1}$)")
        ax.set_ylabel(r"predicted $k_{sw}$ ($MPa^{-1}$)")
        ax.set_title(r"$k_{sw}$ parity")

        ax.locator_params(axis="both", nbins=5)
        ax.tick_params(direction="in")

        cbar = fig.colorbar(mappable, aspect=30)  # higher aspect for smaller colour bar width
        cbar.set_label("T (°C)", fontsize="xx-small")
        cbar.ax.tick_params(labelsize="xx-small")
        ax.legend().set_visible(True)
        fig.tight_layout()
        if display_plot == True:
            plt.show()
        if save_plot_dir != None:
            plt.savefig(save_plot_dir, dpi=1200, transparent=True)
            print(f"Plot saved: {save_plot_dir}")
            print("")


def plot_ksw_AAD(
    xlxs_sheet: str,
    sol_list: list[str] = None,
    display_plot: bool = True,
    save_plot_dir: str = None,
) -> None:
    try:
        databasepath = os.path.join(os.path.dirname(__file__), "litdata")
        databasepath += "/ksw_Tg.xlsx"
        file = pd.ExcelFile(databasepath, engine="openpyxl")
        df = pd.read_excel(file, xlxs_sheet)

    except Exception as e:
        print("")
        print("Error - importing ksw data failed:")
        print(e)
    else:
        # Getting unique list
        df_sol_list = sorted(list(set(df["sol"].values.tolist())))
        df_pol_list = sorted(list(set(df["pol"].values.tolist())))
        df_sol_list.reverse()  # reverse to get CO2 first
        sol_colour = ["red", "green", "blue", "purple", "orange", "brown"]
        pol_marker = ["o", "^", "s"]
        # use all solutes available if sol_list == None
        if sol_list == None:
            sol_list = df_sol_list

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # AAD% data
        for id_i, sol_ in enumerate(sol_list):
            for id_j, pol_ in enumerate(df_pol_list):
                try:
                    a = df[(df["sol"] == sol_) & (df["pol"] == pol_) & (df["Use in plot?"] == "Yes")]
                except:
                    continue
                else:
                    ax.scatter(
                        a["T (C)"].values.tolist(),
                        a["AAD%_ksw_parity (%)"].values.tolist(),
                        marker=pol_marker[id_j],
                        label=f"{sol_}-{pol_}",
                        color=custom_colours[id_j],
                    )
        # labelling
        ax.set_xlabel(r"T (°C)")
        ax.set_ylabel(r"AAD% (%)")
        ax.set_title(r"AAD% for $k_{sw}$ prediction")

        ax.set_yscale("log")
        ax.tick_params(direction="in")
        ax.legend().set_visible(True)
        if display_plot == True:
            plt.show()
        if save_plot_dir != None:
            plt.savefig(save_plot_dir, dpi=1200, transparent=True)
            print(f"Plot saved: {save_plot_dir}")
            print("")


def plot_rho20_parity(
    xlxs_sheet: str,
    sol_list: list[str] = None,
    display_plot: bool = True,
    save_plot_dir: str = None,
) -> None:
    try:
        databasepath = os.path.join(os.path.dirname(__file__), "litdata")
        databasepath += "/ksw_Tg.xlsx"
        file = pd.ExcelFile(databasepath, engine="openpyxl")
        df = pd.read_excel(file, xlxs_sheet)

        rho20_PVT = df["rho20 PVT (g/cm^3)"].values.tolist()
        rho20_fit = df["fitted rho20 (g/cm^3)"].values.tolist()
    except Exception as e:
        print("")
        print("Error - importing ksw data failed:")
        print(e)
    else:
        # Getting unique list
        df_sol_list = sorted(list(set(df["sol"].values.tolist())))
        df_pol_list = sorted(list(set(df["pol"].values.tolist())))
        df_sol_list.reverse()  # reverse to get CO2 first
        sol_colour = ["red", "green", "blue", "purple", "orange", "brown"]
        pol_marker = ["o", "^", "s"]
        # use all solutes available if sol_list == None
        if sol_list == None:
            sol_list = df_sol_list
        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111)
        z = 0
        # y=x line
        ksw_all = rho20_PVT + rho20_fit
        x = linspace(min(ksw_all), max(ksw_all), 10)
        z += 1
        ax.plot(x, x, linestyle="solid", color="black", zorder=z)
        # ksw data
        for id_i, sol_ in enumerate(sol_list):
            for id_j, pol_ in enumerate(df_pol_list):
                try:
                    a = df[(df["sol"] == sol_) & (df["pol"] == pol_) & (df["Use in plot?"] == "Yes")]

                except:
                    continue
                else:
                    z += 1
                    mappable = ax.scatter(
                        a["fitted rho20 (g/cm^3)"].values.tolist(),
                        a["rho20 PVT (g/cm^3)"].values.tolist(),
                        c=a["T (C)"].values.tolist(),
                        cmap="coolwarm",
                        marker=pol_marker[id_j],
                        label=f"{sol_}-{pol_}",
                        zorder=z,
                    )

        # labelling
        ax.set_xlabel(r"fitted $\rho_{20}$ ($g/cm^{3}$)")
        ax.set_ylabel(r"predicted $\rho_{20}$ ($g/cm^{3}$)")
        ax.set_title(r"$\rho_{20}$ parity")

        ax.locator_params(axis="both", nbins=5)
        ax.tick_params(direction="in")

        cbar = fig.colorbar(mappable, aspect=30)  # higher aspect for smaller colour bar width
        cbar.set_label("T (°C)", fontsize="xx-small")
        cbar.ax.tick_params(labelsize="xx-small")
        ax.legend().set_visible(True)
        fig.tight_layout()
        if display_plot == True:
            plt.show()
        if save_plot_dir != None:
            plt.savefig(save_plot_dir, dpi=1200, transparent=True)
            print(f"Plot saved: {save_plot_dir}")
            print("")


def plot_pg_T(
    xlxs_sheet: str,
    sol_list: list[str] = None,
    display_plot: bool = True,
    save_plot_dir: str = None,
) -> None:
    try:
        databasepath = os.path.join(os.path.dirname(__file__), "litdata")
        databasepath += "/ksw_Tg.xlsx"
        file = pd.ExcelFile(databasepath, engine="openpyxl")
        df = pd.read_excel(file, xlxs_sheet)

    except Exception as e:
        print("")
        print("Error - importing Tg data failed:")
        print(e)
    else:
        # Getting unique list
        df_sol_list = sorted(list(set(df["sol"].values.tolist())))
        df_pol_list = sorted(list(set(df["pol"].values.tolist())))
        df_sol_list.reverse()  # reverse to get CO2 first
        sol_colour = ["red", "green", "blue", "purple", "orange", "brown"]
        pol_marker = ["o", "^", "s"]

        # use all solutes available if sol_list == None
        if sol_list == None:
            sol_list = df_sol_list

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # pg data
        for id_i, sol_ in enumerate(sol_list):
            for id_j, pol_ in enumerate(df_pol_list):
                try:
                    a = df[(df["sol"] == sol_) & (df["pol"] == pol_) & (df["Use in plot?"] == "Yes")]
                except:
                    continue
                else:
                    ax.scatter(
                        a["T (C)"],
                        a["pg (Pa)"] * 1e-6,
                        marker=pol_marker[id_j],
                        label=f"{sol_}-{pol_}",
                        color=custom_colours[id_j],
                    )
        # labelling
        ax.set_xlabel(r"T (°C)")
        ax.set_ylabel(r"$p_{g}$ (MPa)")
        ax.set_title(r"$p_{g}$ prediction from EQ-NE intersect")

        # ax.set_yscale('log')
        ax.set_ylim(bottom=0)
        ax.tick_params(direction="in")
        ax.legend().set_visible(True)
        if display_plot == True:
            plt.show()
        if save_plot_dir != None:
            plt.savefig(save_plot_dir, dpi=1200, transparent=True)
            print(f"Plot saved: {save_plot_dir}")
            print("")


def plot_pg_Sg(
    xlxs_sheet: str,
    sol_list: list[str] = None,
    display_plot: bool = True,
    save_plot_dir: str = None,
) -> None:
    try:
        databasepath = os.path.join(os.path.dirname(__file__), "litdata")
        databasepath += "/ksw_Tg.xlsx"
        file = pd.ExcelFile(databasepath, engine="openpyxl")
        df = pd.read_excel(file, xlxs_sheet)

    except Exception as e:
        print("")
        print("Error - importing Tg data failed:")
        print(e)
    else:
        # Getting unique list
        df_sol_list = sorted(list(set(df["sol"].values.tolist())))
        df_pol_list = sorted(list(set(df["pol"].values.tolist())))
        df_sol_list.reverse()  # reverse to get CO2 first
        sol_colour = ["red", "green", "blue", "purple", "orange", "brown"]
        pol_marker = ["o", "^", "s"]

        # use all solutes available if sol_list == None
        if sol_list == None:
            sol_list = df_sol_list

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # pg - Sg data
        for id_i, sol_ in enumerate(sol_list):
            for id_j, pol_ in enumerate(df_pol_list):
                try:
                    a = df[(df["sol"] == sol_) & (df["pol"] == pol_) & (df["Use in plot?"] == "Yes")]
                except:
                    continue
                else:
                    mappable = ax.scatter(
                        a["Sg (g/g)"],
                        a["pg (Pa)"] * 1e-6,
                        c=a["T (C)"],
                        cmap="coolwarm",
                        marker=pol_marker[id_j],
                        label=f"{sol_}-{pol_}",
                    )
        # labelling
        ax.set_xlabel(r"$S_{g}$ (g/g)")
        ax.set_ylabel(r"$p_{g}$ (MPa)")
        ax.set_title(r"Pressure and solubility at EQ-NE intersect")

        # ax.set_yscale('log')
        ax.tick_params(direction="in")
        ax.legend().set_visible(True)
        # Colour bar
        cbar = fig.colorbar(mappable, aspect=30)  # higher aspect for smaller colour bar width
        cbar.set_label("T (°C)", fontsize="xx-small")
        cbar.ax.tick_params(labelsize="xx-small")
        if display_plot == True:
            plt.show()
        if save_plot_dir != None:
            plt.savefig(save_plot_dir, dpi=1200, transparent=True)
            print(f"Plot saved: {save_plot_dir}")
            print("")


def plot_Sg_T(
    xlxs_sheet: str,
    sol_list: list[str] = None,
    display_plot: bool = True,
    save_plot_dir: str = None,
) -> None:
    try:
        databasepath = os.path.join(os.path.dirname(__file__), "litdata")
        databasepath += "/ksw_Tg.xlsx"
        file = pd.ExcelFile(databasepath, engine="openpyxl")
        df = pd.read_excel(file, xlxs_sheet)

    except Exception as e:
        print("")
        print("Error - importing Tg data failed:")
        print(e)
    else:
        # Getting unique list
        df_sol_list = sorted(list(set(df["sol"].values.tolist())))
        df_pol_list = sorted(list(set(df["pol"].values.tolist())))
        df_sol_list.reverse()  # reverse to get CO2 first
        sol_colour = ["red", "green", "blue", "purple", "orange", "brown"]
        pol_marker = ["o", "^", "s"]

        # use all solutes available if sol_list == None
        if sol_list == None:
            sol_list = df_sol_list

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Sg-T data
        for id_i, sol_ in enumerate(sol_list):
            for id_j, pol_ in enumerate(df_pol_list):
                try:
                    a = df[(df["sol"] == sol_) & (df["pol"] == pol_) & (df["Use in plot?"] == "Yes")]
                except:
                    continue
                else:
                    ax.scatter(
                        a["T (C)"],
                        a["Sg (g/g)"],
                        marker=pol_marker[id_j],
                        label=f"{sol_}-{pol_}",
                        color=custom_colours[id_j],
                    )
        # labelling
        ax.set_xlabel(r"T (°C)")
        ax.set_ylabel(r"$S_{g}$ (g/g)")
        ax.set_title(r"$S_{g}$ prediction from EQ-NE intersect")

        # ax.set_yscale('log')
        ax.set_ylim(bottom=0)
        ax.tick_params(direction="in")
        ax.legend().set_visible(True)
        if display_plot == True:
            plt.show()
        if save_plot_dir != None:
            plt.savefig(save_plot_dir, dpi=1200, transparent=True)
            print(f"Plot saved: {save_plot_dir}")
            print("")


def plot_pol_Ldensity_isobar_EQ(
    pol: str,
    MW2: float = None,
    display_plot: bool = True,
    save_plot_dir: str = None,
):
    eos_pol, _MW_2, _MW_monomer, _rho_pol_am = get_mixture_info(sol=None, pol=pol)
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
        df = pd.read_excel(datafile, f"{pol}_rubbery")
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
            rho_EQ_calc[i] = [get_pol_prop_EQ(T=_T_, p=p, pol=pol, MW2=MW2)[0] for _T_ in T_exp[i]]  # [g/cm^3]

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
                # label= "{:.0f} MPa exp".format(p*1e-6),
            )

        ax.set_xlabel("T (°C)")
        ax.set_ylabel(r"$\rho_{pol}$ ($g/cm^{3}$)")
        ax.set_title(f"{pol} liquid density")

        # ax.set_yscale('log')
        ax.tick_params(direction="in")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left").set_visible(True)
        if display_plot == True:
            plt.show()
        if save_plot_dir != None:
            plt.savefig(save_plot_dir, dpi=1200, transparent=True)
            print(f"Plot saved: {save_plot_dir}")
            print("")


def plot_pol_Ldenisty_isobar_EQ_MW2sensitivity(MW2_list: list[float], T: float, p_list: list[float], pol: str):
    rho_EQ_calc = [None for i in p_list]

    for i, p in enumerate(p_list):
        rho_EQ_calc[i] = [get_pol_prop_EQ(T=T, p=p, pol=pol, MW2=_MW2_)[0] for _MW2_ in MW2_list]  # [g/cm^3]

    # Plotting

    fig = plt.figure(figsize=(4.8, 3.5))
    ax = fig.add_subplot(111)
    colours = list(Color("silver").range_to(Color("maroon"), len(p_list)))  # colour gradient
    # EQ
    for i, p in enumerate(p_list):
        ax.plot(
            MW2_list,
            rho_EQ_calc[i],
            color="%s" % colours[i],
            marker="None",
            linestyle="solid",
            label="{:.0f} MPa".format(p * 1e-6),
        )
    # labelling
    ax.set_xlabel(r"$MW_{2}$ (g/mol)")
    ax.set_ylabel(r"$\rho_{pol}$ ($g/cm^{3}$)")
    ax.set_title(f"{pol} liquid denisty at {T-273}°C")
    # styling
    # ax.set_ylim(bottom=0)
    # ax.grid(visible=True)
    ax.tick_params(direction="in")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left").set_visible(True)
    plt.show()


if __name__ == "__main__":
    # T = 140 + 273  # [K]
    # p = 1.0e6  # [Pa]
    # sol = "CO2"
    # pol = "HDPE"
    # plot_p_sensitivity_NE(p_l=0.5e6, p_u=10e6, no_of_points=10, T=T, pol=pol)
    # plot_ksw_sensitivity_NE(ksw_l=0, ksw_u=0.2, no_of_points=10,
    #                      T=T, p=p, pol=pol, S_1_ref=0.0217)
    # plot_MW2_sensitivity_NE(MW2_l=50e3, MW2_u=400e3,
    #                      no_of_points=10, T=T, p=p, pol=pol,
    #                      S_1_ref=0.0217)

    # *Test plot_solubility_MW2sensitivity
    # plot_MW2_sensitivity_EQ(
    #     MW2_l=1e3,
    #     MW2_u=1000e3,
    #     no_of_points=100,
    #     T=100 + 273,
    #     p=20e6,
    #     sol="CO2",
    #     pol="PMMA",
    # )
    # plot_MW2_sensitivity_EQvNE(
    #     MW2_l=10e3,
    #     MW2_u=300e3,
    #     no_of_points=20,
    #     T=30 + 273,
    #     p=20e6,
    #     sol="CO2",
    #     pol="PMMA",
    #     ksw=0,
    #     rho20=1.15143,
    # )
    # * Test plot_ksw_sensitivityEQvNE
    # plot_ksw_sensitivity_EQvNE(
    #     ksw_l=0.001, ksw_u=0.1, no_of_points=20, T=30+273, p=[1e4,0.1e6,1e6,2e6], pol=pol
    # )

    # * Test fit_ksw
    # start_time = time.time()
    # ksw,AAD_pc=fit_ksw_NE(T=30+273,
    #                       sol="CO2",
    #                       pol="PMMA",
    #                       xlxs_sheet_refno="29",
    #                       rho20=1.1514327111737948,
    #                       glassy_filter=True)
    # ksw, AAD_pc=fit_ksw_NE(T=40+273, sol="CO2", pol="PS", xlxs_sheet_refno="13",rho20=1.0245182108769082, ksw_x0_list=linspace(0.005,0.01,50))
    # print(f"ksw = {ksw}")
    # print("AAD%% = %.1f%%"%(AAD_pc))
    # print("\n--- Run time:\t%.0f seconds ---\n" % (time.time() - start_time))

    # * Test fit_rho20
    # fit_rho20_NE(T=65+273, sol="CO2", pol="PS", xlxs_sheet_refno="26",rho20_x0_list=linspace(1.0,1.2,10))

    # * Test plot_isotherm_EQvNE
    # plot_isotherm_EQvNE(
    #     p_l=1,
    #     p_u=20e6,
    #     no_of_points=5,
    #     T=30 + 273,
    #     sol="CO2",
    #     pol="PMMA",
    #     ksw_list=[0.0051],
    #     xlxs_sheet_refno_list=["29"],
    #     display_plot=True,
    # )
    # plot_isobar_EQvsNE(
    #     T_l=25 + 273,
    #     T_u=150 + 273,
    #     no_of_points=20,
    #     p=p,
    #     pol=pol,
    #     k_sw=0.00267143,
    #     MW_2=54730.9,
    # )
    # plot_isotherm_EQ(
    #     p_l=0.01e6,
    #     p_u=10e6,
    #     no_of_points=10,
    #     T_list=[
    #         28 + 273,
    #         35 + 273,
    #         # 100 + 273,
    #         # 125 + 273,
    #         # 132 + 273,
    #         # 150 + 273,
    #         # 175 + 273,
    #         # 200 + 273,
    #     ],
    #     sol="CO2",
    #     pol="HDPE",
    #     # xlxs_sheet_refno_list=["4",]
    # )
    # plot_isotherm_EQ_molFraction(
    #     p_l=0.01e6,
    #     p_u=60e6,
    #     no_of_points=60,
    #     T_list=[
    #         28 + 273,
    #         35 + 273,
    #     ],
    #     sol="CO2",
    #     pol="HDPE",
    # )
    # plot_isotherm_EQ_rhoL(
    #     p_l=0.01e6,
    #     p_u=60e6,
    #     no_of_points=50,
    #     T_list=[
    #         28 + 273,
    #         35 + 273,
    #     ],
    #     sol="CO2",
    #     pol="HDPE",
    # )
    # *NELF
    # plot_isotherm_NELF(
    #     p_l=0.1e6,
    #     p_u=40e6,
    #     no_of_points=50, T=115+273, sol="CO2", pol="PEEK",kij=0, ksw=0)
    # plot_isotherm_NELFvNESAFT(p_l=1000,p_u=4.1e6,no_of_points=20,
    #                           T=35+273, sol="CO2", pol="PS",
    #                           rho20_NELF=1.0223888, kij_NELF=0.032,ksw_NELF= 0.00266024,
    #                         #   rho20_NESAFT=1.01980,ksw_NESAFT=0.0037549,    #*best fit
    #                           rho20_NESAFT=1.04182475364932,ksw_NESAFT=0.00914269192783131,    #*fully predictive
    #                           xlxs_sheet_refno_list=["16"],
    #                           display_plot=False,
    #                           save_plot_dir="C:\\Users\\sn621\\Downloads\\NELFvNESAFT-predictive.png"
    #                           )

    # test get_dmuad_dlnomega1_NE
    # step size
    # step_array = logspace(-5, -1, 10)
    # dmuad_dlnomega = [
    #     get_therm_factor(
    #         T=35 + 273, p=2.0e6, sol="CO2", pol="PS", k_sw=0.0033954, frc_step=step
    #     )
    #     for step in step_array
    # ]
    # plt.scatter(step_array, dmuad_dlnomega)
    # plt.show()
    # pressure
    # pressure_array = linspace(10, 1e6, 10)
    # dmuad_dlnomega = [
    #     get_therm_factor(
    #         T=35 + 273, p=_p, sol="CO2", pol="PS", k_sw=0.0051, frc_step=0.01
    #     )
    #     for _p in pressure_array
    # ]
    # plt.scatter(pressure_array, dmuad_dlnomega)
    # plt.show()
    # test FFV
    # print("1/FFV = ",1/get_FFV("PS"))

    # Test get_S10_EQ
    # print(get_S10_EQ(T=35 + 273, sol="CO2", pol="PS"))

    # # Test S10NE
    # S10 = get_S10_NE(T=35 + 273, sol="CO2", pol="PS")
    # print("S10 = {:.3g} MPa^-1".format(S10))

    # Test S10
    # S10 = get_S10_NE(T=35+273, sol="CO2", pol="PS", k_sw=0.0051)
    # print("S10 = {:.3g} MPa^-1".format(S10))

    # test L10
    # L10 = get_L10("CO2","PS")
    # print("L10 = %g\tcm^2/s"%L10)

    # test L1
    # L1 = get_L1(T=35 + 273, p=0.001e6, sol="CO2", pol="PS", k_sw=0.0051)

    # test P10
    # P10 = get_P10(T=35 + 273, sol="CO2", pol="PMMA", rho20=0.9959, k_sw=0.00669, unit = "barrer")
    # print("P10 = {:.3g} barrer".format(P10))

    # test Z1
    # Z1 = get_Z1(T=45 + 273, p=300e5, sol="CO2")
    # print("Z1 = ", Z1)

    # * Test P1
    # start_time = time.time()
    # P1 = get_P1(
    #     T=35 + 273,
    #     p_u=5 * 101325,
    #     p_d=0.01,
    #     sol="CO2",
    #     pol="PMMA",
    #     rho20=0.99591,
    #     ksw=0.0066917,
    #     unit="barrer",
    # )
    # print("P1 = {:.3g} barrer".format(P1))
    # print("\n--- Run time:\t%.0f seconds ---\n" % (time.time() - start_time))

    # * Test plot_permeability_isotherm
    # start_time = time.time()
    # plot_permeability_isotherm_NE(
    #     T=35 + 273,
    #     p_l=0.01,
    #     p_u=21 * 101325,
    #     no_of_points=3,
    #     sol="CO2",
    #     pol="PMMA",
    #     ksw_list=[0.006692, 0.014418],
    #     rho20=0.99592,
    #     xlxs_sheet_refno_list = [30],
    #     display_plot=True
    # )
    # print("\n--- Run time:\t%.0f seconds ---\n" % (time.time() - start_time))

    # Test get_chi
    print(get_chi("PMMA"))

    # Test get_dVeq_df0
    # frc_arr = [1e-5,1e-4,1e-3,1e-2,1e-1]
    # result=[get_dVeq_df0(T=35+273,sol="CO2",pol="PS",frc_step=step) for step in frc_arr]
    # print(frc_arr)
    # print(result)

    # p0_array = linspace(0.1,10,10)
    # result=[get_dVeq_df0(T=35+273,p0=pi,sol="CO2",pol="PS") for pi in p0_array]
    # print(p0_array)
    # print(result)
    # plt.scatter(p0_array,result)
    # plt.show()
    # print(get_dVeq_df0(T=35+273,p0=1,sol="CO2",pol="PS"))

    # Test import_MultiTait_parameters
    # print(import_MultiTait_parameters("PEMA"))

    # Test get_chi
    # print(get_chi("PEMA"))

    # * Test ksw prediction
    # ksw_p = predict_ksw_NE(T=81+273, sol="CO2", pol="PS", rho20=1.0298)
    # print("ksw_p = ", ksw_p)

    # * Test plot_isotherm_multi_ksw
    # plot_isotherm_NE(
    #     sol="CO2",
    #     pol="PMMA",
    #     T=30 + 273,
    #     p_l=1,
    #     p_u=1e6,
    #     no_of_points=5,
    #     rho20=1.1514327111737948,
    #     ksw_list=[0.001],
    #     xlxs_sheet_refno_list=["29"]
    # )

    # solve_solubility_NE(T=35 + 273, p=4e6, sol="CO2", pol="PMMA", ksw=0.00, rho20=1.1514, MW2=100e3)
    # print(a)
    # Test get_ksw_prediction
    # start_time = time.time()
    # ksw = get_ksw_prediction(T=35 + 273, sol="CO2", pol="PS",rho20=1.05301)
    # ksw_parity_AADpc = get_fitting_AAD(array([ksw]).tolist(),array([ksw]).tolist())*100
    # print("ksw = {:.3g} MPa^-1".format(ksw))
    # print(f"AAD% = {ksw_parity_AADpc}%",)
    # print("\n--- Run time:\t%.0f seconds ---\n" % (time.time() - start_time))

    # * Test plot_ksw_parity
    # plot_ksw_parity(
    #     xlxs_sheet="231016_woParam_rhoPVT",
    #     display_plot=False,
    #     save_plot_dir="C:\\Users\\sn621\\Downloads\\ksw_WOparam_rhoPVT.png"
    #     )

    # * Test plot_ksw_AAD
    # start_time = time.time()
    # plot_ksw_AAD(sol_list=["CO2"],display_plot=False,save_plot_dir="C:\\Users\\sn621\\Downloads\\ksw_AAD.png")
    # print("\n--- Run time:\t%.0f seconds ---\n" % (time.time() - start_time))

    # * Test plot_rho2_parity
    # plot_rho20_parity(
    #     xlxs_sheet="231016_woParam_rhoPVT",
    #     display_plot=False,
    #     save_plot_dir="C:\\Users\\sn621\\Downloads\\rho20parity_WOparam.png"
    #     )

    # # * Test plot_pg_T
    # plot_pg_T(xlxs_sheet="231016_woParam_rhof",
    #           display_plot=False,
    #           save_plot_dir="C:\\Users\\sn621\\Downloads\\pg_T_WOparam_rhof.png"
    #           )
    # * Test plot_pg_Sg
    # plot_pg_Sg(display_plot=False,
    #            save_plot_dir="C:\\Users\\sn621\\Downloads\\pg_Sg.png")
    # * Test plot_Sg_T
    # plot_Sg_T(xlxs_sheet="231016_woParam_rhoPVT",
    #           display_plot=False,
    #           save_plot_dir="C:\\Users\\sn621\\Downloads\\Sg_T_WOparam_rhoPVT.png"
    #           )

    # * Test plot_density or volume _isotherm_EQvME
    # plot_density_isotherm_EQvNE(
    #     p_l=1,
    #     p_u=20e6,
    #     no_of_points=10,
    #     T=33 + 273,
    #     sol="CO2",
    #     pol="PS",
    #     k_sw=[0.027],
    # )
    # plot_V_isotherm_EQvNE(
    #     p_l=1,
    #     p_u=20e6,
    #     no_of_points=10,
    #     T=50 + 273,
    #     sol="CO2",
    #     pol="HDPE",
    #     ksw_list=[],
    #     rho20=1.0242,
    # )
    # plot_dV_V0_isotherm_EQvNE(
    #     p_l=1,
    #     p_u=20e6,
    #     no_of_points=10,
    #     T=35 + 273,
    #     sol="CO2",
    #     pol="PS",
    #     k_sw=[0.00391],
    # )
    # * Test get_p_EQvNE_intersect and compare with plot
    # pg, Sg = plot_V_isotherm_EQvNEvCombined(
    #     p_l=1,
    #     p_u=20e6,
    #     no_of_points=10,
    #     T=35 + 273,
    #     sol="CO2",
    #     pol="PS",
    #     ksw_list=[0.002693],
    #     rho20=1.05301,
    # )
    # print(pg, Sg)
    # pg = get_p_intersect_EQvNE(T=35+273, sol="CO2",pol="PS",k_sw=0.00391,display_result=True)
    # print(pg)

    # * Test plot_Ldensity_EQ
    # plot_pol_Ldensity_isobar_EQ(pol="PS")
    # plot_pol_Ldenisty_isobar_EQ_MW2sensitivity(MW2_list=linspace(1e3,1000e3,100),
    #                                            T=100+273,
    #                                            p_list=arange(0,200e6,20e6),
    #                                            pol="PMMA")

    # * fit_multiTait_polPVT
    # fit_polPVT_multiTait("PMMA_glassy")
    V20 = get_V20_multiTait(T=81 + 273, p=0, pol="PS")
    print(f"rho20 = {1/V20} g/cm^3")
