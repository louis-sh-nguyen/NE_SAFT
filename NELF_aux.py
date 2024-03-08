import pandas as pd
import os
from numpy import *
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit, fsolve
import NET_SAFTgMie_master as NESAFT
R_const = 8.314


def import_NELF_parameters(sol, pol):
    try:  # file name
        databasepath = os.path.join(os.path.dirname(__file__), "litdata")
        databasepath += "/pol-sol_parameters_150223.xlsx"
        file = pd.ExcelFile(databasepath, engine="openpyxl")
        dfSL = pd.read_excel(file, sheet_name="SL")
        dfSL.set_index("Species", inplace=True)
        dfMW = pd.read_excel(file, sheet_name="MW")
        dfMW.set_index("Species", inplace=True)
    except Exception as e:
        print("")
        print(e)
    else:
        try:  # polymer name
            dfSL_filter_pol = dfSL.loc[pol]
            dfSL_filter_sol = dfSL.loc[sol]
            dfMW_filter_sol = dfMW.loc[sol]
            # print("Parameters available.")
        except Exception as e:
            print("")
            print(e)
        else:
            if (
                dfSL_filter_sol[["T* (K)", "p* (MPa)", "V* (cm3/g)"]]
                .isnull()
                .values.any()
                == True
            ):
                print("Null values for %s in SL sheet" % (sol))
            if (
                dfSL_filter_pol[["T* (K)", "p* (MPa)", "V* (cm3/g)"]]
                .isnull()
                .values.any()
                == True
            ):
                print("Null values for %s in SL sheet" % (pol))
            if dfMW_filter_sol.isnull().values.any() == True:
                print("Null values for %s in MW sheet" % (sol))
            T_star_pol = dfSL_filter_pol["T* (K)"].item()  # K
            p_star_pol = dfSL_filter_pol["p* (MPa)"].item()  # MPa
            V_star_pol = dfSL_filter_pol["V* (cm3/g)"].item()  # cm^3/g
            T_star_sol = dfSL_filter_sol["T* (K)"].item()  # K
            p_star_sol = dfSL_filter_sol["p* (MPa)"].item()  # MPa
            V_star_sol = dfSL_filter_sol["V* (cm3/g)"].item()  # cm^3/g
            MW_sol = dfMW_filter_sol["MW"].item()  # g/mol
            return (
                T_star_pol,
                p_star_pol,
                V_star_pol,
                T_star_sol,
                p_star_sol,
                V_star_sol,
                MW_sol,
            )


def SLEOS_aux(sol, pol, T, p):
    p_MPa = p * 1e-6
    (
        T_star_pol,
        p_star_pol,
        V_star_pol,
        T_star_sol,
        p_star_sol,
        V_star_sol,
        MW_sol,
    ) = import_NELF_parameters(sol, pol)
    rho_star_sol = 1 / V_star_sol  # ✓
    T_tilde_sol = T / T_star_sol  # ✓
    p_tilde_sol = p_MPa / p_star_sol  # ✓
    r_0_sol = (MW_sol * p_star_sol) / (R_const * T_star_sol * rho_star_sol)  # eq.13#✓

    def func_SL(variables):  # SL EOS equations
        rho_tilde_sol = variables[0]
        LHS_1 = log(1 - rho_tilde_sol)
        RHS_1 = (
            -(rho_tilde_sol**2) / T_tilde_sol
            - (1 - 1 / r_0_sol) * rho_tilde_sol
            - p_tilde_sol / T_tilde_sol
        )  # use r_0_sol
        eq_SLEOS = LHS_1 - RHS_1  # eq.22#✓
        return [eq_SLEOS]

    x0_array = linspace(0.0001, 0.1, 10)
    i = 0
    while i <= (len(x0_array) - 1):
        try:
            # Solving using fsolve with initial points x0
            solution = fsolve(func_SL, x0=[x0_array[i]])
            residue = func_SL(variables=solution)
            # convert to float for isclose()
            residue_float = [float(i) for i in residue]
            if isclose(residue_float, [0.0]).all() == True:
                rho_tilde_sol = solution[0]
                return rho_tilde_sol
        except:
            print(
                "\tx0 = %g NOT accepted for SL EOS. Moving to x0 = %g."
                % (x0_array[i], x0_array[i + 1])
            )
            i += 1

    if i >= (len(x0_array) - 1):
        print("\n\tNo solutions found in SL EOS.")


def solve_solubility_NELF(sol, pol, T, p, rho_pol_am, kij, ksw):
    (
        T_star_pol,
        p_star_pol,
        V_star_pol,
        T_star_sol,
        p_star_sol,
        V_star_sol,
        MW_sol,
    ) = import_NELF_parameters(sol, pol)
    p_MPa = p * 1e-6
    """
    Working units:
    T [=] K
    p [=] MPa
    V [=] cm^3/g
    rho [=] g/cm^3    """

    xi = 1 - kij
    rho_pol = rho_pol_am * (1 - ksw * p_MPa)
    rho_star_pol = 1 / V_star_pol
    T_tilde_pol = T / T_star_pol
    p_tilde_pol = p / p_star_pol
    rho_star_sol = 1 / V_star_sol
    T_tilde_sol = T / T_star_sol
    p_tilde_sol = p_MPa / p_star_sol
    r_0_sol = (MW_sol * p_star_sol) / (R_const * T_star_sol * rho_star_sol)  # eq.13#✓
    v_star_sol = R_const * T_star_sol / p_star_sol  # eq.14
    v_star_pol = R_const * T_star_pol / p_star_pol  # eq.14
    rho_tilde_sol = SLEOS_aux(sol, pol, T, p)
    # print("rho_tilde_sol = ", rho_tilde_sol)

    def func(variables):  # main equations
        omega_sol = variables[0]

        omega_pol = 1 - omega_sol
        phi_sol = (omega_sol / rho_star_sol) / (
            omega_sol / rho_star_sol + omega_pol / rho_star_pol
        )  # eq.15✓
        phi_pol = 1 - phi_sol  # eq.15✓

        v_star = (
            v_star_sol * v_star_pol / (phi_sol * v_star_pol + phi_pol * v_star_sol)
        )  # eq.17✓
        r_sol = v_star_sol * r_0_sol / v_star  # eq.16✓
        rho_star = (
            rho_star_sol
            * rho_star_pol
            / (omega_sol * rho_star_pol + omega_pol * rho_star_sol)
        )  # eq.18✓
        rho_tilde = rho_pol / (omega_pol * rho_star)  # ✓
        delta_p_star = (
            p_star_sol + p_star_pol - 2 * xi * (p_star_sol * p_star_pol) ** 0.5
        )  # eq.21✓
        p_star = (
            phi_sol * p_star_sol
            + phi_pol * p_star_pol
            - phi_sol * phi_pol * delta_p_star
        )  # eq.20✓

        # RHS_2 = (log(rho_tilde * phi_sol)  # mu/RT #*original
        #     - (r_0_sol + (r_sol - r_0_sol) / rho_tilde) * log(1 - rho_tilde)
        #     - r_sol
        #     - rho_tilde* (r_0_sol* v_star_sol* (p_star_sol + p_star - (phi_pol**2) * delta_p_star))
        #     / (R_const * T)  # eq.12✓
        # )
        RHS_2 = (   #*new
            log(rho_tilde * phi_sol)
            - (r_0_sol + (r_sol - r_0_sol) / rho_tilde) * log(1 - rho_tilde)
            - r_sol
            - rho_tilde
            * (
                r_0_sol
                * v_star_sol
                * (p_star_sol + p_star_pol - (phi_pol ** 2) * delta_p_star)
            )/ (R_const * T) 
        )        
        LHS_2 = (  # mu/RT
            log(rho_tilde_sol)
            - r_0_sol * log(1 - rho_tilde_sol)
            - r_0_sol
            - (rho_tilde_sol * r_0_sol * v_star_sol * p_star_sol) / (R_const * T))

        eq_NELF = LHS_2 - RHS_2
        return [eq_NELF]

    x0_array = linspace(0.00001, 0.2, 20)
    i = 0
    while i <= (len(x0_array) - 1):
        try:  # solving
            # Solving using fsolve with initial points x0
            solution = fsolve(func, x0=[x0_array[i]])
            residue = func(variables=solution)
            
            # convert to float for isclose()
            residue_float = [float(j) for j in residue]
            if isclose(residue_float, [0.0]).all() == True:
                final_solution = solution
                omega_sol = final_solution[0]
                S_1 = omega_sol / (1 - omega_sol)  # [g/g]
            
                return S_1
        except:
            print(
                "\t\tx0 = %g NOT accepted for NELF."
                % (x0_array[i])
            )
            i += 1

    if i >= (len(x0_array) - 1):
        print("\n\tNo solutions found in NELF.")

def fit_rho20_NELF(
    T: float,
    sol: str,
    pol: str,
    xlxs_sheet_refno: str,
    kij: float,
    rho20_x0_list: list[float] = linspace(0.7, 1.5, 20),
    display_plot: bool = True,
    save_plot_dir: str = None,
) -> tuple[float, float]:

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
        solubility_list = [solve_solubility_NELF(sol=sol, pol=pol, T=T, p=p_, rho_pol_am=rho20_fit, ksw=0, kij=kij) for p_ in p_list_]  # [g/g]
        return solubility_list

    x0_list = rho20_x0_list
    # start time
    start_time = time.time()
    for i, _rho20_0_ in enumerate(x0_list):
        try:
            # print("rho20_0 = ", _rho20_0_)
            # Fitting function
            result, cov = curve_fit(rho20_fitting_func, p_exp_trimmed, solubility_exp_trimmed, p0=(_rho20_0_))

        except Exception as e:
            print("Fitting unsuccessful for rho20_0 = ", _rho20_0_)
            if i == (len(x0_list) - 1):
                print("Error - curve_fit unsuccessful.")
            pass
        else:
            print("")
            print("Fitting done. Fitting time: \t %.0f seconds" % (time.time() - start_time))
            rho20_f = result[0]  # [g/cm^3]
            print("rho20 = %g g/cm^3" % rho20_f)
            break

    # fitting error
    print("rhof = ", rho20_f)
    solubility_calc_evaluation = rho20_fitting_func(p_exp_trimmed, rho20_f)
    print("solubility_calc_evaluation = ", solubility_calc_evaluation)
    try:
        AAD_percent = NESAFT.get_fitting_AAD(solubility_exp_trimmed, solubility_calc_evaluation) * 100  # [%]
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
    print("DONE1")
    solubility_calc_full = rho20_fitting_func(p_calc_full, rho20_f)
    print("DONE2")

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # trimmed exp data (fitted)
    ax.plot(
        p_MPa_exp_trimmed,
        solubility_exp_trimmed,
        color="black",
        marker=NESAFT.exp_style["marker"],
        linestyle=NESAFT.exp_style["linestyle"],
        # markerfacecolor=exp_style["markerfacecolor"],
        label=f"fitting exp data: {ref_ID} ({ref_no})",
    )
    # remaining exp data
    ax.plot(
        p_MPa_exp_full[2:],  # skip first 2 used in trimmed data
        solubility_exp_full[2:],  # skip first 2 used in trimmed data
        color="grey",
        marker=NESAFT.exp_style["marker"],
        linestyle=NESAFT.exp_style["linestyle"],
        markerfacecolor=NESAFT.exp_style["markerfacecolor"],
        label=f"unused exp data: {ref_ID} ({ref_no})",
    )

    # calculated from fited rho20
    ax.plot(
        p_MPa_calc_full,
        solubility_calc_full,
        color="black",
        marker=NESAFT.calc_style["marker"],
        linestyle=NESAFT.calc_style["linestyle"],
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
        plt.savefig(save_plot_dir, dpi=1200)
        print(f"Plot saved: {save_plot_dir}")
        print("")
    return rho20_f[0], AAD_percent

def fit_ksw_NELF(
    T: float,
    sol: str,
    pol: str,
    xlxs_sheet_refno: str,
    rho20: float,
    kij: float,
    ksw_x0_list: list[float] = linspace(0.0, 0.01, 10),
    glassy_filter: bool = False,
    display_plot: bool = True,
    save_plot_dir: str = None,
) -> tuple[float, float]:

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
        solubility_list = [solve_solubility_NELF(sol=sol, pol=pol, T=T, p=p_, rho_pol_am=rho20, ksw=ksw_fit, kij=kij) for p_ in p_list_]
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
            ksw_f = result[0]
            print("k_sw = %g MPa^-1" % ksw_f)
            break

    # Fitting error
    solubility_calc_evaluation = ksw_fitting_func(p_exp, ksw_f)
    try:
        AAD_percent = NESAFT.get_fitting_AAD(solubility_exp, solubility_calc_evaluation) * 100  # [%]
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
        marker=NESAFT.exp_style["marker"],
        linestyle=NESAFT.exp_style["linestyle"],
        # markerfacecolor=exp_style["markerfacecolor"],
        label=f"fitting exp data: {ref_ID} ({ref_no})",
    )
    # remaining exp data
    if len(df.index) != len(df1.index):
        ax.plot(
            df[df["_fit_ksw_NE?_"] != "Yes"]["P [MPa]"],
            df[df["_fit_ksw_NE?_"] != "Yes"]["Solubility [g-sol/g-pol-am]"],
            color="grey",
            marker=NESAFT.exp_style["marker"],
            linestyle=NESAFT.exp_style["linestyle"],
            markerfacecolor=NESAFT.exp_style["markerfacecolor"],
            label=f"unused exp data: {ref_ID} ({ref_no})",
        )
    # calculated from fitting values
    ax.plot(
        p_MPa_calc,
        solubility_calc,
        color=NESAFT.calc_style["color"],
        marker=NESAFT.calc_style["marker"],
        linestyle=NESAFT.calc_style["linestyle"],
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




if __name__ == "__main__":
    # print(
    #     solve_solubility_NELF(
    #         # sol="CO2", pol="PEEK", T=115 + 273, p=10e6, rho_pol_am=1.262, kij=0, ksw=0,
    #         sol="CO2", pol="PS", T=35 + 273, p=1000, rho_pol_am=1.056, kij=0.032, ksw=0.0026949
    #     )
    # )
    
    # fit_rho20_NELF(T=35+273, sol="CO2", pol="PS", kij=0.032, xlxs_sheet_refno="16",
    #                rho20_x0_list=linspace(1.0, 1.3, 5))
    fit_ksw_NELF(T=35+273, sol="CO2", pol="PS", kij=0.032, rho20= 1.02239, xlxs_sheet_refno="16",)
