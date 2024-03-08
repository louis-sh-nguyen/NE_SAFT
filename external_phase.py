import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import os
from pcsaft import pcsaft_den


R_const = 8.314  # [Pa m^3 mol^-1 K^-1 = MPa cm^3 mol^-1 K^-1]

excel_file_path = os.path.join(os.path.dirname(
    __file__), 'litdata/pol-sol_parameters_150223.xlsx')
# excel_file = "pol-sol_parameters_150223.xlsx"
# print(excel_file_path)


def muad_SLEOS(sol, T, p):
    """_summary_

    Args:
        sol (str): Name of solute as appeared in the Excel sheet.
        T (float): Temperature [K].
        p (float): Pressure [MPa].

    Returns:
        mu_RT (float): mu_1/RT [adim]
    """

    # Importing parameters from Excel file
    try:
        dfSL = pd.read_excel(excel_file_path, sheet_name="SL")
        dfSL.set_index("Species", inplace=True)
        dfMW = pd.read_excel(excel_file_path, sheet_name="MW")
        dfMW.set_index("Species", inplace=True)
    except:
        print("!File was NOT read.")
    else:
        try:  # solute name
            dfSL_filter_sol = dfSL.loc[sol]
            dfMW_filter_sol = dfMW.loc[sol]
            # print("Parameters available.")
        except:
            print("\nData is NOT available for %s." % sol)
        else:
            if (
                dfSL_filter_sol[["T* (K)", "p* (MPa)", "V* (cm3/g)"]]
                .isnull()
                .values.any()
                == True
            ):
                print("!Null values for %s in SL sheet" % (sol))
            if dfMW_filter_sol.isnull().values.any() == True:
                print("!Null values for %s in MW sheet" % (sol))
            T_star_sol = dfSL_filter_sol["T* (K)"].item()  # K
            p_star_sol = dfSL_filter_sol["p* (MPa)"].item()  # MPa
            V_star_sol = dfSL_filter_sol["V* (cm3/g)"].item()  # cm^3/g
            MW_sol = dfMW_filter_sol["MW"].item()  # g/mol

    """
    Working units:
    T [=] K
    p [=] MPa
    V [=] cm^3/g
    rho [=] g/cm^3
    """

    rho_star_sol = 1 / V_star_sol
    T_tilde_sol = T / T_star_sol
    p_tilde_sol = p / p_star_sol
    r_0_sol = (MW_sol * p_star_sol) / (R_const * T_star_sol * rho_star_sol)

    def func_SL(variables):  # SL EOS equations
        rho_tilde_sol = variables[0]

        LHS_1 = np.log(1 - rho_tilde_sol)
        RHS_1 = (
            -(rho_tilde_sol ** 2) / T_tilde_sol
            - (1 - 1 / r_0_sol) * rho_tilde_sol
            - p_tilde_sol / T_tilde_sol
        )
        eq_SLEOS = LHS_1 - RHS_1

        return [eq_SLEOS]

    x0_array = np.linspace(0.0001, 0.1, 100)
    i = 0
    while i <= (len(x0_array) - 1):
        try:
            # Solving using fsolve with initial points x0
            solution = fsolve(func_SL, x0=[x0_array[i]])
            residue = func_SL(variables=solution)
            # convert to float for isclose()
            residue_float = [float(i) for i in residue]
            if np.isclose(residue_float, [0.0]).all() == True:
                # print("\tx0 = %g ACCEPTED for SL EOS." % x0_array[i])
                rho_tilde_sol = solution[0]
                # rho_sol = rho_tilde_sol * rho_star_sol
                mu_RT = r_0_sol * (-rho_tilde_sol / T_tilde_sol
                                   + p_tilde_sol /
                                   (T_tilde_sol * rho_tilde_sol)
                                   + np.log(rho_tilde_sol) / r_0_sol
                                   + ((1 - rho_tilde_sol) / rho_tilde_sol) *
                                   np.log(1 - rho_tilde_sol)
                                   )
            return mu_RT
        except:
            print(
                "\tx0 = %g NOT accepted for SL EOS. Moving to x0 = %g."
                % (x0_array[i], x0_array[i + 1])
            )
            i += 1

    if i >= (len(x0_array) - 1):
        print("\n\tNo solutions found in SL EOS.")


def muad_PCSAFT(sol, T, p_MPa):
    """Calculate dimensionless chemical potential using parameters from PCSAFT and SL EOS formula

    Args:
        sol (string): solute name
        T (float): temperature [K]
        p_MPa (float): pressure [MPa]

    Returns:
        float: dimensionless chemical potential (i.e. muad).
    """
    p_Pa = p_MPa * 1e6
    try:  # file name
        df = pd.read_excel(
            excel_file_path, sheet_name="PC-SAFT", engine='openpyxl')
        df.set_index("Species", inplace=True)
        dfMW = pd.read_excel(
            excel_file_path, sheet_name="MW", engine='openpyxl')
        dfMW.set_index("Species", inplace=True)
        dfSL = pd.read_excel(excel_file_path,
                             sheet_name="SL", engine='openpyxl')
        dfSL.set_index("Species", inplace=True)
        dfSL_filter_sol = dfSL.loc[sol]
    except:
        print("Excel file was NOT successfully read.")
    else:
        # rho from PCSAFT
        dfMW_filter_sol = dfMW.loc[sol]
        MW_sol = dfMW_filter_sol["MW"].item()  # [g/mol]
        df_filter = df.loc[sol]
        x = np.asarray([1.0])
        m = np.asarray([df_filter["m"].item()])
        s = np.asarray([df_filter["sigma (Angstrong)"].item()])
        e = np.asarray([df_filter["epsilon (K)"].item()])
        pyargs = {"m": m, "s": s, "e": e}
        rho_sol = pcsaft_den(T, p_Pa, x, pyargs, phase="vap")  # [mol/m^3]
        rho_sol = np.asarray(rho_sol)*MW_sol*1e-6  # [g/cm^3]

        # SL EOS parameters
        V_star_sol = dfSL_filter_sol["V* (cm3/g)"].item()  # [cm^3/g]
        p_star_sol = dfSL_filter_sol["p* (MPa)"].item()  # [MPa]
        T_star_sol = dfSL_filter_sol["T* (K)"].item()  # K
        rho_star_sol = 1 / V_star_sol  # [g/cm^3]

        T_tilde_sol = T / T_star_sol
        p_tilde_sol = p_MPa / p_star_sol
        rho_tilde_sol = rho_sol/rho_star_sol
        r_0_sol = (MW_sol * p_star_sol) / (R_const *
                                           T_star_sol * rho_star_sol)  # eq.13#✓
        # Calculate muad based on SL EOS
        mu_RT = r_0_sol * (-rho_tilde_sol / T_tilde_sol
                           + p_tilde_sol /
                           (T_tilde_sol * rho_tilde_sol)
                           + np.log(rho_tilde_sol) / r_0_sol
                           + ((1 - rho_tilde_sol) / rho_tilde_sol) *
                           np.log(1 - rho_tilde_sol)
                           )  # [J/mol]
        
        return mu_RT


if __name__ == "__main__":
    T = 448 # [K]
    p_MPa = 5   # [MPa]
    mu=muad_SLEOS("CO2", T, p_MPa)*(8.314 * T)
    print(f"mu = {mu} J/mol")
    # print("PCSAFT ",muad_PCSAFT("CO2", 40 + 273, 10))
