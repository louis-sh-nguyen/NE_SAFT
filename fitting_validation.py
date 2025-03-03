import math
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import NET_SAFTgMie_master as NE_SAFT
import numpy as np
import re
import logging
from functools import wraps


def test_mu():
    # Hilic 2001 (20), T = 130°C
    T = 130 + 273
    p_list = [5140000.0, 9390000.0, 21480000.0, 33049999.999999996, 44410000.0]   
    
    # v3.4
    mu_gPROMS = [1.10693832202300073E+04, 1.45364068817818315E+04, 1.69807355858596857E+04, 1.81524226527682804E+04, 1.88232335930901972E+04]
    
    mu_sgtpy = [NE_SAFT.solve_solubility_EQ(T=T, p=_p, sol="CO2", pol="PS", return_extended=True)[5]  * (8.314 * T) \
        for _p in p_list]
    print("mu_gPROMS =", mu_gPROMS)
    print("mu_sgtpy =", mu_sgtpy)

def validate_mu_gPROMS_sgt(
    T: float,
    sol: str,
    pol: str,
    p_list: list,
    x_sol_gPROMS: list,
    mu_gPROMS: list,):
    
    # check if p_list, x_sol_gPROMS, mu_gPROMS have the same length
    if not len(p_list) == len(x_sol_gPROMS) == len(mu_gPROMS):
        raise ValueError("p_list, x_sol_gPROMS, mu_gPROMS must have the same length")
    
    MW2 = 104 * 5000    # matching gPROMS
    
    # calculate mu_sgtpy from x_sol_gPROMS and p_list
    mu_from_x = [NE_SAFT.get_mu_from_x(T=T, p=p, x_sol=x, sol=sol, pol=pol, MW2=MW2) for p, x in zip(p_list, x_sol_gPROMS)]
    
    # Calculate percentage error between mu_gPROMS and mu_sgtpy
    percentage_error = [abs((mu_gPROMS[i] - mu_from_x[i]) / mu_gPROMS[i]) * 100 for i in range(len(mu_gPROMS))]
        
    # Create dataframe to compare mu_gPROMS and mu_sgtpy and percentage error
    df = pd.DataFrame({
        "p": p_list,
        "x_sol_gPROMS": x_sol_gPROMS,
        "mu_gPROMS": mu_gPROMS,
        "mu_sgtpy": mu_from_x,
        "percentage_error": percentage_error
    })
    
    print(df)    

if __name__ == "__main__":
    validate_mu_gPROMS_sgt(
        sol="CO2",
        pol="PS",
        
        # v3.5, T = 130°C
        # T = 130 + 273,
        # p_list=[5140000.0, 9390000.0, 21480000.0, 33049999.999999996, 44410000.0],
        # x_sol_gPROMS=[0.994245658087466, 0.9980853915374306, 0.9991193659434793, 0.9993731479370868, 0.9994297179177446],        
        # mu_gPROMS=[1.11907594191769567E+04, 1.45725221669827770E+04, 1.69101053198714471E+04, 1.80288355606673904E+04, 1.86973634812836644E+04],
        
        # v3.5, T = 110°C
        # T = 110 + 273,
        # p_list=[6.55000000000000000E+06, 1.10400000000000000E+07, 1.45100000000000000E+07, 1.76600000000000000E+07, 2.04200000000000000E+07, 2.35500000000000000E+07, 2.70300000000000000E+07, 3.06100000000000000E+07, 3.43800000000000000E+07, 3.86300000000000000E+07, 4.28100000000000000E+07],
        # x_sol_gPROMS=[9.97653133067355036E-01, 9.98544992791100539E-01, 9.98860929895412419E-01, 9.99087946802424187E-01, 9.99195859406982967E-01, 9.99273342229386130E-01, 9.99313156793815827E-01, 9.99387841417536671E-01, 9.99413592916202353E-01, 9.99437676684541865E-01, 9.99473275936371697E-01],
        # mu_gPROMS=[1.29354794974639335E+04, 1.43150747897532365E+04, 1.50078149360258394E+04, 1.55875701317747044E+04, 1.59374061779119547E+04, 1.62475137619172092E+04, 1.64947162604871664E+04, 1.68248685029147164E+04, 1.70554784741351868E+04, 1.73038069082727598E+04, 1.75742354986935752E+04],
        
        # v2.4, T = 200°C
        T = 200 + 273,
        p_list=[2.16700000000000000E+06, 2.88400000000000000E+06, 4.23200000000000000E+06, 4.79500000000000000E+06, 6.10800000000000000E+06, 6.68500000000000000E+06, 8.02699999999999907E+06, 8.81300000000000000E+06, 1.21650000000000000E+07, 1.26590000000000000E+07, 1.60400000000000000E+07, 2.01510000000000000E+07],
        x_sol_gPROMS=[9.88287904599659206E-01, 9.91153289367862178E-01, 9.93949394939493969E-01, 9.94649545211342856E-01, 9.95805848934304572E-01, 9.96166846708715137E-01, 9.96806317684290066E-01, 9.97095709570957056E-01, 9.97867803837952927E-01, 9.97954896164500949E-01, 9.98355213971709676E-01, 9.98643867197613333E-01],
        mu_gPROMS=[1.02382137240169250E+04, 1.13396556904126010E+04, 1.28199672179018689E+04, 1.32967001414035185E+04, 1.42350421029205045E+04, 1.45806353037358676E+04, 1.52789055820028880E+04, 1.56401121910463226E+04, 1.68196456230762560E+04, 1.69760110704342551E+04, 1.78132154334999504E+04, 1.85793938786266444E+04],
        )
