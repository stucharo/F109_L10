import sys
import os
sys.path.insert(0, os.getcwd())

import src.soil.dnv_soil as dnv
import src.hydrodynamics.waves as w
import src.hydrodynamics.current as c
import src.route_selection.lcp_lib as lcp
import src.bottom_roughness.bottom_roughness as br
import numpy as np
from math import sqrt
from scipy.constants import g
from scipy.interpolate import interp2d
from scipy.optimize import newton

def overall_stability(Hs, Tp, d, spectrum, theta_w, s, OD, Vc, z0,     
                      zr, theta_c, soil_type, rho_w, su, gamma, d50,
                      rho_p, Fc, wt, location, z_t, theta_t, mu, grade,
                      T_ss, kv, Fz):

    w_t = wt

    w10D = general_stability_10D(Hs, Tp, d, spectrum, theta_w, s, OD, Vc,
                                 z0, zr, theta_c, soil_type, rho_w, su,
                                 gamma, d50)

    w05D = general_stability_05D(Hs, Tp, d, theta_w, s, Vc, z0, zr, 
                                  theta_c, su, gamma, OD, soil_type, rho_w,
                                  Fc, wt, rho_p, spectrum)
    
    SFs_abs = absolute_stability(rho_w, rho_p, OD, mu, gamma, su, soil_type,
                                location, grade, w_t, Hs, Tp, d, s, T_ss,
                                kv, Fc, theta_t, z_t, Vc,z0, zr, theta_c, 
                                theta_w, Fz, spectrum)

    w_s = br._get_submerged_weight(OD, w_t, rho_p, rho_w)

    SF_w10D = w10D / w_s
    SF_w05D = w05D / w_s
    SF_1ABS = SFs_abs[0]
    SF_2ABS = SFs_abs[1]

    return (SF_w10D, SF_w05D, SF_1ABS, SF_2ABS)


def w_t_10D(Hs, Tp, d, spectrum, theta_w, s, OD, Vc, z0, zr,
            theta_c, soil_type, rho_w, su, gamma, d50, rho_p,):
    

    def SF(w_t):
        w10D = general_stability_10D(Hs, Tp, d, spectrum, theta_w, s, OD, Vc,
                                 z0, zr, theta_c, soil_type, rho_w, su,
                                 gamma, d50)

        w_s = br._get_submerged_weight(OD, w_t, rho_p, rho_w)
        return (w10D / w_s) - 1

    w_t_10D = newton(SF, 10)

    return w_t_10D

def general_stability_10D(Hs, Tp, d, spectrum, theta_w, s, OD, Vc,
                          z0, zr, theta_c, soil_type, rho_w, su,
                          gamma, d50):
    """
        General stability is determined by calculating the minimum
        pipe weight that is required to keep the pipe stable whilst on
        the seabed. This is then compared to the known weight of the pipe
        and a stability assessment made. 
        
        This calculation is to do with the minimum pipe weight required to 
        keep the pipe stable to a lateral displacement of 10 x the diameter
        of the pipe. 
    """

    if soil_type == 'Sand':
        stab_L = get_L10_sand(Hs, Tp, d, spectrum, theta_w, s, OD, Vc, z0, zr, theta_c)
    elif soil_type == 'Clay':
        stab_L = get_L_10_clay(su, OD, gamma, Hs, Tp, d, spectrum, theta_w, s, Vc, z0, zr,
                               theta_c, d50)
    else:
        raise TypeError("Wrong soil type entered")

    return stab_L * 0.5 * rho_w * OD * (2 * sqrt(w.get_M(0, Hs, Tp, d, spectrum)))**2


def general_stability_05D(Hs, Tp, d, theta_w, s, Vc,
                      z0, zr, theta_c, su, gammas, OD,
                      soil_type, rho_w, Fc, wt, rho_p, spectrum):
    """
        General stability is determined by calculating the minimum
        pipe weight that is required to keep the pipe stable whilst on
        the seabed. This is then compared to the known weight of the pipe
        and a stability assessment made. 
        
        This calculation is to do with the minimum pipe weight required to 
        keep the pipe stable to a lateral displacement of 0.5 x the diameter
        of the pipe. 
 
    """
    # Use previous calculations to get the factor used to calculate the
    # minimum pipe weight required for stability. 
    min_L_clay = get_L_stable_clay(Hs, Tp, d, theta_w, s,
                                   Vc, z0, zr, theta_c, su, gammas,
                                   OD, Fc, spectrum)
    
    min_L_sand = get_L_stable_sand(Hs, Tp, d, theta_w, s,
                                   Vc, z0, zr, OD, theta_c, spectrum)

    M = get_M(Vc, z0, zr, OD, theta_c, Hs, Tp, d, spectrum)

    if soil_type == 'Sand':
        gen_stab_w = min_L_clay
    elif soil_type == 'Clay':
        gen_stab_w = min_L_sand
    else:
        raise TypeError("Incorrect soil type")
    
    # To ensure the calculation is within the right bounds, use the sg
    # value calculated and the limits given in Section 3.5.2.
    sg = get_sg(Hs, Tp, d, theta_w, s, OD, wt, rho_p, rho_w, spectrum)

    return sg * 0.5 * OD * gen_stab_w * rho_w * M**2

def get_sg(Hs, Tp, d, theta_w, s, OD, wt, rho_p, rho_w, spectrum):
    """
        Calculate the specific weight of a pipe as per
        Equation 3.33 DNVGL-RP-F109. Only valid if the value
        produced is within a certain range.

        Input
        ------
        N = Spectral accerlation factor
        K = Significant Keulegan-Carpenter number
        L = Significant weight parameter

        Output
        ------
        sg_ok = specific weight of a pipe
    """

    # These factors come from Section 1 of DNVGL-RP-F109
    N = get_N(Hs, Tp, d, spectrum, theta_w, s)
    K = get_K(Hs, Tp, d, spectrum, theta_w, s, OD)
    L = get_L(OD, wt, rho_p, rho_w, Hs, Tp, d, spectrum)

    sg = 1 + 2 / np.pi * N * K * L

    if sg <=1.05:
        print("Sg too low")
        fact = 0
    elif sg > 3:
        print("Sg too high")
        fact = 0
    else:
        fact = 1
    # Equation 3.33

    return fact * sg


def get_L_10_clay(su, OD, gamma, Hs, Tp, d, spectrum,
                  theta_w, s, Vc, z0, zr, theta_c, d50):

    """
        To get the parameter required to calculate the required weight
        of pipe when the seabed is clay, a number of Tables in DNVGL-RP-F109
        need to be maniuplated and the correct value of L_10 extracted based 
        upon Gc, N and M. 
    """

    Gc = dnv.get_Gc(su, OD, gamma)
    N = get_N(Hs, Tp, d, spectrum, theta_w, s)
    K = get_K(Hs, Tp, d, spectrum, theta_w, s, OD)
    M = get_M(Vc, z0, zr, OD, theta_c, Hs, Tp, d, spectrum)
    K_b = 2.5 * d50
    Gc_val = min(Gc_values, key=lambda x:abs(x-Gc))
    x = Gc_values.index(Gc_val)

    if N <= 0.003:
        y = 1
    elif 0.003 < N <= 0.006:
        y = 3
    else:
        y = 2

    d = (str(x) + str(y))

    if y == 1 or y == 2:
        array_set = values[d]
        m_value = array_set[:, 0]
        y_value = lcp._get_closest(m_value, M)
        C1 = array_set[y_value, 1]
        C2 = array_set[y_value, 2]
        C3 = array_set[y_value, 3]
        K_b = array_set[y_value, 4]
    elif y == 3:
        d1 = (str(x) + str(1))
        d2 = (str(x) + str(2))
        array_1 = values[d1]
        array_2 = values[d2]
        m_value_1 = array_1[:, 0]
        m_value_2 = array_2[:, 0]
        y_val_1 = lcp._get_closest(m_value_1, M)
        y_val_2 = lcp._get_closest(m_value_2, M)
        C1_1 = np.array([array_1[y_val_1, 1], array_2[y_val_2, 1]])
        C2_1 = np.array([array_1[y_val_1, 2], array_2[y_val_2, 2]])
        C3_1 = np.array([array_1[y_val_1, 3], array_2[y_val_2, 3]])
        K_b1 = np.array([array_1[y_val_1, 4], array_2[y_val_2, 4]])
        N_val = np.array([0.003, 0.006])
        Z = np.array([0,1])
        C1 = interp2d(C1_1, N_val, Z, kind='linear')
        C2 = interp2d(C2_1, N_val, Z, kind='linear')
        C3 = interp2d(C3_1, N_val, Z, kind='linear')
        K_b = interp2d(K_b1, N_val, Z, kind='linear')

    if K >= K_b:
        L_10_clay = (C1 + C2 / K**C3) * (2 + M)**2
    else:
        L_10_clay = (C1 + C2 / K_b**C3) * (2 + M)**2

    return L_10_clay


def get_L10_sand(Hs, Tp, d, spectrum, theta_w, s, OD, 
                 Vc, z0, zr, theta_c):

    """
        To calculate the parameter required for limiting the 
        lateral displacement of the pipe to 10 pipe diameters is 
        taken from Table 3-4. 
    """

    K = get_K(Hs, Tp, d, spectrum, theta_w, s, OD)
    M = get_M(Vc, z0, zr, OD, theta_c, Hs, Tp, d, spectrum)

    # Extract the required value from the array taken from
    # Table 3-4 in DNVGL-RP-F109
    k_value = L_10_sand[0, :]
    m_value = L_10_sand[:, 0]
    x_val = lcp._get_closest(k_value, K)
    y_val = lcp._get_closest(m_value, M)

    return L_10_sand[y_val, x_val] / (2 + M)**2

def get_L_stable_clay(Hs, Tp, d, theta_w, s, 
                      Vc, z0, zr, theta_c, 
                      su, gammas, OD, Fc, spectrum):
    """
        The minimum pipe weight parameter required to keep the 
        pipe stable on a clayey is calculated using 
        Equation 3.36 from DNVGL-RP-F109

        Input
        ------
        Gc = soil (clay) strength parameter
        N = Spectral accerlation factor
        K = Significant Keulegan-Carpenter number
        M = steady to oscillatory velocity ratio for design spectrum

        Output
        ------
        L_stable_clay = minimum pipe weight required to keep pipe stable
    """

    N = get_N(Hs, Tp, d, spectrum, theta_w, s)
    K = get_K(Hs, Tp, d, spectrum, theta_w, s, OD)
    f_M = get_f_M(Vc, z0, zr, OD, theta_c, Hs, Tp, d, spectrum)

    L_stable_clay = 90 * sqrt((dnv.get_Gc(su, OD, Fc)) / N**0.67 * K) * f_M 

    return L_stable_clay


def get_L_stable_sand(Hs, Tp, d, theta_w, s,
                      Vc, z0, zr, OD, theta_c, spectrum):
    """
        The minimum pipe weight parameter needed to keep a pipe
        stable on the seabed is calculated using Tables 3-2 and 3-3
        in DNVGL-RP-F109. 

    """

    N = get_N(Hs, Tp, d, spectrum, theta_w, s)
    K = get_K(Hs, Tp, d, spectrum, theta_w, s, OD)
    M = get_M(Vc, z0, zr, OD, theta_c, Hs, Tp, d, spectrum)

    k_val_10 = K_10_array[0,:]
    m_val_10 = K_10_array[:,0]
    x_val_10 = lcp._get_closest(k_val_10, K)
    y_val_10 = lcp._get_closest(m_val_10, M)
    n_val_5 = K_5_array[0,:]
    m_val_5 = K_5_array[:,0]
    x_val_5 = lcp._get_closest(n_val_5, N)
    y_val_5 = lcp._get_closest(m_val_5, M) 

    if K > 10:
        L_stable_sand = K_10_array[y_val_10, x_val_10]

    elif 5 <= K < 10:
        # Attempted linear interpolation, will need to see if it works
        L_10 = K_10_array[y_val_10, x_val_10]
        L_5 = K_5_array[y_val_5, x_val_5]
        L_stable_sand = interp2d(L_5, L_10, 5, kind='linear')
    else:
        L_stable_sand = K_5_array[y_val_5, x_val_5]

    return L_stable_sand

def get_f_M(Vc, z0, zr, OD, theta_c, Hs, Tp, d, spectrum):

    M = get_M(Vc, z0, zr, OD, theta_c, Hs, Tp, d, spectrum) 

    f_M = ((0.58 * np.log(M))**2 + 0.6 * np.log(M) + 0.47)

    if f_M > 1.0:
        f_M_end = 0
    else:
        f_M_end = f_M

    return f_M_end

def get_M(Vc, z0, zr, OD, theta_c, Hs, Tp, d, spectrum):

    U_s = 2 * sqrt(w.get_M(0, Hs, Tp, d, spectrum))

    return c.Vc(Vc, z0, zr, OD, theta_c) / U_s

def get_K(Hs, Tp, d, spectrum, theta_w, s, OD):

    U_s = 2 * sqrt(w.get_M(0, Hs, Tp, d, spectrum))
    T_u = w.get_flow(Hs, Tp, d, theta_w, s, spectrum)[1]

    return U_s * T_u / OD

def get_L(OD, wt, rho_p, rho_w, Hs, Tp, d, spectrum):

    w_s = br._get_submerged_weight(OD, wt, rho_p, rho_w) # may need to have concrete wall thickness in there as well, need seperate equation?
    U_s = 2 * sqrt(w.get_M(0, Hs, Tp, d, spectrum))

    return w_s / (0.5 * rho_w * OD * U_s**2)

def get_N(Hs, Tp, d, spectrum, theta_w, s):

    U_s = 2 * sqrt(w.get_M(0, Hs, Tp, d, spectrum))
    T_u = w.get_flow(Hs, Tp, d, theta_w, s, spectrum)[1]

    return U_s / (g * T_u)


def absolute_stability(rho_w, rho_p, OD, mu, gammas, su, soil_type, location, 
                        grade, wt, Hs, Tp, depth, s, T_ss, kv, Fc, theta_t, 
                        z_t, Vc, z0, zr, theta_c, theta_w, Fz, spectrum):
    """
        Function to define whether the forces applied to the pipe
        exceed the absolute stability criteria. 
    """
    gamma_ysc = get_gamma_ysc(location, soil_type, grade)
    
    F_y = get_Fy(rho_w, OD, Hs, Tp, depth, T_ss, kv, 
                 Vc, z0, zr, theta_c, soil_type, 
                 gammas, su, Fc, theta_t, z_t, 
                 theta_w, s, spectrum)
    F_z = get_Fz(rho_w, OD, Hs, Tp, depth, T_ss, 
                 kv, Vc, z0, zr, theta_c, soil_type, 
                 gammas, su, Fc, theta_t, z_t, 
                 theta_w, s, spectrum)

    if soil_type == 'Sand':
        F_r = dnv.get_FR_sand(rho_w, Fz, OD, gammas)
    elif soil_type == 'Clay':
        F_r = dnv.get_FR_clay(rho_w, Fz, OD, gammas, su)
    else:
        raise TypeError('Soil type is invalid')

    w_s = br._get_submerged_weight(OD, wt, rho_p, rho_w)
    # Equation 3.38
    stab = gamma_ysc * (F_y + mu * F_z) / (mu * w_s * F_r)

    # Equation 3.39
    stab_2 = gamma_ysc * (F_z / w_s)

    return (stab, stab_2)

def get_gamma_ysc(location, soil_type, grade):
    """
        To get the safety factor needed, the soil type and location of the 
        underlying soil will required to be entered. This will then work through
        a dictionary located in a JSON to get the correct value of gamma_ysc. 

        Input
        ------
        soil type = type of soil, and grade
        location = where the pipeline is located

        Output
        ------
        gamma_ysc = safety factor for stability
    """
    if not location in data.keys():
        raise KeyError('Location is invalid')
    
    if not soil_type in data[location].keys():
        raise KeyError('Soil type is invalid')

    if not grade in data[location][soil_type].keys():
        raise KeyError('Grade is invalid')

    # Take gamma from Tables 3-5 to 3-8.

    return data[location][soil_type][grade]

def get_Fy(rho_w, OD, Hs, Tp, depth, T_ss, kv,
           Vc, z0, zr, theta_c, soil_type, 
           gammas, su, Fc, theta_t, z_t, 
           theta_w, s, spectrum):
    """
        Calculate the peak horizontal load on the pipe

        Input
        -----
        r_toty =
        rho_w = density of seawater
        OD = pipe outside diameter
        Cy = peak load coefficient
        U_star = oscillatory velocity amplitude for single
            design oscillation, perpendicular to pipeline
        V_star = steady current velocity associated with design
            oscillation, perpendicular to pipe

        Output 
        ------
        Fy = peak horizontal force
    """
    V_star = c.get_V_star(Vc, z0, zr, OD, theta_c, kv)
    U_star = w.get_U_star(Hs, Tp, depth, T_ss, theta_w, s, spectrum)

    r_tot_y = dnv.get_r_tot(soil_type, OD, gammas, su, Fc, theta_t, z_t)[0]

    Cy = get_Cy(Hs, Tp, depth, OD, T_ss, Vc, z0, zr, theta_c, kv, theta_w, s, spectrum)

    # Equation 3.40
    return 0.5 * r_tot_y * rho_w * OD * Cy * (U_star + V_star)**2

def get_Fz(rho_w, OD, Hs, Tp, depth, T_ss, kv, 
           Vc, z0, zr, theta_c, soil_type, 
           gammas, su, Fc, theta_t, z_t, 
           theta_w, s, spectrum):
    """
        Calculate the peak vertical load on the pipe.

        Input
        ------
        r_totz = 
        rho_w = density of seawater
        OD = pipe outside diameter
        Cz = vertical peak load cofficient
        U_star = oscillatory velocity amplitude for single
            design oscillation, perpendicular to pipeline
        V_star = steady current velocity associated with design
            oscillation, perpendicular to pipe

        Ouput 
        ------
        F_z = peak vertical load
    """
    V_star = c.get_V_star(Vc, z0, zr, OD, theta_c, kv)
    U_star = w.get_U_star(Hs, Tp, depth, T_ss, theta_w, s, spectrum)
    Cz = get_Cz(Hs, Tp, depth, OD, T_ss, Vc, z0, zr, theta_c, kv, theta_w, s, spectrum)

    r_tot_z = dnv.get_r_tot(soil_type, OD, gammas, su, Fc, theta_t, z_t)[1]

    # Equation 3.41
    return 0.5 * r_tot_z * rho_w * OD * Cz * (U_star + V_star)**2

def get_Cy(Hs, Tp, d, OD, T_ss, Vc, z0, zr, theta_c, kv, theta_w, s, spectrum):
    """
        To get the peak horizontal load coefficient, a tabulated set 
        of data which depends on M_star and K_star is accessed and used. 

        Input
        ------
        M_star = steady to oscillartory velcoity ratio for design spectrum
        K_star = Keulegan-Carpenter number

        Output
        ------
        Cy = peak horizontal load coefficient
    """

    K_star = get_K_star(Hs, Tp, d, T_ss, OD, theta_w, s, spectrum)
    M_star = get_M_star(Hs, Tp, d, T_ss, Vc, z0, zr, OD, theta_c, kv, theta_w, s, spectrum)

    K_star_values = Cy_array[0, :]
    M_star_values = Cy_array[:, 0]

    # For now I have just used the get closest value instead of interpolating.
    # Further development may be looking at making it so that this is interpolated.
    x_val = lcp._get_closest(K_star_values, K_star)
    y_val = lcp._get_closest(M_star_values, M_star)

    if K_star < 2.5:
        Cy = Cy_array[1, y_val] * 2.5 / K_star
    elif K_star >= 140:
        Cy = Cy_array[11, y_val]
    else:
        Cy = Cy_array[y_val, x_val]

    return Cy 

def get_Cz(Hs, Tp, d, OD, T_ss, Vc, z0, zr, theta_c, kv, theta_w, s, spectrum):
    """
        To get the peak vertical load coefficient, a tabulated set 
        of data which depends on M_star and K_star is accessed and used. 

        Input
        ------
        M_star = steady to oscillartory velcoity ratio for design spectrum
        K_star = Keulegan-Carpenter number

        Output
        ------
        Cz = peak vertical load coefficient
    """
    K_star = get_K_star(Hs, Tp, d, T_ss, OD, theta_w, s, spectrum)
    M_star = get_M_star(Hs, Tp, d, T_ss, Vc, z0, zr, OD, theta_c, kv, theta_w, s, spectrum)

    K_star_values = Cz_array[0, :]
    M_star_values = Cz_array[:, 0]

    # For now I have just used the get closest value instead of interpolating.
    # Further development may be looking at making it so that this is interpolated.
    x_val = lcp._get_closest(K_star_values, K_star)
    y_val = lcp._get_closest(M_star_values, M_star)

    if K_star <= 2.5:
        Cz = Cz_array[1, y_val]
    elif K_star >= 140:
        Cz = Cz_array[11, y_val]
    else:
        Cz = Cz_array[y_val, x_val]

    return Cz

def get_K_star(Hs, Tp, d, T_ss, OD, theta_w, s, spectrum):
    """
        Calculate the Keulegan-Carpenter number for single design oscillation.

        Input
        ------
        U_star = oscillatory velocity amplitude for single design oscillation
        T_star = period associated with single design oscillation
        OD = pipe outside diameter

        Output
        ------
        K_star = Keulegan-Carpenter number for single design oscillation
    """

    U_star = w.get_U_star(Hs, Tp, d, T_ss, theta_w, s, spectrum)
    T_star = get_k_T(Hs, Tp, d, theta_w, s, spectrum) * w.get_flow(Hs, Tp, d, theta_w, s, spectrum)[1]

    return U_star * T_star / OD

def get_M_star(Hs, Tp, depth, T_ss, Vc, z0, zr, OD, theta_c, kv, theta_w, s, spectrum):
    """
        Calculate the steady to oscillatory velocity ratio for 
        single design oscillation

        Input
        ------
        U_star = oscillatory velocity amplitude for single design oscillation
        V_star = steady current velocity associated with design oscillation

        Output
        ------
        M_star = steady to oscillatory velocity ratio for single design oscillation
    """      
    U_star = w.get_U_star(Hs, Tp, depth, T_ss, theta_w, s, spectrum)
    V_star = c.get_V_star(Vc, z0, zr, OD, theta_c, kv)

    return V_star / U_star

def get_k_T(Hs, Tp, depth, theta_w, s, spectrum):
    """
        Calculation to get ratio between design single oscillation velocity period
        and the average zero-up crossing period. 

        Input
        ------
        T_u = average zero-up crossing period
        Tn = reference period
        gamma = non-dimensional peak shape parameter
        
        Output
        ------
        k_T = ratio between single design oscillation velocity period and average
              zero-up crossing period
    """

    T_u = w.get_flow(Hs, Tp, depth, theta_w, s, spectrum)[1]
    Tn = sqrt(depth / g)
    gamma = w.get_gamma(Hs, Tp)

    # Calculate k_t from Equation 3.16
    # Values for k_t have only been given for three values of gamma 
    # so assumed midpoints go to these values.
    if gamma < 2.15:
        k_t = 1.25
    elif 2.15 <= gamma < 4.15:
        k_t = 1.21
    else:
        k_t = 1.17

    # Equation 3.16
    if Tn / T_u <= 0.2:
        k_T = k_t - 5 * (k_t - 1) * Tn / T_u
    else:
        k_T = 1

    return k_T

data = {"North Sea winter": {"Sand and rock": {"Low": 0.98,
                                        "Normal": 1.32,
                                        "High": 1.67},
                    "Clay": {"Low": 1.00,
                                "Normal": 1.40,
                                "High": 1.83}},
"North Sea cyclonic": {"Sand and rock": {"Low": 0.95,
                                          "Normal": 1.50,
                                          "High": 2.16},
                        "Clay": {"Low": 0.95,
                                 "Normal": 1.56,
                                 "High": 2.31}},
"Gulf winter": {"Sand and rock": {"Low": 0.95,
                                  "Normal": 1.41,
                                  "High": 1.99},
                "Clay": {"Low": 0.97,
                         "Normal": 1.50,
                         "High": 2.16}},
"Gulf cyclonic": {"Sand and rock": {"Low": 0.95,
                                     "Normal": 1.64,
                                     "High": 2.46},
                    "Clay": {"Low": 0.93,
                             "Normal": 1.64,
                             "High": 2.54}}
}

Cy_array = np.array([[0,   2.5,   5,    10,    20,    30,    40,   50,    60,    70,   100,  140],
                     [0,   13.0,  6.8,  4.55,  3.33,  2.72,  2.4,  2.15,  1.95,  1.8,  1.52, 1.3],
                     [0.1, 10.7,  5.76, 3.72,  2.72,  2.20,  1.90, 1.71,  1.58,  1.49, 1.33, 1.22],
                     [0.2, 9.02,  5.00, 3.15,  2.30,  1.85,  1.58, 1.42,  1.33,  1.27,  1.18, 1.14],
                     [0.3, 7.64,  4.32, 2.79,  2.01,  1.63,  1.44, 1.33,  1.26,  1.21,  1.14, 1.09],
                     [0.4, 6.63,  3.80, 2.51,  1.78,  1.46,  1.32, 1.25,  1.19,  1.16,  1.10, 1.05],
                     [0.6, 5.07,  3.30, 2.27,  1.71,  1.43,  1.34, 1.29,  1.24,  1.18,  1.08, 1.00],
                     [0.8, 4.01,  2.70, 2.01,  1.57,  1.44,  1.37, 1.31,  1.24,  1.17,  1.05, 1.00],
                     [1.0, 3.25,  2.30, 1.75,  1.49,  1.40,  1.34, 1.27,  1.20,  1.13,  1.01, 1.00],
                     [2.0, 1.52,  1.50, 1.45,  1.39,  1.34,  1.20, 1.08,  1.03,  1.00,  1.00, 1.00],
                     [5.0, 1.11,  1.10, 1.07,  1.06,  1.04,  1.01, 1.00,  1.00,  1.00,  1.00, 1.00],
                     [10,  1.00,  1.00, 1.00,  1.00,  1.00,  1.00, 1.00,  1.00,  1.00,  1.00, 1.00]])

Cz_array = np.array([[0,   2.5,  5,    10,   20,   30,   40,   50,   60,   70,   100,  140],
                     [0.0, 5.00, 5.00, 4.85, 3.21, 2.55, 2.26, 2.01, 1.81, 1.63, 1.26, 1.05],
                     [0.1, 3.87, 4.08, 4.23, 2.87, 2.15, 1.77, 1.55, 1.41, 1.31, 1.11, 0.97],
                     [0.2, 3.16, 3.45, 3.74, 2.60, 1.86, 1.45, 1.26, 1.16, 1.09, 1.00, 0.90],
                     [0.3, 3.01, 3.25, 3.53, 2.14, 1.52, 1.26, 1.10, 1.01, 0.99, 0.95, 0.90],
                     [0.4, 2.87, 3.08, 3.35, 1.82, 1.29, 1.11, 0.98, 0.90, 0.90, 0.90, 0.90],
                     [0.6, 2.21, 2.36, 2.59, 1.59, 1.20, 1.03, 0.92, 0.90, 0.90, 0.90, 0.90],
                     [0.8, 1.53, 1.61, 1.80, 1.18, 1.05, 0.97, 0.92, 0.90, 0.90, 0.90, 0.90],
                     [1.0, 1.05, 1.13, 1.28, 1.12, 0.99, 0.91, 0.90, 0.90, 0.90, 0.90, 0.90],
                     [2.0, 0.96, 1.03, 1.05, 1.00, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90],
                     [5.0, 0.91, 0.92, 0.93, 0.91, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90],
                     [10,  0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90]])

K_10_array = np.array([[0,   10,   15,   20,   30,   40,   60],
                       [0.2, 1.50, 1.42, 1.35, 1.25, 1.22, 1.22],
                       [0.4, 1.82, 1.70, 1.61, 1.53, 1.50, 1.50],
                       [0.5, 2.19, 1.97, 1.83, 1.69, 1.61, 1.61],
                       [0.6, 2.65, 2.35, 2.18, 1.99, 1.85, 1.72],
                       [0.8, 3.05, 2.55, 2.32, 2.13, 2.01, 1.90],
                       [1.0, 3.05, 2.55, 2.40, 2.20, 2.06, 1.95],
                       [1.5, 2.65, 2.45, 2.36, 2.24, 2.11, 2.09],
                       [2.0, 2.50, 2.40, 2.35, 2.27, 2.22, 2.19], 
                       [4.0, 2.45, 2.40, 2.39, 2.37, 2.37, 2.37],
                       [10, 2.50, 2.50, 2.50, 2.50, 2.50, 2.50]])

K_5_array = np.array([[0,   0.003, 0.006, 0.012, 0.024, 0.048],
                      [0.2, 1.55,  1.45,  1.34,  1.24,  1.13],
                      [0.4, 2.00,  1.65,  1.34,  1.24,  1.13],
                      [0.5, 3.30,  2.60,  1.91,  1.24,  1.13],
                      [0.6, 3.75,  3.07,  2.38,  1.70,  1.13],
                      [0.8, 4.00,  3.45,  2.90,  2.36,  1.81],
                      [1.0, 3.90,  3.50,  3.10,  2.71,  2.31], 
                      [1.5, 3.25,  3.13,  3.00,  2.88,  2.75],
                      [2.0, 2.75,  2.75,  2.75,  2.75,  2.75],
                      [4.0, 2.60,  2.60,  2.60,  2.60,  2.60],
                      [10,  2.50,  2.50,  2.50,  2.50,  2.50]])

L_10_sand = np.array([[0,   5,    10,   15,   20,   30,   40,   60,   100],
                      [0.2, 0.20, 0.41, 0.61, 0.81, 0.69, 0.69, 0.69, 0.69],
                      [0.4, 0.31, 0.62, 0.93, 0.81, 0.75, 0.72, 0.70, 0.70],
                      [0.5, 0.34, 0.69, 1.03, 0.93, 0.83, 0.78, 0.75, 1.00],
                      [0.6, 0.79, 1.20, 1.13, 1.10, 1.07, 1.05, 1.03, 1.02],
                      [0.8, 0.85, 1.40, 1.37, 1.35, 1.33, 1.33, 1.32, 1.31],
                      [1.0, 1.60, 1.50, 1.47, 1.45, 1.43, 1.43, 1.42, 1.41],
                      [1.5, 1.80, 1.70, 1.67, 1.65, 1.63, 1.63, 1.62, 1.61],
                      [2.0, 1.90, 1.80, 1.77, 1.75, 1.73, 1.73, 1.72, 1.71],
                      [4.0, 2.10, 2.00, 1.97, 1.95, 1.93, 1.93, 1.92, 1.91],
                      [10,  2.50, 2.50, 2.50, 2.50, 2.50, 2.50, 2.50, 2.50]])


values = {'01': np.array([[0.2, 0,   9, 0.6, 10],
                            [0.4, 0,   8, 0.6, 10],
                            [0.5, 0.1, 7, 0.6, 10],
                            [0.6, 0.1, 7, 0.6, 10],
                            [0.8, 0.1, 7, 0.6, 10],
                            [1.0, 0.4, 5, 0.6, 5],
                            [1.5, 0.4, 5, 0.6, 5],
                            [2.0, 0.7, 3, 0.6, 5],
                            [4.0, 1.4, 1, 0.6, 5]]),
            '02': np.array([[0.2, 0.2, 5, 0.5, 15], 
                            [0.4, 0.2, 5, 0.5, 15],
                            [0.5, 0.4, 4, 0.5, 15],
                            [0.6, 0.4, 4, 0.5, 15],
                            [0.8, 0.7, 3, 0.5, 15],
                            [1.0, 0.6, 3, 0.5, 15],
                            [1.5, 1.1, 2, 0.5, 15],
                            [2.0, 1.6, 0, 0.5, 15],
                            [4.0, 1.9, 0, 0.5, 15]]),
            '11': np.array([[0.2, 0.1, 9, 0.6, 10],
                           [0.4, 0.1, 8, 0.6, 10],
                           [0.5, 0.1, 8, 0.6, 10],
                           [0.6, 0.2, 8, 0.6, 10],
                           [0.8, 0.4, 7, 0.6, 5],
                           [1.0, 0.4, 7, 0.6, 5],
                           [1.5, 0.4, 5, 0.6, 5],
                           [2.0, 0.7, 3, 0.6, 5],
                           [4.0, 1.4, 1, 0.6, 5]]),
            '12': np.array([[0.2, 0.1, 7, 0.6, 10],
                           [0.4, 0.1, 7, 0.6, 10],
                           [0.5, 0.1, 7, 0.6, 10],
                           [0.6, 0.2, 6, 0.6, 10],
                           [0.8, 0.3, 6, 0.6, 10],
                           [1.0, 0.4, 6, 0.6, 10],
                           [1.5, 0.8, 4, 0.6, 10],
                           [2.0, 1.5, 0, 0.6, 10],
                           [4.0, 1.5, 0, 0.6, 10]]),
            '21': np.array([[0.2, 0.1, 8, 0.5, 15],
                           [0.4, 0.1, 7, 0.5, 10],
                           [0.5, 0.1, 7, 0.5, 10],
                           [0.6, 0.1, 7, 0.5, 10],
                           [0.8, 0.1, 7, 0.5, 5],
                           [1.0, 0.1, 7, 0.5, 5],
                           [1.5, 0.1, 7, 0.5, 5],
                           [2.0, 0.1, 7, 0.5, 5],
                           [4.0, 0.1, 7, 0.5, 5],
                           [10, 0.1, 7, 0.5, 5]]),
            '22': np.array([[0.2, 0.1, 8, 0.5, 10],
                           [0.4, -0.3, 8, 0.5, 10],
                           [0.5, -0.1, 7, 0.5, 10],
                           [0.6, 0.0, 7, 0.5, 10],
                           [0.8, 0.1, 6, 0.5, 5],
                           [1.0, 0.1, 6, 0.5, 5],
                           [1.5, 0.5, 3, 0.5, 5],
                           [2.0, 0.9, 2, 0.5, 5],
                           [4.0, 1.7, 0, 0.5, 5],
                           [10, 1.7, 0, 0.5, 5]]),
            '31': np.array([[0.2, 1.4, 3, 0.5, 15],
                           [0.4, 0.5, 6, 0.5, 5],
                           [0.5, 0.5, 6, 0.5, 5],
                           [0.6, 0.5, 6, 0.5, 5],
                           [0.8, 1.1, 4, 0.5, 5],
                           [1.0, 1.3, 4, 0.5, 10],
                           [1.5, 1.2, 7, 0.5, 10],
                           [2.0, 1.2, 7, 0.5, 10],
                           [4.0, 1.2, 7, 0.5, 10],
                           [10, 1.4, 6, 0.5, 10]]),
            '32': np.array([[0.2, 0.0, 8, 0.5, 10],
                           [0.4, 0.3, 6, 0.5, 5],
                           [0.5, 0.3, 6, 0.5, 5],
                           [0.6, 0.3, 6, 0.5, 5],
                           [0.8, 0.4, 7, 0.5, 5],
                           [1.0, 0.4, 7, 0.5, 5],
                           [1.5, 0.8, 6, 0.5, 10],
                           [2.0, 0.8, 6, 0.5, 10],
                           [4.0, 0.8, 6, 0.5, 10],
                           [10, 0.8, 6, 0.5, 10]]),
            '41': np.array([[0.2, 2.1, 1, 0.5, 15],
                          [0.4, 2.4, 2, 0.5, 15],
                          [0.5, 2.4, 2, 0.5, 15],
                          [0.6, 1.9, 6, 0.5, 15],
                          [0.8, 2.2, 8, 0.5, 15],
                          [1.0, 2.2, 8, 0.5, 15],
                          [1.5, 2.4, 8, 0.5, 15]]),
            '42': np.array([[0.2, 1.4, 4, 0.5, 15],
                          [0.4, 1.1, 7, 0.5, 15],
                          [0.5, 1.5, 5, 0.5, 15],
                          [0.6, 1.6, 5, 0.5, 15],
                          [0.8, 1.9, 6, 0.5, 15],
                          [1.0, 2.2, 6, 0.5, 15],
                          [1.5, 2.0, 8, 0.5, 15]]),
            '51': np.array([[0.2, 3.4, 1, 0.5, 20],
                          [0.4, 3.4, 1, 0.5, 20],
                          [0.5, 3.0, 4, 0.5, 20],
                          [0.6, 3.2, 6, 0.5, 15],
                          [0.8, 2.4, 12, 0.5, 15],
                          [1.0, 2.3, 12, 0.5, 15],
                          [1.5, 2.3, 12, 0.5, 15],
                          [2.0, 2.3, 12, 0.5, 15],
                          [4.0, 2.3, 12, 0.5, 15]]),
            '52': np.array([[0.2, 2.7, 3, 0.5, 20],
                          [0.4, 2.4, 4, 0.5, 20],
                          [0.5, 2.2, 7, 0.5, 20],
                          [0.6, 1.9, 9, 0.5, 15],
                          [0.8, 1.9, 12, 0.5, 15],
                          [1.0, 1.5, 14, 0.5, 15],
                          [1.5, 1.5, 14, 0.5, 15],
                          [2.0, 1.5, 14, 0.5, 15],
                          [4.0, 1.5, 14, 0.5, 15]])}

Gc_values = [0.0556, 0.111, 0.222, 0.556, 1.11, 2.78]


if __name__ == '__main__':
    print(overall_stability(1025, 7850, 0.4572, 0.6, 19000, 15000, 'Clay', 'North Sea cyclonic',
                            'Low', 0.05, 10.1, 15.8, 50, 5, 10000, 2, 1215, 30, 1,
                            0.45, 0.00004, 3, 35, 90, 0, spectrum='jonswap'))

