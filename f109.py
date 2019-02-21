import sys
import os
sys.path.insert(0, os.getcwd())

import src.soil.dnv_soil as dnv
import src.hydrodynamics.waves as w
import src.hydrodynamics.current as c
import numpy as np
from math import sqrt
from scipy.constants import g
from scipy.interpolate import interp1d
from scipy.optimize import minimize

def get_wall_thick(Hs, Tp, d, theta_w, mu, s, Vc, rho_s, rho_cont, rho_conc, 
                    z0, zr, theta_c, su, gammas, OD, rho_m, rho_coat, t_corr,
                    t_conc, t_m, t_coat, soil_type, rho_sw, Fc, wt, location, 
                    grade, T_ss, kv, theta_t, z_t, Fz, spectrum):

    w_virt = virtual_stability(Hs, Tp, d, theta_w, s, Vc, rho_s, rho_cont, rho_conc, 
                      z0, zr, theta_c, su, gammas, OD, rho_m, rho_coat, t_corr,
                      t_conc, t_m, soil_type, rho_sw, Fc, wt, spectrum)

    w_abs = absolute_stability(rho_sw, rho_s, rho_cont, OD, mu, gammas, su, soil_type, location, 
                        grade, wt, Hs, Tp, d, s, T_ss, kv, Fc, theta_t, 
                        z_t, Vc, z0, zr, theta_c, theta_w, Fz, rho_conc, rho_m,
                        rho_coat, t_corr, t_conc, t_m, t_coat, spectrum)

    w_sub =  get_wsub(OD, wt, t_corr, t_conc, t_m, t_coat, rho_s, rho_cont,
             rho_conc, rho_m, rho_coat, rho_sw)

    x_0 = OD / 20

    def f_virt(wt):

        return (w_virt / w_sub) - 0.99

    w_virt = minimize(f_virt, x_0)

    def f_abs_1(wt):

        return (w_abs[0] / w_sub) - 0.99
    
    w_abs_1 = minimize(f_abs_1, x_0)

    def f_abs_2(wt):

        return (w_abs[1] / w_sub) - 0.99

    w_abs_2 = minimize(f_abs_2, x_0)

    wt_stab = np.max(w_virt, w_abs_1, w_abs_2)

    return wt_stab


def general_stability(Hs, Tp, d, theta_w, s, OD, Vc,
                          z0, zr, theta_c, soil_type, rho_w, su,
                          gamma, t_corr, t_conc, t_m, spectrum):
    """
        General stability is determined by calculating the minimum
        pipe weight that is required to minimise movement whilst on
        the seabed. This is then compared to the known weight of the pipe
        and a stability assessment made. 
        
        This calculation is to do with the minimum pipe weight required to 
        keep the pipe stable to a lateral displacement of 10 x the diameter
        of the pipe. 
    """
    U_s = get_U_s(Hs, Tp, d, spectrum)
    T_u = get_T_u(Hs, Tp, d, spectrum, theta_w, s)
    Gc = dnv.get_Gc(su, OD, gamma)
    N = get_N(U_s, T_u)
    K = get_K(U_s, T_u, OD)
    M = get_M(Vc, z0, zr, OD, theta_c, U_s)
    D_o = get_D_o(OD, t_corr, t_conc, t_m)

    if soil_type == 'Sand':
        stab_L = get_L10_sand(K, M)
    elif soil_type == 'Clay':
        stab_L = get_L_10_clay(Gc, N, K, M)
    else:
        raise TypeError("Wrong soil type entered")

    return stab_L * 0.5 * rho_w * D_o * U_s**2


def virtual_stability(Hs, Tp, d, theta_w, s, Vc, rho_s, rho_cont, rho_conc, 
                      z0, zr, theta_c, su, gamma, OD, rho_m, rho_coat, t_corr,
                      t_conc, t_m, soil_type, rho_sw, Fc, wt, spectrum):
    """
        Virtual stability is determined by calculating the minimum
        pipe weight that is required to keep the pipe stable whilst on
        the seabed. This is then compared to the known weight of the pipe
        and a stability assessment made. 
        
        This calculation is to do with the minimum pipe weight required to 
        keep the pipe stable to a lateral displacement of 0.5 x the diameter
        of the pipe. 
 
    """
    U_s = get_U_s(Hs, Tp, d, spectrum)
    T_u = get_T_u(Hs, Tp, d, spectrum, theta_w, s)
    Gc = dnv.get_Gc(su, OD, gamma)
    N = get_N(U_s, T_u)
    K = get_K(U_s, T_u, OD)
    M = get_M(Vc, z0, zr, OD, theta_c, U_s)
    f_M = get_f_M(M)

    # Use previous calculations to get the factor used to calculate the
    # minimum pipe weight required for stability. 
    min_L_clay = get_L_stable_clay(Gc, N, K, f_M)
    min_L_sand = get_L_stable_sand(N, K, M)
    D_o = get_D_o(OD, t_corr, t_conc, t_m)
    

    if soil_type == 'Sand':
        virt_stab = min_L_clay
    elif soil_type == 'Clay':
        virt_stab = min_L_sand
    else:
        raise TypeError("Incorrect soil type")

    return 0.5 * virt_stab * rho_sw * D_o * U_s**2

def get_sg(Hs, Tp, d, theta_w, s, OD, wt, rho_s, rho_sw, 
           rho_cont, rho_conc, rho_m, rho_coat, t_corr, 
           t_conc, t_m, t_coat, spectrum):
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
    U_s = get_U_s(Hs, Tp, d, spectrum)
    T_u = get_T_u(Hs, Tp, d, spectrum, theta_w, s)
    w_s = get_wsub(OD, wt, t_corr, t_conc, t_m, t_coat, rho_s, rho_cont,
             rho_conc, rho_m, rho_coat, rho_sw)
    N = get_N(U_s, T_u)
    K = get_K(U_s, T_u, OD)
    L = get_L(U_s, w_s, OD, rho_sw)

    sg = 1 + (2 / np.pi) * N * K * L

    if sg <=1.05:
        print("Sg too low")
        fact = 0
    elif sg > 3:
        print("Sg too high")
        fact = 0
    else:
        fact = 1

    return fact * sg


def get_L_10_clay(Gc, N, K, M):

    """
        To get the parameter required to calculate the required weight
        of pipe when the seabed is clay, a number of Tables in DNVGL-RP-F109
        need to be maniuplated and the correct value of L_10 extracted based 
        upon Gc, N and M. 
    """

    C1, C2, C3, Kb = get_L10_params(Gc, N, M)

    if K >= Kb:
        L_10_clay = (C1 + C2 / K**C3) * (2 + M)**2
    else:
        L_10_clay = (C1 + C2 / Kb**C3) * (2 + M)**2

    return L_10_clay

def get_L10_params(Gc, N, M):

    C1 = interp1d(_M, interp1d(_N, interp1d(_Gc, _C1)(Gc))(N))(M)
    C2 = interp1d(_M, interp1d(_N, interp1d(_Gc, _C2)(Gc))(N))(M)
    C3 = interp1d(_M, interp1d(_N, interp1d(_Gc, _C3)(Gc))(N))(M)
    Kb = interp1d(_M, interp1d(_N, interp1d(_Gc, _Kb)(Gc))(N))(M)

    return C1, C2, C3, Kb

def get_L10_sand(K, M):

    """
        To calculate the parameter required for limiting the 
        lateral displacement of the pipe to 10 pipe diameters is 
        taken from Table 3-4. 
    """

    # Extract the required value from the array taken from
    # Table 3-4 in DNVGL-RP-F109
    k_value = L_10_sand[0, :]
    m_value = L_10_sand[:, 0]
    x_val = _get_closest(k_value, K)
    y_val = _get_closest(m_value, M)

    return L_10_sand[y_val, x_val] * (2 + M)**2

def get_L_stable_clay(Gc, N, K, f_M):
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

    L_stable_clay = 90 * sqrt(Gc / N**0.67 * K) * f_M 

    return L_stable_clay


def get_L_stable_sand(N, K, M):
    """
        The minimum pipe weight parameter needed to keep a pipe
        stable on the seabed is calculated using Tables 3-2 and 3-3
        in DNVGL-RP-F109. 

    """

    k_val_10 = K_10_array[0,:]
    m_val_10 = K_10_array[:,0]
    x_val_10 = _get_closest(k_val_10, K)
    y_val_10 = _get_closest(m_val_10, M)
    n_val_5 = K_5_array[0,:]
    m_val_5 = K_5_array[:,0]
    x_val_5 = _get_closest(n_val_5, N)
    y_val_5 = _get_closest(m_val_5, M) 

    if K > 10:
        L_stable_sand = K_10_array[y_val_10, x_val_10]

    elif 5 <= K < 10:
        L_10 = K_10_array[y_val_10, x_val_10]
        L_5 = K_5_array[y_val_5, x_val_5]
        L_stable_sand = ((L_10 - L_5) / 5) * K + L_5
    elif K < 5:
        L_stable_sand = K_5_array[y_val_5, x_val_5]

    return L_stable_sand * (2 + M)**2

def get_f_M(M):

    """
        Function of M used in the calculation of the specific
        weight parameter, sg. 
    """

    f_M = ((0.58 * np.log(M))**2 + 0.6 * np.log(M) + 0.47)

    if f_M > 1.0:
        f_M_end = 0
    else:
        f_M_end = f_M

    return f_M_end

def get_M(Vc, z0, zr, OD, theta_c, U_s):

    """
        Steady to oscillatory velocity ratio for design 
        spectrum. 
    """

    return c.Vc(Vc, z0, zr, OD, theta_c) / U_s


def get_K(U_s, T_u, OD):

    """
        Significant Keulegan-Carpenter number
    """

    return U_s * T_u / OD

def get_L(U_s, w_s, OD, rho_sw):

    """
        Significant weight parameter 
    """

    return w_s / (0.5 * rho_sw * OD * U_s**2)

def get_N(U_s, T_u):

    """
        Spectral acceleration factor
    """

    return U_s / (g * T_u)

def get_U_s(Hs, Tp, d, spectrum):

    return 2 * sqrt(w.get_M(0, Hs, Tp, d, spectrum))

def get_T_u(Hs, Tp, d, spectrum, theta_w, s):

    return w.get_flow(Hs, Tp, d, theta_w, s, spectrum)[1]

def absolute_stability(rho_w, rho_s, rho_cont, OD, mu, gammas, su, soil_type, location, 
                        grade, wt, Hs, Tp, depth, s, T_ss, kv, Fc, theta_t, 
                        z_t, Vc, z0, zr, theta_c, theta_w, Fz, rho_conc, rho_m,
                        rho_coat, t_corr, t_conc, t_m, t_coat, spectrum):
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

    w_s = get_wsub(OD, wt, t_corr, t_conc, t_m, t_coat, rho_s, rho_cont,
             rho_conc, rho_m, rho_coat, rho_w)
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
    x_val = _get_closest(K_star_values, K_star)
    y_val = _get_closest(M_star_values, M_star)

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
    x_val = _get_closest(K_star_values, K_star)
    y_val = _get_closest(M_star_values, M_star)

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

def get_wsub(OD, t, t_corr, t_conc, t_m, t_coat, rho_s, rho_cont,
             rho_conc, rho_m, rho_coat, rho_sw):
    """
        Function to get the submerged weight of the pipe

        Inputs
        ------
        OD = pipe outside diameter
        t = wall thickness
        t_corr = corrosion coating thickness
        t_conc = concrete thickness
        t_m = marine growth thickness
        t_coat = coating thickness
        rho_s = steel density
        rho_cont = contents density
        rho_conc = concrete density
        rho_m = marine growth density
        rho_coat = coating density
        rho_sw = seawater density

        Outputs
        ------
        w_sub = submerged weight of pipe
        buoy = buoyancy of the pipe
        D_o = overall outside diameter of pipe

    """

    D_o = get_D_o(OD, t_corr, t_conc, t_m)
    buoy = buoyancy(OD, t_corr, t_conc, t_m, rho_sw)
    ID = OD - 2 * t

    w_steel = layer_weight(OD, rho_s, ID)
    w_cont = layer_weight(ID, rho_cont)
    w_conc = layer_weight((D_o - 2 * t_m), rho_conc, (OD + 2 * t_corr))
    w_m = layer_weight(D_o, rho_m, (D_o - 2 * t_m))
    w_coat = layer_weight(ID, rho_coat, (ID - 2 * t_coat))

    return (w_steel + w_cont + w_conc + w_m + w_coat) - buoy

def layer_weight(OD, rho, ID=0):
    """
        Generic function to calculate layer weight
        dependent on the diameters and density of material.

        Inputs
        ------
        OD = outside diameter
        ID = inside diameter
        rho = density of material

        Outputs
        ------
        w = layer weight
    """
    return get_area(OD, ID) * rho

def get_area(OD, ID=0):
    """
        Generic function to calculate area of pipe based on outer and 
        internal diameter.

        Inputs
        ------
        OD = outer diameter
        ID = inner diameter

        Outputs
        ------
        A = area of pipe
    """
    return np.pi * (OD**2 - ID**2) / 4

def get_D_o(OD, t_corr, t_conc, t_m):
    """
        Function to calculate the full outside diameter of 
        the pipe with all coatings etc considered.

        Inputs
        ------
        OD = outside diameter of pipe
        t_corr = corrosion protection thickness
        t_conc = concrete thickness
        t_m = marine growth thickness

        Outputs
        ------
        D_o = overall outside diameter
    """
    return OD + 2 * t_corr + 2 * t_conc + 2 * t_m

def buoyancy(OD, t_corr, t_conc, t_m, rho_sw):
    """
        Calculate submerged buoyancy weight of pipe. 

        Inputs
        -------
        OD = pipe outside diameter
        t_corr = corrosion protection thickness
        t_conc = concrete thickness
        t_m = marine growth thickness
        rho_sw = seawater density

        Outputs
        ------
        buoy = calculated submerged buoyancy of pipe
    """
    D_o = get_D_o(OD, t_corr, t_conc, t_m)

    return np.pi / 4 * D_o**2 * rho_sw

def _get_closest(array, values):
    """ Helper function to find the closest cell to a point.
    Mainly to be used to 

    Inputs
    ------
    array: array of values to be searched
    values: list of values to search for index

    Returns
    -------
    NumPy array of indices corresponding to each item in values
    """
    if isinstance(values, (int, float)):
        values = np.array([values])
    if not isinstance(values, np.ndarray):
        raise TypeError('values should be a numeric type or a numpy array')
    closest = np.empty(values.shape, dtype=int)
    for i in range(values.size):
        closest[i] = (np.abs(array - values[i])).argmin()
    if closest.shape == (1,):
        closest = closest[0]
    return closest

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


_C1 = np.array([[# Gc = 0.0556
                 [0, 0.2, 0.2],
                 [0, 0.2, 0.2],
                 [0.1, 0.4, 0.4],
                 [0.1, 0.4, 0.4],
                 [0.1, 0.7, 0.7],
                 [0.4, 0.7, 0.7],
                 [0.4, 1.1, 1.1],
                 [0.7, 1.6, 1.6],
                 [1.4, 1.9, 1.9],
                 [1.4, 1.9, 1.9]],
                 [# Gc = 0.111
                 [0.1, 0.1, 0.1],
                 [0.1, 0.1, 0.1],
                 [0.1, 0.1, 0.1],
                 [0.2, 0.2, 0.2],
                 [0.4, 0.3, 0.3],
                 [0.4, 0.4, 0.4],
                 [0.4, 0.8, 0.8],
                 [0.7, 1.5, 1.5],
                 [1.4, 1.5, 1.5],
                 [1.4, 1.5, 1.5]],
                 [# Gc = 0.222
                 [0.1, 0.1, 0.1],
                 [0.1, -0.3, -0.3],
                 [0.1, -0.1, -0.1],
                 [0.1, 0, 0],
                 [0.1, 0.1, 0.1],
                 [0.1, 0.1, 0.1],
                 [0.1, 0.5, 0.5],
                 [0.1, 0.9, 0.9],
                 [0.1, 1.7, 1.7],
                 [0.1, 1.7, 1.7]],
                 [# Gc = 0.556
                 [1.4, 0, 0],
                 [0.5, 0.3, 0.3], 
                 [0.5, 0.3, 0.3],
                 [0.5, 0.3, 0.3],
                 [1.1, 0.4, 0.4],
                 [1.3, 0.4, 0.4],
                 [1.2, 0.8, 0.8],
                 [1.2, 0.8, 0.8],
                 [1.2, 0.8, 0.8],
                 [1.4, 0.8, 0.8]],
                 [#Gc = 1.11
                 [2.1, 1.4, 1.4],
                 [2.4, 1.1, 1.1],
                 [2.4, 1.5, 1.5],
                 [1.9, 1.6, 1.6],
                 [2.2, 1.9, 1.9],
                 [2.2, 2.2, 2.2],
                 [2.4, 2, 2],
                 [2.4, 2, 2],
                 [2.4, 2, 2],
                 [2.4, 2, 2]],
                 [#Gc = 2.78
                 [3.4, 2.7, 2.7],
                 [3.4, 2.4, 2.4],
                 [3.0, 2.2, 2.2],
                 [3.2, 1.9, 1.9],
                 [2.4, 1.9, 1.9],
                 [2.3, 1.5, 1.5],
                 [2.3, 1.5, 1.5],
                 [2.3, 1.5, 1.5],
                 [2.3, 1.5, 1.5],
                 [2.3, 1.5, 1.5]]])

_C2 = np.array([[#Gc = 0.0556
                 [9, 5, 5],
                 [8, 5, 5],
                 [7, 4, 4],
                 [7, 4, 4],
                 [7, 3, 3],
                 [5, 3, 3],
                 [5, 2, 2],
                 [3, 0, 0],
                 [1, 0, 0],
                 [1, 0, 0]],
                 [#Gc = 0.111
                 [9, 7, 7],
                 [8, 7, 7],
                 [8, 7, 7],
                 [8, 6, 6],
                 [7, 6, 6],
                 [7, 6, 6],
                 [5, 4, 4],
                 [3, 0, 0],
                 [1, 0, 0],
                 [1, 0, 0]],
                 [#Gc = 0.222
                 [8, 8, 8],
                 [7, 8, 8],
                 [7, 7, 7],
                 [7, 7, 7],
                 [7, 6, 6],
                 [7, 6, 6],
                 [7, 3, 3],
                 [7, 2, 2],
                 [7, 0, 0],
                 [7, 0, 0]],
                 [#Gc = 0.556
                 [3, 8, 8],
                 [6, 6, 6],
                 [6, 6, 6],
                 [6, 6, 6],
                 [4, 7, 7],
                 [4, 7, 7],
                 [7, 6, 6],
                 [7, 6, 6],
                 [7, 6, 6],
                 [6, 6, 6]],
                 [#Gc = 1.11
                 [1, 4, 4],
                 [2, 7, 7],
                 [2, 5, 5],
                 [6, 5, 5],
                 [8, 6, 6],
                 [8, 6, 6],
                 [8, 8, 8],
                 [8, 8, 8],
                 [8, 8, 8],
                 [8, 8, 8]],
                 [#Gc = 2.78
                 [1, 3, 3],
                 [1, 4, 4],
                 [4, 7, 7],
                 [6, 9, 9],
                 [12, 12, 12],
                 [12, 14, 14],
                 [12, 14, 14],
                 [12, 14, 14],
                 [12, 14, 14],
                 [12, 14, 14],
                 [12, 14, 14]]])

_C3 = np.array([[#Gc = 0.0556
                 [0.6, 0.5, 0.5],
                 [0.6, 0.5, 0.5],
                 [0.6, 0.5, 0.5],
                 [0.6, 0.5, 0.5],
                 [0.6, 0.5, 0.5],
                 [0.6, 0.5, 0.5],
                 [0.6, 0.5, 0.5],
                 [0.6, 0.5, 0.5],
                 [0.6, 0.5, 0.5],
                 [0.6, 0.5, 0.5]],
                 [#Gc = 0.111
                 [0.6, 0.6, 0.6],
                 [0.6, 0.6, 0.6],
                 [0.6, 0.6, 0.6],
                 [0.6, 0.6, 0.6],
                 [0.6, 0.6, 0.6],
                 [0.6, 0.6, 0.6],
                 [0.6, 0.6, 0.6],
                 [0.6, 0.6, 0.6],
                 [0.6, 0.6, 0.6],
                 [0.6, 0.6, 0.6]],
                 [#Gc = 0.222
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5]],
                 [#Gc = 0.556
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5]],
                 [#Gc = 1.11
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5]],
                 [#Gc = 2.78
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5]]])

_Kb = np.array([[#Gc = 0.0556
                [10, 15, 15],
                [10, 15, 15],
                [10, 15, 15],
                [10, 15, 15],
                [10, 15, 15],
                [5, 15, 15],
                [5, 15, 15],
                [5, 15, 15],
                [5, 15, 15],
                [5, 15, 15]],
                [#Gc = 0.111
                [10, 10, 10],
                [10, 10, 10],
                [10, 10, 10],
                [10, 10, 10],
                [5, 10, 10],
                [5, 10, 10],
                [5, 10, 10],
                [5, 10, 10],
                [5, 10, 10],
                [5, 10, 10]],
                [#Gc = 0.222
                [15, 10, 10],
                [10, 10, 10],
                [10, 10, 10],
                [10, 10, 10],
                [5, 5, 5],
                [5, 5, 5],
                [5, 5, 5],
                [5, 5, 5],
                [5, 5, 5],
                [5, 5, 5]],
                [#Gc = 0.556
                [15, 10, 10],
                [5, 5, 5],
                [5, 5, 5],
                [5, 5, 5],
                [5, 5, 5],
                [10, 5, 5],
                [10, 10, 10],
                [10, 10, 10],
                [10, 10, 10],
                [10, 10, 10]],
                [#Gc = 1.11
                [15, 15, 15],
                [15, 15, 15],
                [15, 15, 15],
                [15, 15, 15],
                [15, 15, 15],
                [15, 15, 15],
                [15, 15, 15],
                [15, 15, 15],
                [15, 15, 15],
                [15, 15, 15]],
                [#Gc = 2.78
                [20, 20, 20],
                [20, 20, 20],
                [20, 20, 20],
                [15, 15, 15],
                [15, 15, 15],
                [15, 15, 15],
                [15, 15, 15],
                [15, 15, 15],
                [15, 15, 15],
                [15, 15, 15]]])

_Gc = np.array([0.0556, 0.111, 0.222, 0.556, 1.11, 2.78])
_M = np.array([0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2, 4, 10])
_N = np.array([0.003, 0.006, 0.024])

if __name__ == '__main__':
    print(get_wall_thick(Hs, Tp, d, theta_w, mu, s, Vc, rho_s, rho_cont, rho_conc, 
                    z0, zr, theta_c, su, gammas, OD, rho_m, rho_coat, t_corr,
                    t_conc, t_m, t_coat, soil_type, rho_sw, Fc, wt, 
                    grade, T_ss, kv, theta_t, z_t, Fz, spectrum=jonswap))

