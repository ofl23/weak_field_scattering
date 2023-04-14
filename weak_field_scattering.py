import numpy as np
from scipy.special import ellipe, ellipk, spence
import argparse



def scattering_angle_PM(m1, m2, E, J, PM_order=3):
    """
    Function to return the scattering angle at a given PM order
    Formula from arXiv:2211.01399v2 (2.4)

    m1, m2 = black hole masses 
    E = centre of mass total energy
    J = centre of mass angular momentum
    """

    M = m1 + m2                                         # Sum of masses
    nu = m1*m2 * pow(M,-2)                               # Symmetric mass ratio

    gamma = 1 + (-1 + pow(E,2)/pow(M,2))/(2.*nu)        # Relative Lorentz factor
    j = J/(m1*m2)                                       # rescaled angular momentum

    if nu_gamma_PM_conditions(nu, gamma, PM_order):
        return 0

    angle = 0

    for i in range(1,PM_order+1):
        angle += scattering_angle_PM_contribution(nu, gamma, i) / pow(j,i)

    return 2 * angle


def scattering_angle_PM_scL_resum(m1, m2, E, J, PM_order=3):
    """
    Function to return the scL resummed scattering angle at a given PM order
    Formula from arXiv:2211.01399v2 (4.5)

    m1, m2 = black hole masses 
    E = centre of mass total energy
    J = centre of mass angular momentum
    """
    
    M = m1 + m2                                         # Sum of masses
    nu = m1*m2 * pow(M,-2)                               # Symmetric mass ratio

    gamma = 1 + (-1 + pow(E,2)/pow(M,2))/(2.*nu)        # Relative Lorentz factor
    j = J/(m1*m2)                                       # rescaled angular momentum

    if nu_gamma_PM_conditions(nu, gamma, PM_order):
        return 0

    angle = 0

    for i in range(1,PM_order+1):
        angle += scattering_angle_PM_contribution_scL_resum(nu, gamma, i, PM_order) / pow(j,i)

    return 2 * scL(critical_rescaled_angular_momentum_PM(nu, gamma, PM_order) / j) * angle



def scattering_angle_PM_EOB(m1, m2, E, J, PM_order=2):
    """
    Function to return the EOB scattering angle at a given PM order
    Formulae from arXiv:2211.01399v2 (3.12) - (3.15)

    m1, m2 = black hole masses 
    E = centre of mass total energy
    J = centre of mass angular momentum
    """
    
    M = m1 + m2                                         # Sum of masses
    nu = m1*m2 * pow(M,-2)                               # Symmetric mass ratio

    gamma = 1 + (-1 + pow(E,2)/pow(M,2))/(2.*nu)        # Relative Lorentz factor
    j = J/(m1*m2)                                       # rescaled angular momentum

    if nu_gamma_PM_conditions(nu, gamma, PM_order):
        return 0

    if PM_order == 1:
        return 2 * np.arctan(w_potential_PM(nu, gamma, 1) / (2*j* np.sqrt(pow(gamma,2)-1)))

    if PM_order == 2:
        w_1 = w_potential_PM(nu, gamma, 1)
        w_2 = w_potential_PM(nu, gamma, 2)

        if pow(j,2) < w_2:
            print('Error: j^2 < w_2 in 2PM EOB gives a plunge orbit')
            return 0

        sqrt_factor = np.sqrt(pow(w_1,2) - 4*(pow(gamma,2)-1)*(w_2 - pow(j,2)))
        angle = 4*j*np.arctan(np.sqrt((sqrt_factor+w_1)/(sqrt_factor-w_1))) / np.sqrt(pow(j,2)-w_2)

        return angle - np.pi
    
    if PM_order > 2:
        print('Error: Above 2PM not currently implemented in scattering_angle_PM_EOB')
        return 0



def scattering_angle_PM_contribution(nu, gamma, PM_order):
    """
    Function to return the PM coefficient of the scattering angle at a given PM order
    Defined as chi_i in arXiv:2211.01399v2 (2.4)

    nu = m1 m2 / (m1 + m2)^2 = symmetric mass ratio 
    gamma = relative Lorentz factor
    """
    if nu_gamma_PM_conditions(nu, gamma, PM_order):
        return 0

    if PM_order == 1:
        return w_potential_PM(nu, gamma, 1) / (2*np.sqrt(pow(gamma,2) - 1))

    if PM_order == 2:
        return w_potential_PM(nu, gamma, 2)*np.pi/4.

    if PM_order == 3:

        pInf = np.sqrt(pow(gamma,2) - 1)

        chi_3 = -1./24. * pow(w_potential_PM(nu, gamma, 1) / pInf,3)
        chi_3 += w_potential_PM(nu, gamma, 1) * w_potential_PM(nu, gamma, 2) / (2*pInf)
        chi_3 += w_potential_PM(nu, gamma, 3) * pInf

        return chi_3
    
    if PM_order == 4:

        h2 = 169 + 380*pow(gamma,2)
        h3 = 834 + 2095*gamma + 1200*pow(gamma,2)
        h4 = 1183 + 2929*gamma + 2660*pow(gamma,2) + 1200*pow(gamma,3)
        h5 = -12 + 76*gamma - 129*pow(gamma,2) + 60*pow(gamma,3) + 30*pow(gamma,4) - 25*pow(gamma,6)
        h6 = 1151 - 3336*gamma + 3148*pow(gamma,2) - 912*pow(gamma,3) + 339*pow(gamma,4) - 552*pow(gamma,5) + 210*pow(gamma,6)
        h7 = -(gamma*(-3 + 2*pow(gamma,2))*(4 - 15*gamma + 15*pow(gamma,2)))
        h9 = (-210 - 210*gamma + 885*pow(gamma,2) + 885*pow(gamma,3) - 3457*pow(gamma,4) - 3457*pow(gamma,5) + 9593*pow(gamma,6) 
            + 9593*pow(gamma,7) + 3259*pow(gamma,8) - 181493*pow(gamma,9) + 535259*pow(gamma,10) - 500785*pow(gamma,11) 
            - 32675*pow(gamma,12) + 333545*pow(gamma,13) - 304761*pow(gamma,14) + 232751*pow(gamma,15) + 74431*pow(gamma,16) 
            - 216185*pow(gamma,17) - 34080*pow(gamma,18) + 116100*pow(gamma,19) + 11340*pow(gamma,20) - 22680*pow(gamma,21))

        h11 = (2074 + 10643*gamma + 18958*pow(gamma,2) + 11391*pow(gamma,3) + 5242*pow(gamma,4) - 9826*pow(gamma,5) 
            + 1818*pow(gamma,6) + 13198*pow(gamma,7) - 700*pow(gamma,8) - 10065*pow(gamma,9) + 2835*pow(gamma,11))
        h12 = gamma*(5369 + 8077*pow(gamma,2) - 5014*pow(gamma,4) + 4874*pow(gamma,6) - 2955*pow(gamma,8) + 945*pow(gamma,10))
        h13 = gamma*(-1965 + 2169*gamma + 1289*pow(gamma,2) - 2211*pow(gamma,3) - 856*pow(gamma,4) + 90*pow(gamma,5) 
                     + 580*pow(gamma,6) + 280*pow(gamma,7))
        h14 = gamma*(-3 + 2*pow(gamma,2))*(85 - 82*gamma - 716*pow(gamma,2) + 380*pow(gamma,3) + 1537*pow(gamma,4) 
                                           - 610*pow(gamma,5) - 890*pow(gamma,6) + 280*pow(gamma,7))
        h15 = -5 + 76*gamma - 150*pow(gamma,2) + 60*pow(gamma,3) + 35*pow(gamma,4)
        h16 = gamma*(-3 + 2*pow(gamma,2))*(11 - 30*pow(gamma,2) + 35*pow(gamma,4))
        h17 = 299 - 1216*gamma + 1732*pow(gamma,2) - 960*pow(gamma,3) + 690*pow(gamma,4) - 860*pow(gamma,6) + 315*pow(gamma,8)
        h18 = 21 + 65*pow(gamma,2) - 145*pow(gamma,4) + 315*pow(gamma,6)
        
        h20 = (-45 + 207*pow(gamma,2) - 1471*pow(gamma,4) + 13349*pow(gamma,6) - 37478*pow(gamma,7) + 63848*pow(gamma,8) 
            - 47540*pow(gamma,9) - 9872*pow(gamma,10) + 16138*pow(gamma,11) + 14128*pow(gamma,12) + 7824*pow(gamma,13) 
            - 23840*pow(gamma,14) + 4320*pow(gamma,15) + 3600*pow(gamma,16))
        h22 = (1759 + 6744*gamma + 3692*pow(gamma,2) + 2044*pow(gamma,3) + 2787*pow(gamma,4) + 1112*pow(gamma,5) + 210*pow(gamma,6) 
            - 300*pow(gamma,7))
        h23 = gamma*(-852 - 283*pow(gamma,2) - 140*pow(gamma,4) + 75*pow(gamma,6))
        h24 = gamma*(-3 + 2*pow(gamma,2))*(1151 - 3504*gamma + 3148*pow(gamma,2) - 576*pow(gamma,3) + 339*pow(gamma,4)
                                            - 720*pow(gamma,5) + 210*pow(gamma,6))
        h25 = gamma*(-3 + 2*pow(gamma,2))*(96 - 93*gamma - 768*pow(gamma,2) + 432*pow(gamma,3) + 1632*pow(gamma,4) 
                                           - 705*pow(gamma,5) - 960*pow(gamma,6) + 350*pow(gamma,7))
        h26 = pow(gamma,2)*pow(3 - 2*pow(gamma,2),2)*(11 - 30*pow(gamma,2) + 35*pow(gamma,4))
        h27 = 8 + 19*gamma + 60*pow(gamma,2) + 15*pow(gamma,3)
        h28 = gamma*(63 + 768*pow(gamma,2) - 645*pow(gamma,4) + 70*pow(gamma,6))
        h29 = 60 + 333*pow(gamma,2) + 90*pow(gamma,4) - 75*pow(gamma,6)

        h30 = 12 + 76*gamma + 129*pow(gamma,2) + 60*pow(gamma,3) - 30*pow(gamma,4) + 25*pow(gamma,6)

        h61 = 35*(-1 + gamma)*(1 + gamma)*(1 - 18*pow(gamma,2) + 33*pow(gamma,4))
        h62 = (-45 + 207*pow(gamma,2) - 1471*pow(gamma,4) + 13349*pow(gamma,6) - 38135*pow(gamma,7) + 64424*pow(gamma,8) 
            - 32177*pow(gamma,9) - 15056*pow(gamma,10) - 25145*pow(gamma,11) + 27952*pow(gamma,12) + 33249*pow(gamma,13) 
            - 35360*pow(gamma,14) + 4320*pow(gamma,15) + 3600*pow(gamma,16))
        h63 = pow(gamma,2)*(-3 + 2*pow(gamma,2))*(-1 + 2*pow(gamma,2))*(11 - 30*pow(gamma,2) + 35*pow(gamma,4))
        h64 = (-102 + 2681*gamma - 6210*pow(gamma,2) + 10052*pow(gamma,3) - 9366*pow(gamma,4) - 8491*pow(gamma,5) 
            + 15018*pow(gamma,6) + 702*pow(gamma,7) - 4140*pow(gamma,8))
        h65 = (124 - 295*gamma - 508*pow(gamma,2) + 1200*pow(gamma,3) + 216*pow(gamma,4) - 755*pow(gamma,5) - 240*pow(gamma,6) 
            + 210*pow(gamma,7))
        h66 = gamma*(-3 + 2*pow(gamma,2))*(-1 + 2*pow(gamma,2))*(11 - 30*pow(gamma,2) + 35*pow(gamma,4))
        h67 = -((-1 + gamma)*(-947 - 3177*gamma + 14910*pow(gamma,2) - 26710*pow(gamma,3) + 18929*pow(gamma,4) + 21667*pow(gamma,5) 
                              - 30840*pow(gamma,6) - 2040*pow(gamma,7) + 7596*pow(gamma,8) + 420*pow(gamma,9)))
        h68 = (-1 + gamma)*(253 - 661*gamma - 952*pow(gamma,2) + 2632*pow(gamma,3) + 189*pow(gamma,4) - 1725*pow(gamma,5) 
                            - 290*pow(gamma,6) + 490*pow(gamma,7))

        
        return (pow(-1 + pow(gamma,2),2)*np.pi*(16*nu*((-3*(h25 + h25*(-3 + gamma)*nu + h14*(1 + gamma)*nu)*np.arccosh(gamma))/pow(-1 + pow(gamma,2),4) + 
          (12*(h63 + h66*pow(-1 + gamma,2)*nu)*np.arcsinh(np.sqrt(-1 + gamma)/np.sqrt(2)))/pow(-1 + pow(gamma,2),4) + 
          (h64 + h67*nu + 6*(-1 + pow(gamma,2))*(h65 + h68*nu)*np.log((1 + gamma)/2.))/pow(-1 + pow(gamma,2),3.5)) + 
       pow(nu,2)*(-((h9 - 4*h20*pow(gamma,2)*(1 + gamma))/(pow(gamma,9)*pow(-1 + pow(gamma,2),3))) + 
          (48*(2*h13*pow(-1 + gamma,2) - h24*(1 + gamma))*np.arccosh(gamma))/pow(-1 + pow(gamma,2),3.5) - 
          (72*h26*pow(np.arccosh(gamma),2))/(pow(-1 + gamma,4)*pow(1 + gamma,3)) - 
          (48*(h12 - 8*h23*(-1 + pow(gamma,2)))*np.log(gamma))/(pow(-1 + gamma,3)*pow(1 + gamma,2)) + 
          (24*(h11 + 2*h22*(-1 + pow(gamma,2)) + 6*(h16 + h28)*np.sqrt(-1 + pow(gamma,2))*np.arccosh(gamma))*np.log((1 + gamma)/2.))/(pow(-1 + gamma,3)*pow(1 + gamma,2)) - 
          (288*(h15 - 4*h27)*pow(np.log((1 + gamma)/2.),2))/(-1 + gamma) + (24*(8*h29 + 3*h18*(-1 + pow(gamma,2)))*spence(1 - (1 - gamma)/(1 + gamma)))/(-1 + gamma) + 
          (36*(h17 + 8*h30)*spence(1 - (-1 + gamma)/(1 + gamma)))/(-1 + gamma)) + 
       4*(1 + 2*(-1 + gamma)*nu)*((9*h61)/pow(-1 + pow(gamma,2),3) + 
          nu*(-(h62/(pow(gamma,7)*pow(-1 + pow(gamma,2),3))) - (24*h5*pow(np.pi,2))/(-1 + pow(gamma,2)) + (12*h24*np.arccosh(gamma))/pow(-1 + pow(gamma,2),3.5) + 
             (18*h26*pow(np.arccosh(gamma),2))/pow(-1 + pow(gamma,2),4) - (126*h2*pow(ellipe((-1 + gamma)/(1 + gamma)),2))/(pow(-1 + gamma,2)*(1 + gamma)) + 
             (36*h4*ellipe((-1 + gamma)/(1 + gamma))*ellipk((-1 + gamma)/(1 + gamma)))/pow(-1 + pow(gamma,2),2) - 
             (36*h3*pow(ellipk((-1 + gamma)/(1 + gamma)),2))/pow(-1 + pow(gamma,2),2) - (96*h23*np.log(gamma))/pow(-1 + pow(gamma,2),2) - 
             (12*(h22*np.sqrt(-1 + pow(gamma,2)) + 3*h28*np.arccosh(gamma))*np.log((1 + gamma)/2.))/pow(-1 + pow(gamma,2),2.5) - 
             (288*h27*pow(np.log((1 + gamma)/2.),2))/(-1 + pow(gamma,2)) + 
             (12*np.log((-1 + gamma)/2.)*(-3*h16*np.arccosh(gamma) + np.sqrt(-1 + pow(gamma,2))*(-h6 + 6*h15*(-1 + pow(gamma,2))*np.log((1 + gamma)/2.))))/
              pow(-1 + pow(gamma,2),2.5) - (576*h7*np.sqrt(-1 + pow(gamma,2))*spence(1 - np.sqrt((-1 + gamma)/(1 + gamma))))/(pow(-1 + gamma,2)*pow(1 + gamma,3)) - 
             (48*h29*spence(1 - (1 - gamma)/(1 + gamma)))/(-1 + pow(gamma,2)) + 
             (72*(-(h30*(-1 + gamma)*pow(1 + gamma,2)) + 2*h7*np.sqrt(-1 + pow(gamma,2)))*spence(1 - (-1 + gamma)/(1 + gamma)))/(pow(-1 + gamma,2)*pow(1 + gamma,3))))))/(1536.*pow(1 + 2*(-1 + gamma)*nu,2.5))
    


def scattering_angle_PM_contribution_scL_resum(nu, gamma, PM_contribution_order, PM_order):
    """
    Function to return the PM coefficient of the scattering angle at a given PM order for the resummed in scL
    Expansion defined in arXiv:2211.01399v2 (4.6)
    Defined as {\tilde chi_i} in arXiv:2211.01399v2 (4.7)

    nu = m1 m2 / (m1 + m2)^2 = symmetric mass ratio 
    gamma = relative Lorentz factor

    PM_contribution_order is the order of the contribution requested
    PM_order is the order of the final result (which is needed of j0)

    """
    if nu_gamma_PM_conditions(nu, gamma, PM_order):
        return 0

    if PM_contribution_order > PM_order:
        print('Error: PM_contribution_order is greater than PM_order in scattering_angle_PM_contribution_scL_resum')
    
    j0 = critical_rescaled_angular_momentum_PM(nu, gamma, PM_order)

    if PM_contribution_order == 1:
        return scattering_angle_PM_contribution(nu, gamma, 1)

    if PM_contribution_order == 2:
        chi_2 = - 0.5*j0 * scattering_angle_PM_contribution(nu, gamma, 1)
        return chi_2 + scattering_angle_PM_contribution(nu, gamma, 2)

    if PM_contribution_order == 3:
        chi_3 = - pow(j0,2) / 12. * scattering_angle_PM_contribution(nu, gamma, 1)
        chi_3 += - 0.5 * j0 * scattering_angle_PM_contribution(nu, gamma, 2)
        return chi_3 + scattering_angle_PM_contribution(nu, gamma, 3)
    
    if PM_contribution_order == 4:
        chi_4 = - pow(j0,3) / 24. * scattering_angle_PM_contribution(nu, gamma, 1)
        chi_4 += - pow(j0,2) / 12. * scattering_angle_PM_contribution(nu, gamma, 2)
        chi_4 += - 0.5 * j0 * scattering_angle_PM_contribution(nu, gamma, 3)
        return chi_4 + scattering_angle_PM_contribution(nu, gamma, 4)



def w_potential_PM(nu, gamma, PM_order):
    """
    Function to return the w potential at a given PM order
    Formulae from arXiv:2211.01399v2 (2.36)

    nu = m1 m2 / (m1 + m2)^2 = symmetric mass ratio 
    gamma = relative Lorentz factor
    """

    if nu_gamma_PM_conditions(nu, gamma, PM_order):
        return 0

    if PM_order == 1:
        return 2 * (2*pow(gamma,2) - 1)
    
    if PM_order == 2:
        return 3./2. * (5*pow(gamma,2) - 1) / rescaled_energy(nu, gamma)
    
    if PM_order == 3:

        A = 2 * np.arcsinh(np.sqrt((gamma - 1) / 2.))
        B = 3./2. * (2*pow(gamma, 2)-1)*(5*pow(gamma,2) - 1)/(pow(gamma,2)-1)
        pInf = np.sqrt(pow(gamma,2) - 1)

        C_cons = 2./3. * gamma*(14*pow(gamma,2) + 25) + 2*(4*pow(gamma,4) - 12*pow(gamma,2) - 3)*A/pInf
        C_rad = gamma*pow(2*pow(gamma,2) - 1, 2)/(3*pow(pow(gamma,2) - 1, 2))
        C_rad *= (pInf*(5*pow(gamma,2) - 8) / gamma + (9 - 6*pow(gamma,2))*A)

        w_cons = 9*pow(gamma,2) - 1./2. - B*(1/rescaled_energy(nu, gamma) - 1) - 2*C_cons*nu/pow(rescaled_energy(nu, gamma),2)
        w_rad = -2*C_rad*nu/pow(rescaled_energy(nu, gamma),2)

        return w_cons + w_rad


def rescaled_energy(nu, gamma):
    """
    Function to return the rescaled energy
    Formulae from arXiv:2211.01399v2 (2.6)

    nu = m1 m2 / (m1 + m2)^2 = symmetric mass ratio 
    gamma = relative Lorentz factor
    """

    return np.sqrt(1 + 2 * nu * (gamma - 1))



def critical_rescaled_angular_momentum_PM(nu, gamma, PM_order):
    """
    Function to return the critical value of the rescaled angular momentum
    Formulae from arXiv:2211.01399v2 (4.10)

    nu = m1 m2 / (m1 + m2)^2 = symmetric mass ratio 
    gamma = relative Lorentz factor
    """

    if nu_gamma_PM_conditions(nu, gamma, PM_order):
        return 0
    
    if PM_order == 1:
        print('Error: PM_order of critical_rescaled_angular_momentum_PM has to be greater than 1')
        return 0

    j0 = PM_order * scattering_angle_PM_contribution(nu, gamma, PM_order)/scattering_angle_PM_contribution(nu, gamma, 1)
    return pow(j0, 1/(PM_order-1.))



def scL(x):
    """
    Function to return the resum function scL
    Formulae from arXiv:2211.01399v2 (4.3)
    """

    return 1/x * np.log(1/(1-x))


def nu_gamma_PM_conditions(nu, gamma, PM_order):
    """
    Function to check values of nu, gamma and PM_order
    Return FALSE if any errors detected

    nu = m1 m2 / (m1 + m2)^2 = symmetric mass ratio 
    gamma = relative Lorentz factor
    """
    error = 0

    if nu > 0.25:
        print('Error: value of symmetric mass ratio (nu) must be less than 1/4')
        error = 1
    
    if gamma < 1:
        print('Error: value of relative Lorentz factor (gamma) must be greater than 1')
        error = 1

    if not isinstance(PM_order, int):
        print('Error: PM value must be an integer')
        error = 1

    if PM_order <= 0:
        print('Error: PM value must be greater than zero')
        error = 1

    if PM_order > 4:
        print('Error: PM results only available up to 4PM')
        error = 1

    return bool(error)