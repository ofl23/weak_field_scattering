import numpy as np
import argparse



def scattering_angle_PM(m1, m2, E, J, PM_order=4):
    """
    Function to return the scattering angle at a given PM order
    Formula from arXiv:2211.01399v2 (2.4)

    m1, m2 = black hole masses 
    E = centre of mass total energy
    J = centre of mass angular momentum
    """
        
    M = m1 + m2
    nu = m1*m2 / pow(M,2)

    gamma = 1 + (-1 + pow(E,2)/pow(M,2))/(2.*nu)
    j = J/(m1*m2)

    chi = 0

    for i in range(1,PM_order+1):
        chi += scattering_angle_PM_contribution(nu, gamma, i) / pow(j,i)

    return 2 * chi



def scattering_angle_PM_contribution(nu, gamma, PM_order):
    """
    Function to return the PM coefficient of the scattering angle at a given PM order
    Defined as chi_i in arXiv:2211.01399v2 (2.4)

    nu = m1 m2 / (m1 + m2)^2 = symmetric mass ratio 
    gamma = relative Lorentz factor
    """

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



def w_potential_PM(nu, gamma, PM_order=4):
    """
    Function to return the w potential at a given PM order
    Formulae from arXiv:2211.01399v2 (2.36)

    nu = m1 m2 / (m1 + m2)^2 = symmetric mass ratio 
    gamma = relative Lorentz factor
    """

    if PM_order > 4:
        print('Error: w_potenital only avilable up to 4PM')

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

        print((9 - 6*pow(gamma,2))*A)
        print((pInf*(5*pow(gamma,2) - 8) / gamma))
        print(C_rad)

        return w_cons + w_rad



def rescaled_energy(nu, gamma):
    """
    Function to return the rescaled energy
    Formulae from arXiv:2211.01399v2 (2.6)

    nu = m1 m2 / (m1 + m2)^2 = symmetric mass ratio 
    gamma = relative Lorentz factor
    """

    return np.sqrt(1 + 2 * nu * (gamma - 1))


