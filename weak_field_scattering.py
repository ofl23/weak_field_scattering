import numpy as np
import argparse



def scattering_angle_PM(m1, m2, E, J, PM_order=3):
    """
    Function to return the scattering angle at a given PM order
    Formula from arXiv:2211.01399v2 (2.4)

    m1, m2 = black hole masses 
    E = centre of mass total energy
    J = centre of mass angular momentum
    """
    if nu_gamma_PM_conditions(nu, gamma, PM_order):
        return 0
    
    M = m1 + m2                                         # Sum of masses
    nu = m1*m2 * pow(M,-2)                               # Symmetric mass ratio

    gamma = 1 + (-1 + pow(E,2)/pow(M,2))/(2.*nu)        # Relative Lorentz factor
    j = J/(m1*m2)                                       # rescaled angular momentum

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
        print(angle)

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
        chi_4 += - pow(j0,3) / 24. * scattering_angle_PM_contribution(nu, gamma, 1)
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