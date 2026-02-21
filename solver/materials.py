import numpy as np
from numba import njit


class TPUMaterial:
    def __init__(self, params=None):
        p = params or {}
        self.rho_polymer = p.get('rho_polymer', 1150.0)
        self.rho_air = p.get('rho_air', 100.0)
        self.cp_polymer = p.get('cp_polymer', 1800.0)
        self.cp_air = p.get('cp_air', 1005.0)
        self.k_polymer = p.get('k_polymer', 0.22)
        self.k_air = p.get('k_air', 0.026)

        self.eta_0 = p.get('eta_0', 3000.0)
        self.eta_inf = p.get('eta_inf', 50.0)
        self.lambda_cy = p.get('lambda_cy', 0.1)
        self.a_cy = p.get('a_cy', 2.0)
        self.n_cy = p.get('n_cy', 0.35)

        self.eta_min = p.get('eta_min', 1.0)
        self.eta_max = p.get('eta_max', 5000.0)

        self.T_ref = p.get('T_ref', 493.15)
        self.E_a = p.get('E_a', 40000.0)
        self.R_gas = 8.314

        self.sigma = p.get('sigma', 0.035)

        self.mu_air = p.get('mu_air', 1.8e-5)

    def get_params_tuple(self):
        return (
            self.eta_0, self.eta_inf, self.lambda_cy, self.a_cy, self.n_cy,
            self.eta_min, self.eta_max, self.T_ref, self.E_a, self.R_gas,
            self.rho_polymer, self.rho_air, self.cp_polymer, self.cp_air,
            self.k_polymer, self.k_air, self.sigma, self.mu_air
        )


@njit(cache=True)
def carreau_yasuda_viscosity(gamma_dot, T, eta_0, eta_inf, lam, a, n, T_ref, E_a, R_gas):
    aT = np.exp((E_a / R_gas) * (1.0 / T - 1.0 / T_ref))
    gamma_eff = gamma_dot / aT
    factor = (1.0 + (lam * gamma_eff) ** a) ** ((n - 1.0) / a)
    eta = aT * (eta_inf + (eta_0 - eta_inf) * factor)
    return eta


@njit(cache=True)
def blend_property(alpha, prop_polymer, prop_air):
    return alpha * prop_polymer + (1.0 - alpha) * prop_air


@njit(cache=True)
def compute_shear_rate_2d(dur_dr, duz_dz, dur_dz, duz_dr):
    D_rr = dur_dr
    D_zz = duz_dz
    D_rz = 0.5 * (dur_dz + duz_dr)
    mag = np.sqrt(2.0 * (D_rr**2 + D_zz**2 + 2.0 * D_rz**2))
    return mag
