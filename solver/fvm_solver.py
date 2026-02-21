import numpy as np
import math
from numba import njit
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import bicgstab
import time


def sanitize_float(v):
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return 0.0
    return v


def sanitize_list(lst):
    if isinstance(lst, list):
        return [sanitize_list(x) if isinstance(x, list) else sanitize_float(x) for x in lst]
    return sanitize_float(lst)


def sanitize_dict(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = sanitize_dict(v)
        elif isinstance(v, list):
            out[k] = sanitize_list(v)
        elif isinstance(v, float):
            out[k] = sanitize_float(v)
        else:
            out[k] = v
    return out


@njit(cache=True)
def compute_velocity_gradients(ur, uz, nr, nz, dr, dz):
    dur_dr = np.zeros((nr, nz))
    dur_dz = np.zeros((nr, nz))
    duz_dr = np.zeros((nr, nz))
    duz_dz = np.zeros((nr, nz))

    for i in range(nr):
        for j in range(nz):
            if i == 0:
                dur_dr[i, j] = (ur[1, j] - ur[0, j]) / dr if nr > 1 else 0.0
                duz_dr[i, j] = (uz[1, j] - uz[0, j]) / dr if nr > 1 else 0.0
            elif i == nr - 1:
                dur_dr[i, j] = (ur[i, j] - ur[i - 1, j]) / dr
                duz_dr[i, j] = (uz[i, j] - uz[i - 1, j]) / dr
            else:
                dur_dr[i, j] = (ur[i + 1, j] - ur[i - 1, j]) / (2.0 * dr)
                duz_dr[i, j] = (uz[i + 1, j] - uz[i - 1, j]) / (2.0 * dr)

            if j == 0:
                dur_dz[i, j] = (ur[i, 1] - ur[i, 0]) / dz if nz > 1 else 0.0
                duz_dz[i, j] = (uz[i, 1] - uz[i, 0]) / dz if nz > 1 else 0.0
            elif j == nz - 1:
                dur_dz[i, j] = (ur[i, j] - ur[i, j - 1]) / dz
                duz_dz[i, j] = (uz[i, j] - uz[i, j - 1]) / dz
            else:
                dur_dz[i, j] = (ur[i, j + 1] - ur[i, j - 1]) / (2.0 * dz)
                duz_dz[i, j] = (uz[i, j + 1] - uz[i, j - 1]) / (2.0 * dz)

    return dur_dr, dur_dz, duz_dr, duz_dz


@njit(cache=True)
def compute_shear_rate_field(dur_dr, dur_dz, duz_dr, duz_dz, ur, r_centers, nr, nz):
    gamma_dot = np.zeros((nr, nz))
    for i in range(nr):
        for j in range(nz):
            D_rr = dur_dr[i, j]
            D_zz = duz_dz[i, j]
            D_rz = 0.5 * (dur_dz[i, j] + duz_dr[i, j])
            r = max(r_centers[i], 1e-12)
            D_tt = ur[i, j] / r
            gamma_dot[i, j] = np.sqrt(2.0 * (D_rr**2 + D_zz**2 + D_tt**2 + 2.0 * D_rz**2))
    return gamma_dot


@njit(cache=True)
def compute_viscosity_field(gamma_dot, T, alpha, nr, nz,
                            eta_0, eta_inf, lam, a_cy, n_cy,
                            T_ref, E_a, R_gas, eta_min, eta_max, mu_air):
    eta = np.zeros((nr, nz))
    for i in range(nr):
        for j in range(nz):
            if alpha[i, j] > 0.01:
                aT = np.exp((E_a / R_gas) * (1.0 / T[i, j] - 1.0 / T_ref))
                aT = min(max(aT, 1e-6), 1e6)
                gamma_eff = gamma_dot[i, j] / max(aT, 1e-30)
                factor = (1.0 + (lam * gamma_eff) ** a_cy) ** ((n_cy - 1.0) / a_cy)
                eta_val = aT * (eta_inf + (eta_0 - eta_inf) * factor)
                eta_val = min(max(eta_val, eta_min), eta_max)
                eta[i, j] = alpha[i, j] * eta_val + (1.0 - alpha[i, j]) * mu_air
            else:
                eta[i, j] = mu_air
    return eta


@njit(cache=True)
def compute_momentum_rhs_axisym(ur, uz, rho, eta, alpha,
                                 r_centers, r_faces, dr, dz, nr, nz,
                                 gravity, f_st_r, f_st_z):
    rhs_ur = np.zeros((nr, nz))
    rhs_uz = np.zeros((nr, nz))

    for i in range(nr):
        for j in range(nz):
            r_c = max(r_centers[i], 1e-12)
            rho_c = max(rho[i, j], 1e-10)
            V = np.pi * (r_faces[i + 1]**2 - r_faces[i]**2) * dz

            conv_ur = 0.0
            conv_uz = 0.0
            r_e = r_faces[i + 1]
            r_w = r_faces[i]
            A_e = 2.0 * np.pi * r_e * dz
            A_w = 2.0 * np.pi * r_w * dz
            A_n = np.pi * (r_e**2 - r_w**2)
            A_s = A_n

            if i < nr - 1:
                ur_e = 0.5 * (ur[i, j] + ur[i + 1, j])
                rho_e = 0.5 * (rho[i, j] + rho[i + 1, j])
                mdot_e = rho_e * ur_e * A_e
                if ur_e >= 0:
                    conv_ur += mdot_e * ur[i, j]
                    conv_uz += mdot_e * uz[i, j]
                else:
                    conv_ur += mdot_e * ur[i + 1, j]
                    conv_uz += mdot_e * uz[i + 1, j]
            else:
                conv_ur += 0.0
                conv_uz += 0.0

            if i > 0:
                ur_w = 0.5 * (ur[i - 1, j] + ur[i, j])
                rho_w = 0.5 * (rho[i - 1, j] + rho[i, j])
                mdot_w = rho_w * ur_w * A_w
                if ur_w >= 0:
                    conv_ur -= mdot_w * ur[i - 1, j]
                    conv_uz -= mdot_w * uz[i - 1, j]
                else:
                    conv_ur -= mdot_w * ur[i, j]
                    conv_uz -= mdot_w * uz[i, j]
            else:
                pass

            if j < nz - 1:
                uz_n = 0.5 * (uz[i, j] + uz[i, j + 1])
                rho_n = 0.5 * (rho[i, j] + rho[i, j + 1])
                mdot_n = rho_n * uz_n * A_n
                if uz_n >= 0:
                    conv_ur += mdot_n * ur[i, j]
                    conv_uz += mdot_n * uz[i, j]
                else:
                    conv_ur += mdot_n * ur[i, j + 1]
                    conv_uz += mdot_n * uz[i, j + 1]
            else:
                uz_n = uz[i, j]
                mdot_n = rho[i, j] * uz_n * A_n
                conv_ur += mdot_n * ur[i, j]
                conv_uz += mdot_n * uz[i, j]

            if j > 0:
                uz_s = 0.5 * (uz[i, j - 1] + uz[i, j])
                rho_s = 0.5 * (rho[i, j - 1] + rho[i, j])
                mdot_s = rho_s * uz_s * A_s
                if uz_s >= 0:
                    conv_ur -= mdot_s * ur[i, j - 1]
                    conv_uz -= mdot_s * uz[i, j - 1]
                else:
                    conv_ur -= mdot_s * ur[i, j]
                    conv_uz -= mdot_s * uz[i, j]

            diff_ur = 0.0
            diff_uz = 0.0

            if i < nr - 1:
                eta_e = 0.5 * (eta[i, j] + eta[i + 1, j])
                diff_ur += eta_e * A_e * (ur[i + 1, j] - ur[i, j]) / dr
                diff_uz += eta_e * A_e * (uz[i + 1, j] - uz[i, j]) / dr
            if i > 0:
                eta_w = 0.5 * (eta[i - 1, j] + eta[i, j])
                diff_ur -= eta_w * A_w * (ur[i, j] - ur[i - 1, j]) / dr
                diff_uz -= eta_w * A_w * (uz[i, j] - uz[i - 1, j]) / dr
            elif i == 0:
                diff_ur -= eta[i, j] * A_w * ur[i, j] / (0.5 * dr)
                diff_uz -= 0.0

            if j < nz - 1:
                eta_n = 0.5 * (eta[i, j] + eta[i, j + 1])
                diff_ur += eta_n * A_n * (ur[i, j + 1] - ur[i, j]) / dz
                diff_uz += eta_n * A_n * (uz[i, j + 1] - uz[i, j]) / dz
            if j > 0:
                eta_s = 0.5 * (eta[i, j - 1] + eta[i, j])
                diff_ur -= eta_s * A_s * (ur[i, j] - ur[i, j - 1]) / dz
                diff_uz -= eta_s * A_s * (uz[i, j] - uz[i, j - 1]) / dz

            hoop_stress = -eta[i, j] * ur[i, j] / (r_c**2) * V

            rhs_ur[i, j] = (-conv_ur + diff_ur + hoop_stress +
                           f_st_r[i, j] * V) / V
            rhs_uz[i, j] = (-conv_uz + diff_uz +
                           rho_c * gravity * V + f_st_z[i, j] * V) / V

    return rhs_ur, rhs_uz


@njit(cache=True)
def compute_divergence_axisym(ur, uz, r_centers, r_faces, dr, dz, nr, nz):
    div = np.zeros((nr, nz))
    for i in range(nr):
        for j in range(nz):
            r_c = max(r_centers[i], 1e-12)
            r_e = r_faces[i + 1]
            r_w = r_faces[i]

            if i < nr - 1:
                ur_e = 0.5 * (ur[i, j] + ur[i + 1, j])
            else:
                ur_e = 0.0

            if i > 0:
                ur_w = 0.5 * (ur[i - 1, j] + ur[i, j])
            else:
                ur_w = 0.0

            div_r = (r_e * ur_e - r_w * ur_w) / (r_c * dr)

            if j < nz - 1:
                uz_n = 0.5 * (uz[i, j] + uz[i, j + 1])
            else:
                uz_n = uz[i, j]

            if j > 0:
                uz_s = 0.5 * (uz[i, j - 1] + uz[i, j])
            else:
                uz_s = uz[i, j]

            div_z = (uz_n - uz_s) / dz
            div[i, j] = div_r + div_z

    return div


@njit(cache=True)
def advect_vof_axisym(alpha, ur, uz, r_centers, r_faces, dr, dz, nr, nz, dt):
    alpha_new = alpha.copy()

    for i in range(nr):
        for j in range(nz):
            r_c = max(r_centers[i], 1e-12)
            r_e = r_faces[i + 1]
            r_w = r_faces[i]
            V = np.pi * (r_e**2 - r_w**2) * dz

            flux_e = 0.0
            if i < nr - 1:
                ur_e = 0.5 * (ur[i, j] + ur[i + 1, j])
                A_e = 2.0 * np.pi * r_e * dz
                if ur_e >= 0:
                    flux_e = ur_e * alpha[i, j] * A_e
                else:
                    flux_e = ur_e * alpha[i + 1, j] * A_e

            flux_w = 0.0
            if i > 0:
                ur_w = 0.5 * (ur[i - 1, j] + ur[i, j])
                A_w = 2.0 * np.pi * r_w * dz
                if ur_w >= 0:
                    flux_w = ur_w * alpha[i - 1, j] * A_w
                else:
                    flux_w = ur_w * alpha[i, j] * A_w

            flux_n = 0.0
            A_z = np.pi * (r_e**2 - r_w**2)
            if j < nz - 1:
                uz_n = 0.5 * (uz[i, j] + uz[i, j + 1])
                if uz_n >= 0:
                    flux_n = uz_n * alpha[i, j] * A_z
                else:
                    flux_n = uz_n * alpha[i, j + 1] * A_z
            else:
                uz_n = uz[i, j]
                if uz_n >= 0:
                    flux_n = uz_n * alpha[i, j] * A_z
                else:
                    flux_n = 0.0

            flux_s = 0.0
            if j > 0:
                uz_s = 0.5 * (uz[i, j - 1] + uz[i, j])
                if uz_s >= 0:
                    flux_s = uz_s * alpha[i, j - 1] * A_z
                else:
                    flux_s = uz_s * alpha[i, j] * A_z

            d_alpha = -dt * (flux_e - flux_w + flux_n - flux_s) / V
            alpha_new[i, j] = min(max(alpha[i, j] + d_alpha, 0.0), 1.0)

    return alpha_new


@njit(cache=True)
def apply_vof_compression(alpha, ur, uz, r_centers, r_faces, dr, dz, nr, nz, dt, c_alpha):
    alpha_new = alpha.copy()

    for i in range(nr):
        for j in range(nz):
            r_c = max(r_centers[i], 1e-12)

            grad_a_r = 0.0
            grad_a_z = 0.0
            if i > 0 and i < nr - 1:
                grad_a_r = (alpha[i + 1, j] - alpha[i - 1, j]) / (2.0 * dr)
            elif i == 0 and nr > 1:
                grad_a_r = (alpha[1, j] - alpha[0, j]) / dr
            elif i == nr - 1:
                grad_a_r = (alpha[i, j] - alpha[i - 1, j]) / dr

            if j > 0 and j < nz - 1:
                grad_a_z = (alpha[i, j + 1] - alpha[i, j - 1]) / (2.0 * dz)
            elif j == 0 and nz > 1:
                grad_a_z = (alpha[i, j + 1] - alpha[i, j]) / dz
            elif j == nz - 1:
                grad_a_z = (alpha[i, j] - alpha[i, j - 1]) / dz

            mag_grad = np.sqrt(grad_a_r**2 + grad_a_z**2)
            if mag_grad < 1e-10:
                continue

            n_r = grad_a_r / mag_grad
            n_z = grad_a_z / mag_grad

            u_max_local = max(abs(ur[i, j]), abs(uz[i, j]))
            u_comp_r = c_alpha * u_max_local * n_r
            u_comp_z = c_alpha * u_max_local * n_z

            a = alpha[i, j]
            comp_term = a * (1.0 - a)

            flux_comp = 0.0
            if i < nr - 1:
                comp_e = 0.5 * (comp_term + alpha[i + 1, j] * (1.0 - alpha[i + 1, j]))
                flux_comp += u_comp_r * comp_e * r_faces[i + 1] / (r_c * dr)
            if i > 0:
                comp_w = 0.5 * (alpha[i - 1, j] * (1.0 - alpha[i - 1, j]) + comp_term)
                flux_comp -= u_comp_r * comp_w * r_faces[i] / (r_c * dr)

            if j < nz - 1:
                comp_n = 0.5 * (comp_term + alpha[i, j + 1] * (1.0 - alpha[i, j + 1]))
                flux_comp += u_comp_z * comp_n / dz
            if j > 0:
                comp_s = 0.5 * (alpha[i, j - 1] * (1.0 - alpha[i, j - 1]) + comp_term)
                flux_comp -= u_comp_z * comp_s / dz

            alpha_new[i, j] = min(max(alpha[i, j] + dt * flux_comp, 0.0), 1.0)

    return alpha_new


@njit(cache=True)
def compute_csf_force(alpha, sigma, r_centers, dr, dz, nr, nz):
    f_r = np.zeros((nr, nz))
    f_z = np.zeros((nr, nz))

    grad_a_r = np.zeros((nr, nz))
    grad_a_z = np.zeros((nr, nz))
    for i in range(nr):
        for j in range(nz):
            if i > 0 and i < nr - 1:
                grad_a_r[i, j] = (alpha[i + 1, j] - alpha[i - 1, j]) / (2.0 * dr)
            elif i == 0 and nr > 1:
                grad_a_r[i, j] = (alpha[1, j] - alpha[0, j]) / dr
            elif i == nr - 1:
                grad_a_r[i, j] = (alpha[i, j] - alpha[i - 1, j]) / dr

            if j > 0 and j < nz - 1:
                grad_a_z[i, j] = (alpha[i, j + 1] - alpha[i, j - 1]) / (2.0 * dz)
            elif j == 0 and nz > 1:
                grad_a_z[i, j] = (alpha[i, j + 1] - alpha[i, j]) / dz
            elif j == nz - 1:
                grad_a_z[i, j] = (alpha[i, j] - alpha[i, j - 1]) / dz

    for i in range(nr):
        for j in range(nz):
            mag = np.sqrt(grad_a_r[i, j]**2 + grad_a_z[i, j]**2)
            if mag < 1e-10:
                continue

            n_r = grad_a_r[i, j] / mag
            n_z = grad_a_z[i, j] / mag

            dn_r_dr = 0.0
            if i > 0 and i < nr - 1:
                mag_p = np.sqrt(grad_a_r[i + 1, j]**2 + grad_a_z[i + 1, j]**2)
                mag_m = np.sqrt(grad_a_r[i - 1, j]**2 + grad_a_z[i - 1, j]**2)
                n_r_p = grad_a_r[i + 1, j] / max(mag_p, 1e-10)
                n_r_m = grad_a_r[i - 1, j] / max(mag_m, 1e-10)
                dn_r_dr = (n_r_p - n_r_m) / (2.0 * dr)

            dn_z_dz = 0.0
            if j > 0 and j < nz - 1:
                mag_p = np.sqrt(grad_a_r[i, j + 1]**2 + grad_a_z[i, j + 1]**2)
                mag_m = np.sqrt(grad_a_r[i, j - 1]**2 + grad_a_z[i, j - 1]**2)
                n_z_p = grad_a_z[i, j + 1] / max(mag_p, 1e-10)
                n_z_m = grad_a_z[i, j - 1] / max(mag_m, 1e-10)
                dn_z_dz = (n_z_p - n_z_m) / (2.0 * dz)

            r_c = max(r_centers[i], 1e-12)
            kappa = -(dn_r_dr + n_r / r_c + dn_z_dz)

            f_r[i, j] = sigma * kappa * grad_a_r[i, j]
            f_z[i, j] = sigma * kappa * grad_a_z[i, j]

    return f_r, f_z


@njit(cache=True)
def advect_temperature_axisym(T, ur, uz, rho, cp, k, r_centers, r_faces,
                               dr, dz, nr, nz, dt, h_conv, T_amb, alpha):
    T_new = T.copy()

    for i in range(nr):
        for j in range(nz):
            r_c = max(r_centers[i], 1e-12)
            r_e = r_faces[i + 1]
            r_w = r_faces[i]
            rho_cp = max(rho[i, j] * cp[i, j], 1e-10)
            V = np.pi * (r_e**2 - r_w**2) * dz

            A_e = 2.0 * np.pi * r_e * dz
            A_w = 2.0 * np.pi * r_w * dz
            A_z = np.pi * (r_e**2 - r_w**2)

            conv = 0.0
            if i < nr - 1:
                ur_e = 0.5 * (ur[i, j] + ur[i + 1, j])
                rho_e = 0.5 * (rho[i, j] + rho[i + 1, j])
                cp_e = 0.5 * (cp[i, j] + cp[i + 1, j])
                mdot_e = rho_e * cp_e * ur_e * A_e
                if ur_e >= 0:
                    conv += mdot_e * T[i, j]
                else:
                    conv += mdot_e * T[i + 1, j]
            if i > 0:
                ur_w = 0.5 * (ur[i - 1, j] + ur[i, j])
                rho_w = 0.5 * (rho[i - 1, j] + rho[i, j])
                cp_w = 0.5 * (cp[i - 1, j] + cp[i, j])
                mdot_w = rho_w * cp_w * ur_w * A_w
                if ur_w >= 0:
                    conv -= mdot_w * T[i - 1, j]
                else:
                    conv -= mdot_w * T[i, j]

            if j < nz - 1:
                uz_n = 0.5 * (uz[i, j] + uz[i, j + 1])
                rho_n = 0.5 * (rho[i, j] + rho[i, j + 1])
                cp_n = 0.5 * (cp[i, j] + cp[i, j + 1])
                mdot_n = rho_n * cp_n * uz_n * A_z
                if uz_n >= 0:
                    conv += mdot_n * T[i, j]
                else:
                    conv += mdot_n * T[i, j + 1]
            else:
                conv += rho[i, j] * cp[i, j] * uz[i, j] * A_z * T[i, j]

            if j > 0:
                uz_s = 0.5 * (uz[i, j - 1] + uz[i, j])
                rho_s = 0.5 * (rho[i, j - 1] + rho[i, j])
                cp_s = 0.5 * (cp[i, j - 1] + cp[i, j])
                mdot_s = rho_s * cp_s * uz_s * A_z
                if uz_s >= 0:
                    conv -= mdot_s * T[i, j - 1]
                else:
                    conv -= mdot_s * T[i, j]

            diff = 0.0
            if i < nr - 1:
                k_e = 0.5 * (k[i, j] + k[i + 1, j])
                diff += k_e * A_e * (T[i + 1, j] - T[i, j]) / dr
            if i > 0:
                k_w = 0.5 * (k[i - 1, j] + k[i, j])
                diff -= k_w * A_w * (T[i, j] - T[i - 1, j]) / dr

            if j < nz - 1:
                k_n = 0.5 * (k[i, j] + k[i, j + 1])
                diff += k_n * A_z * (T[i, j + 1] - T[i, j]) / dz
            if j > 0:
                k_s = 0.5 * (k[i, j - 1] + k[i, j])
                diff -= k_s * A_z * (T[i, j] - T[i, j - 1]) / dz

            cooling = 0.0
            if 0.01 < alpha[i, j] < 0.99:
                cooling = h_conv * (T[i, j] - T_amb) * V

            dTdt = (-conv + diff - cooling) / (rho_cp * V)
            T_new[i, j] = T[i, j] + dt * dTdt

    return T_new


class CFDSolver:
    def __init__(self, config=None):
        c = config or {}
        self.nozzle_diameter = c.get('nozzle_diameter', 0.4e-3)
        self.nozzle_length = c.get('nozzle_length', 2.0e-3)
        self.nozzle_radius = self.nozzle_diameter / 2.0

        self.domain_r = c.get('domain_r', self.nozzle_radius * 3.0)
        self.domain_z_min = -self.nozzle_length
        self.domain_z_max = c.get('domain_z_ext', self.nozzle_length * 1.5)

        self.nr = c.get('nr', 30)
        self.nz = c.get('nz', 60)
        self.dt = c.get('dt', 1e-6)
        self.gravity = c.get('gravity', -9.81)
        self.cfl_max = c.get('cfl_max', 0.5)

        self.flow_rate = c.get('flow_rate', 5e-9)
        self.T_nozzle = c.get('T_nozzle', 493.15)
        self.T_ambient = c.get('T_ambient', 298.15)
        self.h_conv = c.get('h_conv', 10.0)

        self.c_alpha = c.get('c_alpha', 1.0)
        self.n_piso = c.get('n_piso', 2)
        self.p_relax = c.get('p_relax', 0.3)
        self.u_relax = c.get('u_relax', 0.7)

        from solver.materials import TPUMaterial
        self.material = TPUMaterial(c.get('material', {}))

        from solver.grid import AxiSymGrid
        self.grid = AxiSymGrid(self.nr, self.nz, self.domain_r,
                               self.domain_z_min, self.domain_z_max)

        self._init_fields()

        self.step_count = 0
        self.time = 0.0
        self.diagnostics = []
        self.running = False
        self.paused = False

    def _init_fields(self):
        nr, nz = self.nr, self.nz
        self.ur = np.zeros((nr, nz))
        self.uz = np.zeros((nr, nz))
        self.p = np.zeros((nr, nz))
        self.T = np.full((nr, nz), self.T_ambient)
        self.alpha = np.zeros((nr, nz))

        g = self.grid
        nozzle_r_idx = np.searchsorted(g.r_faces, self.nozzle_radius)
        nozzle_z_idx = np.searchsorted(g.z_centers, 0.0)

        for i in range(min(nozzle_r_idx, nr)):
            for j in range(min(nozzle_z_idx, nz)):
                self.alpha[i, j] = 1.0
                self.T[i, j] = self.T_nozzle

        inlet_area = np.pi * self.nozzle_radius**2
        self.u_inlet = self.flow_rate / max(inlet_area, 1e-20)

        for i in range(min(nozzle_r_idx, nr)):
            r = g.r_centers[i]
            R = self.nozzle_radius
            self.uz[i, :nozzle_z_idx] = 2.0 * self.u_inlet * (1.0 - (r / R)**2)

        self.rho = np.zeros((nr, nz))
        self.cp = np.zeros((nr, nz))
        self.k_field = np.zeros((nr, nz))
        self.eta = np.zeros((nr, nz))
        self._update_properties()

    def _update_properties(self):
        m = self.material
        for i in range(self.nr):
            for j in range(self.nz):
                a = self.alpha[i, j]
                self.rho[i, j] = a * m.rho_polymer + (1.0 - a) * m.rho_air
                self.cp[i, j] = a * m.cp_polymer + (1.0 - a) * m.cp_air
                self.k_field[i, j] = a * m.k_polymer + (1.0 - a) * m.k_air

    def _apply_boundary_conditions(self):
        g = self.grid
        nozzle_r_idx = min(np.searchsorted(g.r_faces, self.nozzle_radius), self.nr)
        nozzle_z_end = np.searchsorted(g.z_centers, 0.0)

        for i in range(nozzle_r_idx):
            r = g.r_centers[i]
            R = self.nozzle_radius
            profile = 2.0 * self.u_inlet * (1.0 - (r / R)**2)
            self.uz[i, 0] = profile
            self.ur[i, 0] = 0.0
            self.T[i, 0] = self.T_nozzle
            self.alpha[i, 0] = 1.0

        for j in range(0, nozzle_z_end):
            if nozzle_r_idx < self.nr:
                self.ur[nozzle_r_idx - 1, j] = 0.0
                self.uz[nozzle_r_idx - 1, j] = 0.0
                self.T[nozzle_r_idx - 1, j] = self.T_nozzle

        self.ur[0, :] = 0.0

        self.ur[-1, :] = 0.0
        self.uz[-1, :] = 0.0

        self.p[:, -1] = 0.0

        for i in range(self.nr):
            self.uz[i, -1] = self.uz[i, -2] if self.nz > 1 else 0.0
            self.T[i, -1] = self.T[i, -2] if self.nz > 1 else self.T_ambient

    def _compute_adaptive_dt(self):
        g = self.grid
        u_max_r = np.max(np.abs(self.ur))
        u_max_z = np.max(np.abs(self.uz))

        dt_cfl = self.dt
        if u_max_r > 1e-10:
            dt_cfl = min(dt_cfl, self.cfl_max * g.dr / u_max_r)
        if u_max_z > 1e-10:
            dt_cfl = min(dt_cfl, self.cfl_max * g.dz / u_max_z)

        dt_cap = self.dt
        rho_min = np.min(self.rho[self.rho > 0]) if np.any(self.rho > 0) else 1.0
        if self.material.sigma > 0:
            dt_cap = np.sqrt(rho_min * min(g.dr, g.dz)**3 /
                           (2.0 * np.pi * self.material.sigma))

        dt_adapt = min(dt_cfl, dt_cap, self.dt)
        return max(dt_adapt, self.dt * 0.01)

    def _solve_pressure_poisson(self, div_star, dt_sub):
        nr, nz = self.nr, self.nz
        g = self.grid
        p = self.p.copy()

        for iteration in range(100):
            p_old = p.copy()
            max_resid = 0.0

            for i in range(nr):
                for j in range(nz):
                    if j == nz - 1:
                        p[i, j] = 0.0
                        continue

                    r_c = max(g.r_centers[i], 1e-12)
                    rhs = 0.0
                    diag = 0.0

                    if i < nr - 1:
                        r_e = g.r_faces[i + 1]
                        coeff = r_e / (r_c * g.dr**2)
                        rhs += coeff * p_old[i + 1, j]
                        diag += coeff
                    if i > 0:
                        r_w = g.r_faces[i]
                        coeff = r_w / (r_c * g.dr**2)
                        rhs += coeff * p_old[i - 1, j]
                        diag += coeff

                    if j < nz - 1:
                        coeff = 1.0 / g.dz**2
                        rhs += coeff * p_old[i, j + 1]
                        diag += coeff
                    if j > 0:
                        coeff = 1.0 / g.dz**2
                        rhs += coeff * p_old[i, j - 1]
                        diag += coeff

                    if diag < 1e-30:
                        continue

                    rho_c = max(self.rho[i, j], 1e-10)
                    source = rho_c * div_star[i, j] / dt_sub

                    p_new = (rhs - source) / diag
                    p[i, j] = 0.7 * p_new + 0.3 * p_old[i, j]

                    resid = abs(p[i, j] - p_old[i, j])
                    if resid > max_resid:
                        max_resid = resid

            if max_resid < 1e-6 * max(np.max(np.abs(p)), 1.0):
                break

        return p

    def step(self):
        g = self.grid
        nr, nz = self.nr, self.nz
        m = self.material

        self._apply_boundary_conditions()
        self._update_properties()

        dur_dr, dur_dz, duz_dr, duz_dz = compute_velocity_gradients(
            self.ur, self.uz, nr, nz, g.dr, g.dz)
        gamma_dot = compute_shear_rate_field(
            dur_dr, dur_dz, duz_dr, duz_dz, self.ur, g.r_centers, nr, nz)

        params = m.get_params_tuple()
        self.eta = compute_viscosity_field(
            gamma_dot, self.T, self.alpha, nr, nz,
            params[0], params[1], params[2], params[3], params[4],
            params[7], params[8], params[9], params[5], params[6], params[17])

        dt = self._compute_adaptive_dt()

        f_st_r, f_st_z = compute_csf_force(
            self.alpha, m.sigma, g.r_centers, g.dr, g.dz, nr, nz)

        rhs_ur, rhs_uz = compute_momentum_rhs_axisym(
            self.ur, self.uz, self.rho, self.eta, self.alpha,
            g.r_centers, g.r_faces, g.dr, g.dz, nr, nz,
            self.gravity, f_st_r, f_st_z)

        ur_star = np.zeros_like(self.ur)
        uz_star = np.zeros_like(self.uz)
        for i in range(nr):
            for j in range(nz):
                rho_ij = max(self.rho[i, j], 1e-10)
                ur_star[i, j] = self.ur[i, j] + dt * rhs_ur[i, j] / rho_ij
                uz_star[i, j] = self.uz[i, j] + dt * rhs_uz[i, j] / rho_ij

        for piso_iter in range(self.n_piso):
            div_star = compute_divergence_axisym(
                ur_star, uz_star, g.r_centers, g.r_faces, g.dr, g.dz, nr, nz)

            p_corr = self._solve_pressure_poisson(div_star, dt)

            np.nan_to_num(p_corr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            for i in range(nr):
                for j in range(nz):
                    rho_ij = max(self.rho[i, j], 1e-10)

                    dp_dr = 0.0
                    if i > 0 and i < nr - 1:
                        dp_dr = (p_corr[i + 1, j] - p_corr[i - 1, j]) / (2.0 * g.dr)
                    elif i == 0 and nr > 1:
                        dp_dr = (p_corr[1, j] - p_corr[0, j]) / g.dr
                    elif i == nr - 1:
                        dp_dr = (p_corr[i, j] - p_corr[i - 1, j]) / g.dr

                    dp_dz = 0.0
                    if j > 0 and j < nz - 1:
                        dp_dz = (p_corr[i, j + 1] - p_corr[i, j - 1]) / (2.0 * g.dz)
                    elif j == 0 and nz > 1:
                        dp_dz = (p_corr[i, j + 1] - p_corr[i, j]) / g.dz

                    ur_star[i, j] -= dt * dp_dr / rho_ij
                    uz_star[i, j] -= dt * dp_dz / rho_ij

            self.p += self.p_relax * p_corr

        self.ur = self.u_relax * ur_star + (1.0 - self.u_relax) * self.ur
        self.uz = self.u_relax * uz_star + (1.0 - self.u_relax) * self.uz

        np.nan_to_num(self.ur, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(self.uz, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(self.p, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        u_max_phys = max(self.u_inlet * 10.0, 1.0)
        np.clip(self.ur, -u_max_phys, u_max_phys, out=self.ur)
        np.clip(self.uz, -u_max_phys, u_max_phys, out=self.uz)
        np.clip(self.p, -1e7, 1e7, out=self.p)

        self.alpha = advect_vof_axisym(
            self.alpha, self.ur, self.uz,
            g.r_centers, g.r_faces, g.dr, g.dz, nr, nz, dt)

        if self.c_alpha > 0:
            self.alpha = apply_vof_compression(
                self.alpha, self.ur, self.uz,
                g.r_centers, g.r_faces, g.dr, g.dz, nr, nz, dt, self.c_alpha)

        self.T = advect_temperature_axisym(
            self.T, self.ur, self.uz, self.rho, self.cp, self.k_field,
            g.r_centers, g.r_faces, g.dr, g.dz, nr, nz, dt,
            self.h_conv, self.T_ambient, self.alpha)

        np.nan_to_num(self.T, copy=False, nan=self.T_ambient, posinf=self.T_ambient, neginf=self.T_ambient)
        np.clip(self.T, 200.0, 600.0, out=self.T)

        self._apply_boundary_conditions()
        self.step_count += 1
        self.time += dt

        diag = self._compute_diagnostics(dt)
        self.diagnostics.append(diag)
        return diag

    def _compute_diagnostics(self, dt_used=None):
        g = self.grid
        mass_polymer = np.sum(self.alpha * self.material.rho_polymer * g.cell_volumes)

        u_mag = np.sqrt(self.ur**2 + self.uz**2)
        u_max_val = np.max(u_mag)
        cfl_r = np.max(np.abs(self.ur)) * (dt_used or self.dt) / g.dr if g.dr > 0 else 0
        cfl_z = np.max(np.abs(self.uz)) * (dt_used or self.dt) / g.dz if g.dz > 0 else 0
        cfl = max(cfl_r, cfl_z)

        rho_min = np.min(self.rho[self.rho > 0]) if np.any(self.rho > 0) else 1.0
        if rho_min > 0 and g.dr > 0 and self.material.sigma > 0:
            cap_dt = np.sqrt(rho_min * min(g.dr, g.dz)**3 /
                            (2.0 * np.pi * self.material.sigma))
        else:
            cap_dt = 1e10

        nozzle_z_end = np.searchsorted(g.z_centers, 0.0)

        swell_ratios = []
        for j in range(nozzle_z_end, min(nozzle_z_end + 20, self.nz)):
            for i in range(self.nr - 1, -1, -1):
                if self.alpha[i, j] > 0.5:
                    swell_ratios.append(g.r_centers[i] / self.nozzle_radius)
                    break
            else:
                swell_ratios.append(0.0)

        p_inlet = np.mean(self.p[:, 0])
        p_outlet = np.mean(self.p[:, -1])

        centerline_T = self.T[0, :].tolist()

        result = {
            'step': self.step_count,
            'time': self.time,
            'dt_used': dt_used or self.dt,
            'mass_polymer': float(mass_polymer),
            'cfl': float(cfl),
            'capillary_dt': float(cap_dt),
            'dt': self.dt,
            'alpha_min': float(np.min(self.alpha)),
            'alpha_max': float(np.max(self.alpha)),
            'T_min': float(np.min(self.T)),
            'T_max': float(np.max(self.T)),
            'eta_min': float(np.min(self.eta)),
            'eta_max': float(np.max(self.eta)),
            'u_max': float(u_max_val),
            'p_min': float(np.min(self.p)),
            'p_max': float(np.max(self.p)),
            'pressure_drop': float(p_inlet - p_outlet),
            'swell_ratios': swell_ratios,
            'centerline_T': centerline_T,
        }
        return sanitize_dict(result)

    def get_frame_data(self):
        u_mag = np.sqrt(self.ur**2 + self.uz**2)
        g = self.grid

        contour_r = []
        contour_z = []
        for j in range(self.nz):
            for i in range(self.nr - 1):
                a1 = self.alpha[i, j]
                a2 = self.alpha[i + 1, j]
                if (a1 - 0.5) * (a2 - 0.5) < 0:
                    t = (0.5 - a1) / (a2 - a1) if abs(a2 - a1) > 1e-10 else 0.5
                    r_interp = g.r_centers[i] + t * g.dr
                    contour_r.append(float(r_interp))
                    contour_z.append(float(g.z_centers[j]))

        result = {
            'alpha': self.alpha.tolist(),
            'u_mag': u_mag.tolist(),
            'T': self.T.tolist(),
            'p': self.p.tolist(),
            'eta': self.eta.tolist(),
            'ur': self.ur.tolist(),
            'uz': self.uz.tolist(),
            'contour_r': contour_r,
            'contour_z': contour_z,
            'r_centers': g.r_centers.tolist(),
            'z_centers': g.z_centers.tolist(),
            'nr': self.nr,
            'nz': self.nz,
            'nozzle_radius': self.nozzle_radius,
            'nozzle_z_end': float(0.0),
        }
        return sanitize_dict(result)

    def reset(self, config=None):
        if config:
            self.__init__(config)
        else:
            self._init_fields()
            self.step_count = 0
            self.time = 0.0
            self.diagnostics = []
