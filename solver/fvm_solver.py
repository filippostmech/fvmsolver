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
def compute_velocity_gradients(ur, uz, nr, nz, r_centers, z_centers):
    dur_dr = np.zeros((nr, nz))
    dur_dz = np.zeros((nr, nz))
    duz_dr = np.zeros((nr, nz))
    duz_dz = np.zeros((nr, nz))

    for i in range(nr):
        for j in range(nz):
            if i == 0:
                if nr > 1:
                    dr_f = r_centers[1] - r_centers[0]
                    dur_dr[i, j] = (ur[1, j] - ur[0, j]) / dr_f
                    duz_dr[i, j] = (uz[1, j] - uz[0, j]) / dr_f
            elif i == nr - 1:
                dr_b = r_centers[i] - r_centers[i - 1]
                dur_dr[i, j] = (ur[i, j] - ur[i - 1, j]) / dr_b
                duz_dr[i, j] = (uz[i, j] - uz[i - 1, j]) / dr_b
            else:
                dr_c = r_centers[i + 1] - r_centers[i - 1]
                dur_dr[i, j] = (ur[i + 1, j] - ur[i - 1, j]) / dr_c
                duz_dr[i, j] = (uz[i + 1, j] - uz[i - 1, j]) / dr_c

            if j == 0:
                if nz > 1:
                    dz_f = z_centers[1] - z_centers[0]
                    dur_dz[i, j] = (ur[i, 1] - ur[i, 0]) / dz_f
                    duz_dz[i, j] = (uz[i, 1] - uz[i, 0]) / dz_f
            elif j == nz - 1:
                dz_b = z_centers[j] - z_centers[j - 1]
                dur_dz[i, j] = (ur[i, j] - ur[i, j - 1]) / dz_b
                duz_dz[i, j] = (uz[i, j] - uz[i, j - 1]) / dz_b
            else:
                dz_c = z_centers[j + 1] - z_centers[j - 1]
                dur_dz[i, j] = (ur[i, j + 1] - ur[i, j - 1]) / dz_c
                duz_dz[i, j] = (uz[i, j + 1] - uz[i, j - 1]) / dz_c

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
def solve_simple_momentum(ur, uz, ur_old, uz_old, rho, eta, alpha, p_old,
                          r_centers, r_faces, dz_arr, nr, nz,
                          gravity, f_st_r, f_st_z, dt, rho_floor):
    ur_star = ur.copy()
    uz_star = uz.copy()
    a_P_r = np.zeros((nr, nz))
    a_P_z = np.zeros((nr, nz))
    alpha_thresh = 0.01

    for gs_iter in range(30):
        max_resid = 0.0
        for i in range(nr):
            for j in range(nz):
                if alpha[i, j] < alpha_thresh:
                    ur_star[i, j] = 0.0
                    uz_star[i, j] = 0.0
                    a_P_r[i, j] = 1.0
                    a_P_z[i, j] = 1.0
                    continue

                r_c = max(r_centers[i], 1e-12)
                rho_c = max(rho[i, j], rho_floor)
                r_e = r_faces[i + 1]
                r_w = r_faces[i]
                dz_j = dz_arr[j]
                V = np.pi * (r_e**2 - r_w**2) * dz_j
                A_e = 2.0 * np.pi * r_e * dz_j
                A_w = 2.0 * np.pi * r_w * dz_j
                A_n = np.pi * (r_e**2 - r_w**2)
                A_s = A_n

                a_diag_r = rho_c * V / dt
                a_diag_z = rho_c * V / dt
                b_r = rho_c * V / dt * ur_old[i, j]
                b_z = rho_c * V / dt * uz_old[i, j]

                b_z += rho_c * gravity * V
                b_r += f_st_r[i, j] * V
                b_z += f_st_z[i, j] * V

                dp_dr = 0.0
                if i > 0 and i < nr - 1:
                    dp_dr = (p_old[i + 1, j] - p_old[i - 1, j]) / (r_centers[i + 1] - r_centers[i - 1])
                elif i == 0 and nr > 1:
                    dp_dr = (p_old[1, j] - p_old[0, j]) / (r_centers[1] - r_centers[0])
                elif i == nr - 1:
                    dp_dr = (p_old[i, j] - p_old[i - 1, j]) / (r_centers[i] - r_centers[i - 1])
                dp_dz = 0.0
                if j > 0 and j < nz - 1:
                    dp_dz = (p_old[i, j + 1] - p_old[i, j - 1]) / (dz_arr[j] + 0.5 * (dz_arr[j - 1] + dz_arr[j + 1]))
                elif j == 0 and nz > 1:
                    dp_dz = (p_old[i, j + 1] - p_old[i, j]) / (0.5 * (dz_arr[j] + dz_arr[j + 1]))
                elif j == nz - 1 and nz > 1:
                    dp_dz = (p_old[i, j] - p_old[i, j - 1]) / (0.5 * (dz_arr[j - 1] + dz_arr[j]))
                b_r -= dp_dr * V
                b_z -= dp_dz * V

                if i < nr - 1 and alpha[i + 1, j] >= alpha_thresh:
                    ur_e = 0.5 * (ur_star[i, j] + ur_star[i + 1, j])
                    rho_e = 0.5 * (rho[i, j] + rho[i + 1, j])
                    mdot_e = rho_e * ur_e * A_e
                    F_e = max(mdot_e, 0.0)
                    eta_e = 0.5 * (eta[i, j] + eta[i + 1, j])
                    dr_e = r_centers[i + 1] - r_centers[i]
                    D_e = eta_e * A_e / dr_e
                    a_diag_r += F_e + D_e
                    a_diag_z += F_e + D_e
                    a_nb_r = max(-mdot_e, 0.0) + D_e
                    a_nb_z = a_nb_r
                    b_r += a_nb_r * ur_star[i + 1, j]
                    b_z += a_nb_z * uz_star[i + 1, j]

                if i > 0 and alpha[i - 1, j] >= alpha_thresh:
                    ur_w = 0.5 * (ur_star[i - 1, j] + ur_star[i, j])
                    rho_w = 0.5 * (rho[i - 1, j] + rho[i, j])
                    mdot_w = rho_w * ur_w * A_w
                    F_w = max(-mdot_w, 0.0)
                    eta_w = 0.5 * (eta[i - 1, j] + eta[i, j])
                    dr_w = r_centers[i] - r_centers[i - 1]
                    D_w = eta_w * A_w / dr_w
                    a_diag_r += F_w + D_w
                    a_diag_z += F_w + D_w
                    a_nb_r = max(mdot_w, 0.0) + D_w
                    a_nb_z = a_nb_r
                    b_r += a_nb_r * ur_star[i - 1, j]
                    b_z += a_nb_z * uz_star[i - 1, j]
                elif i == 0:
                    dr_half = r_centers[0]
                    D_w_sym = eta[i, j] * A_w / max(dr_half, 1e-12)
                    a_diag_r += D_w_sym

                if j < nz - 1 and alpha[i, j + 1] >= alpha_thresh:
                    uz_n = 0.5 * (uz_star[i, j] + uz_star[i, j + 1])
                    rho_n = 0.5 * (rho[i, j] + rho[i, j + 1])
                    mdot_n = rho_n * uz_n * A_n
                    F_n = max(mdot_n, 0.0)
                    eta_n = 0.5 * (eta[i, j] + eta[i, j + 1])
                    dz_n = 0.5 * (dz_arr[j] + dz_arr[j + 1])
                    D_n = eta_n * A_n / dz_n
                    a_diag_r += F_n + D_n
                    a_diag_z += F_n + D_n
                    a_nb_r = max(-mdot_n, 0.0) + D_n
                    a_nb_z = a_nb_r
                    b_r += a_nb_r * ur_star[i, j + 1]
                    b_z += a_nb_z * uz_star[i, j + 1]

                if j > 0 and alpha[i, j - 1] >= alpha_thresh:
                    uz_s = 0.5 * (uz_star[i, j - 1] + uz_star[i, j])
                    rho_s = 0.5 * (rho[i, j - 1] + rho[i, j])
                    mdot_s = rho_s * uz_s * A_s
                    F_s = max(-mdot_s, 0.0)
                    eta_s = 0.5 * (eta[i, j - 1] + eta[i, j])
                    dz_s = 0.5 * (dz_arr[j - 1] + dz_arr[j])
                    D_s = eta_s * A_s / dz_s
                    a_diag_r += F_s + D_s
                    a_diag_z += F_s + D_s
                    a_nb_r = max(mdot_s, 0.0) + D_s
                    a_nb_z = a_nb_r
                    b_r += a_nb_r * ur_star[i, j - 1]
                    b_z += a_nb_z * uz_star[i, j - 1]

                hoop = eta[i, j] / (r_c**2) * V
                a_diag_r += hoop

                if a_diag_r > 1e-30:
                    ur_new = b_r / a_diag_r
                    resid = abs(ur_new - ur_star[i, j])
                    if resid > max_resid:
                        max_resid = resid
                    ur_star[i, j] = 0.8 * ur_new + 0.2 * ur_star[i, j]
                if a_diag_z > 1e-30:
                    uz_new = b_z / a_diag_z
                    resid = abs(uz_new - uz_star[i, j])
                    if resid > max_resid:
                        max_resid = resid
                    uz_star[i, j] = 0.8 * uz_new + 0.2 * uz_star[i, j]

                a_P_r[i, j] = a_diag_r
                a_P_z[i, j] = a_diag_z

        if max_resid < 1e-6:
            break

    return ur_star, uz_star, a_P_z


@njit(cache=True)
def compute_divergence_axisym(ur, uz, r_centers, r_faces, dr_arr, dz_arr, nr, nz):
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

            div_r = (r_e * ur_e - r_w * ur_w) / (r_c * dr_arr[i])

            if j < nz - 1:
                uz_n = 0.5 * (uz[i, j] + uz[i, j + 1])
            else:
                uz_n = uz[i, j]

            if j > 0:
                uz_s = 0.5 * (uz[i, j - 1] + uz[i, j])
            else:
                uz_s = uz[i, j]

            div_z = (uz_n - uz_s) / dz_arr[j]
            div[i, j] = div_r + div_z

    return div


@njit(cache=True)
def advect_vof_axisym(alpha, ur, uz, r_centers, r_faces, dz_arr, nr, nz, dt):
    alpha_new = alpha.copy()

    for i in range(nr):
        for j in range(nz):
            r_c = max(r_centers[i], 1e-12)
            r_e = r_faces[i + 1]
            r_w = r_faces[i]
            dz_j = dz_arr[j]
            V = np.pi * (r_e**2 - r_w**2) * dz_j

            flux_e = 0.0
            if i < nr - 1:
                ur_e = 0.5 * (ur[i, j] + ur[i + 1, j])
                A_e = 2.0 * np.pi * r_e * dz_j
                if ur_e >= 0:
                    flux_e = ur_e * alpha[i, j] * A_e
                else:
                    flux_e = ur_e * alpha[i + 1, j] * A_e

            flux_w = 0.0
            if i > 0:
                ur_w = 0.5 * (ur[i - 1, j] + ur[i, j])
                A_w = 2.0 * np.pi * r_w * dz_j
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
def apply_vof_compression(alpha, ur, uz, r_centers, r_faces, dz_arr, nr, nz, dt, c_alpha):
    alpha_new = alpha.copy()

    for i in range(nr):
        for j in range(nz):
            r_c = max(r_centers[i], 1e-12)
            dr_i = r_faces[i + 1] - r_faces[i]

            grad_a_r = 0.0
            grad_a_z = 0.0
            if i > 0 and i < nr - 1:
                grad_a_r = (alpha[i + 1, j] - alpha[i - 1, j]) / (r_centers[i + 1] - r_centers[i - 1])
            elif i == 0 and nr > 1:
                grad_a_r = (alpha[1, j] - alpha[0, j]) / (r_centers[1] - r_centers[0])
            elif i == nr - 1:
                grad_a_r = (alpha[i, j] - alpha[i - 1, j]) / (r_centers[i] - r_centers[i - 1])

            if j > 0 and j < nz - 1:
                grad_a_z = (alpha[i, j + 1] - alpha[i, j - 1]) / (0.5 * (dz_arr[j - 1] + dz_arr[j]) + 0.5 * (dz_arr[j] + dz_arr[j + 1]))
            elif j == 0 and nz > 1:
                grad_a_z = (alpha[i, j + 1] - alpha[i, j]) / (0.5 * (dz_arr[j] + dz_arr[j + 1]))
            elif j == nz - 1:
                grad_a_z = (alpha[i, j] - alpha[i, j - 1]) / (0.5 * (dz_arr[j - 1] + dz_arr[j]))

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
                flux_comp += u_comp_r * comp_e * r_faces[i + 1] / (r_c * dr_i)
            if i > 0:
                comp_w = 0.5 * (alpha[i - 1, j] * (1.0 - alpha[i - 1, j]) + comp_term)
                flux_comp -= u_comp_r * comp_w * r_faces[i] / (r_c * dr_i)

            dz_j = dz_arr[j]
            if j < nz - 1:
                comp_n = 0.5 * (comp_term + alpha[i, j + 1] * (1.0 - alpha[i, j + 1]))
                flux_comp += u_comp_z * comp_n / dz_j
            if j > 0:
                comp_s = 0.5 * (alpha[i, j - 1] * (1.0 - alpha[i, j - 1]) + comp_term)
                flux_comp -= u_comp_z * comp_s / dz_j

            alpha_new[i, j] = min(max(alpha[i, j] + dt * flux_comp, 0.0), 1.0)

    return alpha_new


@njit(cache=True)
def compute_csf_force(alpha, sigma, r_centers, r_faces, dz_arr, nr, nz):
    f_r = np.zeros((nr, nz))
    f_z = np.zeros((nr, nz))

    grad_a_r = np.zeros((nr, nz))
    grad_a_z = np.zeros((nr, nz))
    for i in range(nr):
        for j in range(nz):
            if i > 0 and i < nr - 1:
                grad_a_r[i, j] = (alpha[i + 1, j] - alpha[i - 1, j]) / (r_centers[i + 1] - r_centers[i - 1])
            elif i == 0 and nr > 1:
                grad_a_r[i, j] = (alpha[1, j] - alpha[0, j]) / (r_centers[1] - r_centers[0])
            elif i == nr - 1:
                grad_a_r[i, j] = (alpha[i, j] - alpha[i - 1, j]) / (r_centers[i] - r_centers[i - 1])

            if j > 0 and j < nz - 1:
                dz_c = 0.5 * (dz_arr[j - 1] + dz_arr[j]) + 0.5 * (dz_arr[j] + dz_arr[j + 1])
                grad_a_z[i, j] = (alpha[i, j + 1] - alpha[i, j - 1]) / dz_c
            elif j == 0 and nz > 1:
                grad_a_z[i, j] = (alpha[i, j + 1] - alpha[i, j]) / (0.5 * (dz_arr[j] + dz_arr[j + 1]))
            elif j == nz - 1:
                grad_a_z[i, j] = (alpha[i, j] - alpha[i, j - 1]) / (0.5 * (dz_arr[j - 1] + dz_arr[j]))

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
                dn_r_dr = (n_r_p - n_r_m) / (r_centers[i + 1] - r_centers[i - 1])

            dn_z_dz = 0.0
            if j > 0 and j < nz - 1:
                mag_p = np.sqrt(grad_a_r[i, j + 1]**2 + grad_a_z[i, j + 1]**2)
                mag_m = np.sqrt(grad_a_r[i, j - 1]**2 + grad_a_z[i, j - 1]**2)
                n_z_p = grad_a_z[i, j + 1] / max(mag_p, 1e-10)
                n_z_m = grad_a_z[i, j - 1] / max(mag_m, 1e-10)
                dz_c = 0.5 * (dz_arr[j - 1] + dz_arr[j]) + 0.5 * (dz_arr[j] + dz_arr[j + 1])
                dn_z_dz = (n_z_p - n_z_m) / dz_c

            r_c = max(r_centers[i], 1e-12)
            kappa = -(dn_r_dr + n_r / r_c + dn_z_dz)

            f_r[i, j] = sigma * kappa * grad_a_r[i, j]
            f_z[i, j] = sigma * kappa * grad_a_z[i, j]

    return f_r, f_z


@njit(cache=True)
def advect_temperature_axisym(T, ur, uz, rho, cp, k, r_centers, r_faces,
                               dz_arr, nr, nz, dt, h_conv, T_amb, alpha):
    T_new = T.copy()

    for i in range(nr):
        for j in range(nz):
            r_c = max(r_centers[i], 1e-12)
            r_e = r_faces[i + 1]
            r_w = r_faces[i]
            rho_cp = max(rho[i, j] * cp[i, j], 1e-10)
            dz_j = dz_arr[j]
            V = np.pi * (r_e**2 - r_w**2) * dz_j

            A_e = 2.0 * np.pi * r_e * dz_j
            A_w = 2.0 * np.pi * r_w * dz_j
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
                dr_e = r_centers[i + 1] - r_centers[i]
                diff += k_e * A_e * (T[i + 1, j] - T[i, j]) / dr_e
            if i > 0:
                k_w = 0.5 * (k[i - 1, j] + k[i, j])
                dr_w = r_centers[i] - r_centers[i - 1]
                diff -= k_w * A_w * (T[i, j] - T[i - 1, j]) / dr_w

            if j < nz - 1:
                k_n = 0.5 * (k[i, j] + k[i, j + 1])
                dz_n = 0.5 * (dz_arr[j] + dz_arr[j + 1])
                diff += k_n * A_z * (T[i, j + 1] - T[i, j]) / dz_n
            if j > 0:
                k_s = 0.5 * (k[i, j - 1] + k[i, j])
                dz_s = 0.5 * (dz_arr[j - 1] + dz_arr[j])
                diff -= k_s * A_z * (T[i, j] - T[i, j - 1]) / dz_s

            cooling = 0.0
            if 0.01 < alpha[i, j] < 0.99:
                cooling = h_conv * (T[i, j] - T_amb) * V

            dTdt = (-conv + diff - cooling) / (rho_cp * V)
            T_new[i, j] = T[i, j] + dt * dTdt

    return T_new


@njit(cache=True)
def _implicit_diffusion_solve(u, u_star, eta, rho, alpha, r_centers, r_faces, dz_arr, nr, nz, dt, rho_floor):
    omega = 1.2
    alpha_thresh = 0.01
    for iteration in range(50):
        max_resid = 0.0
        for i in range(nr):
            for j in range(nz):
                if alpha[i, j] < alpha_thresh:
                    continue

                r_c = max(r_centers[i], 1e-12)
                r_e = r_faces[i + 1]
                r_w = r_faces[i]
                dz_j = dz_arr[j]
                V = np.pi * (r_e**2 - r_w**2) * dz_j
                rho_ij = max(rho[i, j], rho_floor)
                A_e = 2.0 * np.pi * r_e * dz_j
                A_w = 2.0 * np.pi * r_w * dz_j
                A_n = np.pi * (r_e**2 - r_w**2)
                A_s = A_n

                diag = rho_ij * V / dt
                rhs = rho_ij * V / dt * u_star[i, j]

                if i < nr - 1 and alpha[i + 1, j] >= alpha_thresh:
                    eta_e = 0.5 * (eta[i, j] + eta[i + 1, j])
                    dr_e = r_centers[i + 1] - r_centers[i]
                    coeff = eta_e * A_e / dr_e
                    diag += coeff
                    rhs += coeff * u[i + 1, j]
                if i > 0 and alpha[i - 1, j] >= alpha_thresh:
                    eta_w = 0.5 * (eta[i - 1, j] + eta[i, j])
                    dr_w = r_centers[i] - r_centers[i - 1]
                    coeff = eta_w * A_w / dr_w
                    diag += coeff
                    rhs += coeff * u[i - 1, j]

                if j < nz - 1 and alpha[i, j + 1] >= alpha_thresh:
                    eta_n = 0.5 * (eta[i, j] + eta[i, j + 1])
                    dz_n = 0.5 * (dz_arr[j] + dz_arr[j + 1])
                    coeff = eta_n * A_n / dz_n
                    diag += coeff
                    rhs += coeff * u[i, j + 1]
                if j > 0 and alpha[i, j - 1] >= alpha_thresh:
                    eta_s = 0.5 * (eta[i, j - 1] + eta[i, j])
                    dz_s = 0.5 * (dz_arr[j - 1] + dz_arr[j])
                    coeff = eta_s * A_s / dz_s
                    diag += coeff
                    rhs += coeff * u[i, j - 1]

                if diag < 1e-30:
                    continue

                u_new = rhs / diag
                u_sor = u[i, j] + omega * (u_new - u[i, j])
                resid = abs(u_sor - u[i, j])
                if resid > max_resid:
                    max_resid = resid
                u[i, j] = u_sor

        if max_resid < 1e-8 * max(np.max(np.abs(u)), 1e-10):
            break

    return u


@njit(cache=True)
def _sor_pressure_solve(p, div_star, rho, r_centers, r_faces, dr_arr, dz_arr, nr, nz, dt_sub, rho_floor):
    omega = 1.5
    for iteration in range(500):
        max_resid = 0.0
        for i in range(nr):
            for j in range(nz):
                if i == nr - 1 and j == 0:
                    p[i, j] = 0.0
                    continue

                r_c = max(r_centers[i], 1e-12)
                rhs = 0.0
                diag = 0.0

                if i < nr - 1:
                    rho_e = 0.5 * (rho[i, j] + rho[i + 1, j])
                    rho_e = max(rho_e, rho_floor)
                    r_e = r_faces[i + 1]
                    dr_e = r_centers[i + 1] - r_centers[i]
                    coeff = r_e / (rho_e * r_c * dr_arr[i] * dr_e)
                    rhs += coeff * p[i + 1, j]
                    diag += coeff
                if i > 0:
                    rho_w = 0.5 * (rho[i - 1, j] + rho[i, j])
                    rho_w = max(rho_w, rho_floor)
                    r_w = r_faces[i]
                    dr_w = r_centers[i] - r_centers[i - 1]
                    coeff = r_w / (rho_w * r_c * dr_arr[i] * dr_w)
                    rhs += coeff * p[i - 1, j]
                    diag += coeff

                if j < nz - 1:
                    rho_n = 0.5 * (rho[i, j] + rho[i, j + 1])
                    rho_n = max(rho_n, rho_floor)
                    dz_n = 0.5 * (dz_arr[j] + dz_arr[j + 1])
                    coeff = 1.0 / (rho_n * dz_arr[j] * dz_n)
                    rhs += coeff * p[i, j + 1]
                    diag += coeff
                if j > 0:
                    rho_s = 0.5 * (rho[i, j - 1] + rho[i, j])
                    rho_s = max(rho_s, rho_floor)
                    dz_s = 0.5 * (dz_arr[j - 1] + dz_arr[j])
                    coeff = 1.0 / (rho_s * dz_arr[j] * dz_s)
                    rhs += coeff * p[i, j - 1]
                    diag += coeff

                if diag < 1e-30:
                    continue

                source = div_star[i, j] / dt_sub

                p_gs = (rhs - source) / diag
                p_new = p[i, j] + omega * (p_gs - p[i, j])

                resid = abs(p_new - p[i, j])
                if resid > max_resid:
                    max_resid = resid

                p[i, j] = p_new

        p_abs_max = 0.0
        for i in range(nr):
            for j in range(nz):
                if abs(p[i, j]) > p_abs_max:
                    p_abs_max = abs(p[i, j])
        if max_resid < 1e-6 * max(p_abs_max, 1.0):
            break

    return p


class CFDSolver:
    def __init__(self, config=None):
        c = config or {}
        self.nozzle_diameter = c.get('nozzle_diameter', 0.4e-3)
        self.nozzle_length = c.get('nozzle_length', 2.0e-3)
        self.nozzle_radius = self.nozzle_diameter / 2.0

        self.domain_r = c.get('domain_r', self.nozzle_radius * 3.0)
        self.domain_z_min = -c.get('domain_z_ext', self.nozzle_length * 1.5)
        self.domain_z_max = self.nozzle_length

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

        self.stretch_type = c.get('stretch_type', 'uniform')
        self.stretch_ratio = c.get('stretch_ratio', 1.5)

        from solver.materials import TPUMaterial
        self.material = TPUMaterial(c.get('material', {}))

        from solver.grid import AxiSymGrid
        self.grid = AxiSymGrid(self.nr, self.nz, self.domain_r,
                               self.domain_z_min, self.domain_z_max,
                               self.stretch_type, self.stretch_ratio)

        self._init_fields()

        self.step_count = 0
        self.time = 0.0
        self.diagnostics = []
        self.running = False
        self.paused = False

    def _init_fields(self):
        nr, nz = self.nr, self.nz
        g = self.grid
        self.ur = np.zeros((nr, nz))
        self.uz = np.zeros((nr, nz))
        self.p = np.zeros((nr, nz))
        self.T = np.full((nr, nz), self.T_ambient)
        self.alpha = np.zeros((nr, nz))

        inlet_area = np.pi * self.nozzle_radius**2
        self.u_inlet = self.flow_rate / max(inlet_area, 1e-20)

        nozzle_r_idx = min(np.searchsorted(g.r_faces, self.nozzle_radius), nr)
        nozzle_z_start = np.searchsorted(g.z_centers, 0.0)

        for j in range(nozzle_z_start, nz):
            for i in range(nozzle_r_idx):
                r = g.r_centers[i]
                R = self.nozzle_radius
                profile = 2.0 * self.u_inlet * (1.0 - (r / R)**2)
                self.uz[i, j] = -profile
                self.T[i, j] = self.T_nozzle
                self.alpha[i, j] = 1.0

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
        nozzle_z_start = np.searchsorted(g.z_centers, 0.0)

        for i in range(nozzle_r_idx):
            r = g.r_centers[i]
            R = self.nozzle_radius
            profile = 2.0 * self.u_inlet * (1.0 - (r / R)**2)
            self.uz[i, -1] = -profile
            self.ur[i, -1] = 0.0
            self.T[i, -1] = self.T_nozzle
            self.alpha[i, -1] = 1.0

        for j in range(nozzle_z_start, self.nz):
            if nozzle_r_idx < self.nr:
                self.ur[nozzle_r_idx, j] = 0.0
                self.uz[nozzle_r_idx, j] = 0.0
                self.T[nozzle_r_idx - 1, j] = self.T_nozzle

        self.ur[0, :] = 0.0

        self.ur[-1, :] = 0.0
        self.uz[-1, :] = 0.0

        for i in range(self.nr):
            self.uz[i, 0] = 0.0
            self.ur[i, 0] = 0.0
            self.T[i, 0] = self.T[i, 1] if self.nz > 1 else self.T_ambient

        self.p[-1, 0] = 0.0

    def _compute_adaptive_dt(self):
        g = self.grid
        u_max_r = np.max(np.abs(self.ur))
        u_max_z = np.max(np.abs(self.uz))

        dt_cfl = self.dt
        if u_max_r > 1e-10:
            dt_cfl = min(dt_cfl, self.cfl_max * g.dr / u_max_r)
        if u_max_z > 1e-10:
            dt_cfl = min(dt_cfl, self.cfl_max * g.dz / u_max_z)

        dx_min = min(g.dr, g.dz)

        k_max = max(self.material.k_polymer, self.material.k_air)
        rho_cp_min = min(self.material.rho_polymer * self.material.cp_polymer,
                         self.material.rho_air * self.material.cp_air)
        dt_thermal = 0.25 * rho_cp_min * dx_min**2 / max(k_max, 1e-10)

        dt_adapt = min(dt_cfl, dt_thermal, self.dt)
        return max(dt_adapt, self.dt * 0.001)

    def _solve_pressure_correction(self, div_star, dt_sub):
        nr, nz = self.nr, self.nz
        g = self.grid
        p_prime = np.zeros((nr, nz))
        rho_floor = self.material.rho_polymer * 0.1

        p_prime = _sor_pressure_solve(
            p_prime, div_star, self.rho, g.r_centers, g.r_faces,
            g.dr_arr, g.dz_arr, nr, nz, dt_sub, rho_floor)
        return p_prime

    def step(self):
        g = self.grid
        nr, nz = self.nr, self.nz
        m = self.material

        self._apply_boundary_conditions()
        self._update_properties()

        dur_dr, dur_dz, duz_dr, duz_dz = compute_velocity_gradients(
            self.ur, self.uz, nr, nz, g.r_centers, g.z_centers)
        gamma_dot = compute_shear_rate_field(
            dur_dr, dur_dz, duz_dr, duz_dz, self.ur, g.r_centers, nr, nz)

        params = m.get_params_tuple()
        self.eta = compute_viscosity_field(
            gamma_dot, self.T, self.alpha, nr, nz,
            params[0], params[1], params[2], params[3], params[4],
            params[7], params[8], params[9], params[5], params[6], params[17])

        dt = self._compute_adaptive_dt()

        f_st_r, f_st_z = compute_csf_force(
            self.alpha, m.sigma, g.r_centers, g.r_faces, g.dz_arr, nr, nz)

        alpha_threshold = 0.01
        nozzle_r_idx = min(np.searchsorted(g.r_faces, self.nozzle_radius), nr)
        nozzle_z_start = np.searchsorted(g.z_centers, 0.0)

        nozzle_exit_uz = np.zeros(nr)
        for i in range(nozzle_r_idx):
            r = g.r_centers[i]
            R = self.nozzle_radius
            nozzle_exit_uz[i] = -2.0 * self.u_inlet * (1.0 - (r / R)**2)

        ur_star = self.ur.copy()
        uz_star = self.uz.copy()

        bed_interaction_height = 3

        for i in range(nr):
            for j in range(nz):
                if self.alpha[i, j] < alpha_threshold:
                    ur_star[i, j] = 0.0
                    uz_star[i, j] = 0.0
                    continue

                if j >= nozzle_z_start and i < nozzle_r_idx:
                    continue

                if j == 0:
                    uz_star[i, j] = 0.0
                    ur_star[i, j] = 0.0
                elif j < bed_interaction_height and self.alpha[i, 0] > 0.1:
                    rho_c = max(self.rho[i, j], m.rho_polymer * 0.1)
                    eta_c = self.eta[i, j]

                    dp_dz = 0.0
                    if j > 0 and j < nz - 1:
                        dz_c = 0.5 * (g.dz_arr[j - 1] + g.dz_arr[j]) + 0.5 * (g.dz_arr[j] + g.dz_arr[j + 1])
                        dp_dz = (self.p[i, j + 1] - self.p[i, j - 1]) / dz_c
                    elif j == 0 and nz > 1:
                        dp_dz = (self.p[i, j + 1] - self.p[i, j]) / (0.5 * (g.dz_arr[j] + g.dz_arr[j + 1]))

                    dp_dr = 0.0
                    if i > 0 and i < nr - 1:
                        dp_dr = (self.p[i + 1, j] - self.p[i - 1, j]) / (g.r_centers[i + 1] - g.r_centers[i - 1])
                    elif i == 0 and nr > 1:
                        dp_dr = (self.p[1, j] - self.p[0, j]) / (g.r_centers[1] - g.r_centers[0])

                    visc_z = 0.0
                    visc_r = 0.0
                    if j > 0 and j < nz - 1:
                        dz_n = 0.5 * (g.dz_arr[j] + g.dz_arr[j + 1])
                        dz_s = 0.5 * (g.dz_arr[j - 1] + g.dz_arr[j])
                        dz_avg = 0.5 * (dz_n + dz_s)
                        visc_z += eta_c * (self.uz[i, j+1] - 2*self.uz[i, j] + self.uz[i, j-1]) / (dz_avg**2)
                        visc_r += eta_c * (self.ur[i, j+1] - 2*self.ur[i, j] + self.ur[i, j-1]) / (dz_avg**2)
                    if i > 0 and i < nr - 1:
                        dr_e = g.r_centers[i + 1] - g.r_centers[i]
                        dr_w = g.r_centers[i] - g.r_centers[i - 1]
                        dr_avg = 0.5 * (dr_e + dr_w)
                        visc_z += eta_c * (self.uz[i+1, j] - 2*self.uz[i, j] + self.uz[i-1, j]) / (dr_avg**2)
                        visc_r += eta_c * (self.ur[i+1, j] - 2*self.ur[i, j] + self.ur[i-1, j]) / (dr_avg**2)

                    dr_min = g.dr_arr[i]
                    dz_min = g.dz_arr[j]
                    dt_local = min(dt, rho_c * min(dr_min, dz_min)**2 / (4 * max(eta_c, 1.0)))

                    uz_star[i, j] = self.uz[i, j] + dt_local * (-dp_dz / rho_c + self.gravity + visc_z / rho_c)
                    ur_star[i, j] = self.ur[i, j] + dt_local * (-dp_dr / rho_c + visc_r / rho_c)
                elif i < nozzle_r_idx:
                    uz_star[i, j] = nozzle_exit_uz[i]
                    ur_star[i, j] = 0.0
                else:
                    uz_star[i, j] = 0.0
                    ur_star[i, j] = 0.0

        for i in range(nozzle_r_idx):
            for j in range(nozzle_z_start, nz):
                r = g.r_centers[i]
                R = self.nozzle_radius
                profile = 2.0 * self.u_inlet * (1.0 - (r / R)**2)
                uz_star[i, j] = -profile
                ur_star[i, j] = 0.0

        ur_star[0, :] = 0.0
        ur_star[-1, :] = 0.0
        uz_star[-1, :] = 0.0

        for piso_iter in range(self.n_piso):
            div_star = compute_divergence_axisym(
                ur_star, uz_star, g.r_centers, g.r_faces, g.dr_arr, g.dz_arr, nr, nz)

            p_prime = self._solve_pressure_correction(div_star, dt)

            np.nan_to_num(p_prime, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            rho_min_corr = self.material.rho_polymer * 0.1

            for i in range(nr):
                for j in range(nz):
                    if j >= nozzle_z_start and i < nozzle_r_idx:
                        continue
                    if self.alpha[i, j] < alpha_threshold:
                        continue

                    rho_ij = max(self.rho[i, j], rho_min_corr)

                    dp_dr = 0.0
                    if i > 0 and i < nr - 1:
                        dp_dr = (p_prime[i + 1, j] - p_prime[i - 1, j]) / (g.r_centers[i + 1] - g.r_centers[i - 1])
                    elif i == 0 and nr > 1:
                        dp_dr = (p_prime[1, j] - p_prime[0, j]) / (g.r_centers[1] - g.r_centers[0])
                    elif i == nr - 1:
                        dp_dr = (p_prime[i, j] - p_prime[i - 1, j]) / (g.r_centers[i] - g.r_centers[i - 1])

                    dp_dz = 0.0
                    if j > 0 and j < nz - 1:
                        dz_c = 0.5 * (g.dz_arr[j - 1] + g.dz_arr[j]) + 0.5 * (g.dz_arr[j] + g.dz_arr[j + 1])
                        dp_dz = (p_prime[i, j + 1] - p_prime[i, j - 1]) / dz_c
                    elif j == 0 and nz > 1:
                        dp_dz = (p_prime[i, j + 1] - p_prime[i, j]) / (0.5 * (g.dz_arr[j] + g.dz_arr[j + 1]))
                    elif j == nz - 1 and nz > 1:
                        dp_dz = (p_prime[i, j] - p_prime[i, j - 1]) / (0.5 * (g.dz_arr[j - 1] + g.dz_arr[j]))

                    ur_star[i, j] -= dt * dp_dr / rho_ij
                    uz_star[i, j] -= dt * dp_dz / rho_ij

            self.p += self.p_relax * p_prime

        ur_star[0, :] = 0.0
        ur_star[-1, :] = 0.0
        uz_star[-1, :] = 0.0
        uz_star[:, 0] = 0.0
        ur_star[:, 0] = 0.0

        self.ur = self.u_relax * ur_star + (1.0 - self.u_relax) * self.ur
        self.uz = self.u_relax * uz_star + (1.0 - self.u_relax) * self.uz

        self.uz[:, 0] = 0.0
        self.ur[:, 0] = 0.0

        np.nan_to_num(self.ur, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(self.uz, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(self.p, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        u_max_phys = max(self.u_inlet * 50.0, 5.0)
        np.clip(self.ur, -u_max_phys, u_max_phys, out=self.ur)
        np.clip(self.uz, -u_max_phys, u_max_phys, out=self.uz)
        np.clip(self.p, -1e12, 1e12, out=self.p)

        self.alpha = advect_vof_axisym(
            self.alpha, self.ur, self.uz,
            g.r_centers, g.r_faces, g.dz_arr, nr, nz, dt)

        if self.c_alpha > 0:
            self.alpha = apply_vof_compression(
                self.alpha, self.ur, self.uz,
                g.r_centers, g.r_faces, g.dz_arr, nr, nz, dt, self.c_alpha)

        self.T = advect_temperature_axisym(
            self.T, self.ur, self.uz, self.rho, self.cp, self.k_field,
            g.r_centers, g.r_faces, g.dz_arr, nr, nz, dt,
            self.h_conv, self.T_ambient, self.alpha)

        np.nan_to_num(self.T, copy=False, nan=self.T_ambient, posinf=self.T_ambient, neginf=self.T_ambient)
        T_clamp_min = self.T_ambient - 5.0
        T_clamp_max = self.T_nozzle + 20.0
        np.clip(self.T, T_clamp_min, T_clamp_max, out=self.T)

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

        nozzle_z_start = np.searchsorted(g.z_centers, 0.0)

        swell_ratios = []
        for j in range(nozzle_z_start - 1, max(int(nozzle_z_start) - 21, -1), -1):
            found = False
            for i in range(self.nr - 1, -1, -1):
                if self.alpha[i, j] > 0.5:
                    swell_ratios.append(max(g.r_centers[i] / self.nozzle_radius, 0.0))
                    found = True
                    break
            if not found:
                if len(swell_ratios) > 0:
                    swell_ratios.append(swell_ratios[-1])
                else:
                    swell_ratios.append(1.0)

        nozzle_r_idx = min(np.searchsorted(g.r_faces, self.nozzle_radius), self.nr)
        p_nozzle_cells = self.p[:nozzle_r_idx, -1]
        p_inlet = np.mean(p_nozzle_cells) if len(p_nozzle_cells) > 0 else 0.0
        p_outlet_cells = self.p[:nozzle_r_idx, 0]
        p_outlet = np.mean(p_outlet_cells) if len(p_outlet_cells) > 0 else 0.0

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
        u_mag_masked = u_mag.copy()
        u_mag_masked[self.alpha < 0.5] = 0.0
        g = self.grid

        contour_r = []
        contour_z = []
        for j in range(self.nz):
            for i in range(self.nr - 1):
                a1 = self.alpha[i, j]
                a2 = self.alpha[i + 1, j]
                if (a1 - 0.5) * (a2 - 0.5) < 0:
                    t = (0.5 - a1) / (a2 - a1) if abs(a2 - a1) > 1e-10 else 0.5
                    r_interp = g.r_centers[i] + t * (g.r_centers[i + 1] - g.r_centers[i])
                    contour_r.append(float(r_interp))
                    contour_z.append(float(g.z_centers[j]))

        for i in range(self.nr):
            for j in range(self.nz - 1):
                a1 = self.alpha[i, j]
                a2 = self.alpha[i, j + 1]
                if (a1 - 0.5) * (a2 - 0.5) < 0:
                    t = (0.5 - a1) / (a2 - a1) if abs(a2 - a1) > 1e-10 else 0.5
                    z_interp = g.z_centers[j] + t * (g.z_centers[j + 1] - g.z_centers[j])
                    contour_r.append(float(g.r_centers[i]))
                    contour_z.append(float(z_interp))

        result = {
            'alpha': self.alpha.tolist(),
            'u_mag': u_mag_masked.tolist(),
            'T': self.T.tolist(),
            'p': self.p.tolist(),
            'eta': self.eta.tolist(),
            'ur': self.ur.tolist(),
            'uz': self.uz.tolist(),
            'contour_r': contour_r,
            'contour_z': contour_z,
            'r_centers': g.r_centers.tolist(),
            'z_centers': g.z_centers.tolist(),
            'r_faces': g.r_faces.tolist(),
            'z_faces': g.z_faces.tolist(),
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
