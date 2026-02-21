import numpy as np
from numba import njit


class AxiSymGrid:
    def __init__(self, nr, nz, r_max, z_min, z_max, stretch_type='uniform', stretch_ratio=1.0):
        self.nr = nr
        self.nz = nz
        self.r_max = r_max
        self.z_min = z_min
        self.z_max = z_max
        self.stretch_type = stretch_type
        self.stretch_ratio = max(stretch_ratio, 1.0)

        if stretch_type == 'stretched' and stretch_ratio > 1.0:
            self.r_faces = self._stretch_radial(nr, r_max, stretch_ratio)
            self.z_faces = self._stretch_axial(nz, z_min, z_max, stretch_ratio)
        else:
            self.r_faces = np.linspace(0, r_max, nr + 1)
            self.z_faces = np.linspace(z_min, z_max, nz + 1)

        self.r_centers = 0.5 * (self.r_faces[:-1] + self.r_faces[1:])
        self.z_centers = 0.5 * (self.z_faces[:-1] + self.z_faces[1:])

        self.dr_arr = np.diff(self.r_faces)
        self.dz_arr = np.diff(self.z_faces)

        self.dr = float(np.min(self.dr_arr))
        self.dz = float(np.min(self.dz_arr))

        self.cell_volumes = self._compute_volumes()
        self.face_areas_r = self._compute_face_areas_r()
        self.face_areas_z = self._compute_face_areas_z()

    def _stretch_radial(self, nr, r_max, ratio):
        xi = np.linspace(0, 1, nr + 1)
        beta = ratio
        stretched = xi ** beta
        return r_max * stretched

    def _stretch_axial(self, nz, z_min, z_max, ratio):
        z_nozzle_exit = 0.0
        total_length = z_max - z_min
        frac_below = (z_nozzle_exit - z_min) / total_length
        n_below = max(int(round(nz * frac_below)), 2)
        n_above = nz - n_below

        beta = ratio

        xi_below = np.linspace(0, 1, n_below + 1)
        stretched_below = 1.0 - (1.0 - xi_below) ** beta
        z_below = z_min + (z_nozzle_exit - z_min) * stretched_below

        xi_above = np.linspace(0, 1, n_above + 1)
        stretched_above = xi_above ** beta
        z_above = z_nozzle_exit + (z_max - z_nozzle_exit) * stretched_above

        z_faces = np.concatenate([z_below, z_above[1:]])
        return z_faces

    def _compute_volumes(self):
        volumes = np.zeros((self.nr, self.nz))
        for i in range(self.nr):
            r_inner = self.r_faces[i]
            r_outer = self.r_faces[i + 1]
            for j in range(self.nz):
                volumes[i, j] = np.pi * (r_outer**2 - r_inner**2) * self.dz_arr[j]
        return volumes

    def _compute_face_areas_r(self):
        areas = np.zeros(self.nr + 1)
        for i in range(self.nr + 1):
            areas[i] = 2.0 * np.pi * self.r_faces[i]
        return areas

    def _compute_face_areas_z(self):
        areas = np.zeros(self.nr)
        for i in range(self.nr):
            r_inner = self.r_faces[i]
            r_outer = self.r_faces[i + 1]
            areas[i] = np.pi * (r_outer**2 - r_inner**2)
        return areas


@njit(cache=True)
def interp_face_r(phi, i_face, j):
    nr = phi.shape[0]
    if i_face == 0:
        return phi[0, j]
    elif i_face == nr:
        return phi[nr - 1, j]
    else:
        return 0.5 * (phi[i_face - 1, j] + phi[i_face, j])


@njit(cache=True)
def interp_face_z(phi, i, j_face):
    nz = phi.shape[1]
    if j_face == 0:
        return phi[i, 0]
    elif j_face == nz:
        return phi[i, nz - 1]
    else:
        return 0.5 * (phi[i, j_face - 1] + phi[i, j_face])
