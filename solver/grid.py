import numpy as np
from numba import njit


class AxiSymGrid:
    def __init__(self, nr, nz, r_max, z_min, z_max):
        self.nr = nr
        self.nz = nz
        self.r_max = r_max
        self.z_min = z_min
        self.z_max = z_max

        self.dr = r_max / nr
        self.dz = (z_max - z_min) / nz

        self.r_faces = np.linspace(0, r_max, nr + 1)
        self.z_faces = np.linspace(z_min, z_max, nz + 1)

        self.r_centers = 0.5 * (self.r_faces[:-1] + self.r_faces[1:])
        self.z_centers = 0.5 * (self.z_faces[:-1] + self.z_faces[1:])

        self.cell_volumes = self._compute_volumes()
        self.face_areas_r = self._compute_face_areas_r()
        self.face_areas_z = self._compute_face_areas_z()

    def _compute_volumes(self):
        volumes = np.zeros((self.nr, self.nz))
        for i in range(self.nr):
            r_inner = self.r_faces[i]
            r_outer = self.r_faces[i + 1]
            volumes[i, :] = np.pi * (r_outer**2 - r_inner**2) * self.dz
        return volumes

    def _compute_face_areas_r(self):
        areas = np.zeros(self.nr + 1)
        for i in range(self.nr + 1):
            areas[i] = 2.0 * np.pi * self.r_faces[i] * self.dz
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
