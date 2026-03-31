import numpy as np


class Projection:
    def __init__(self, near_plane, far_plane, fov, aspect_ratio):
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.fov = fov
        self.aspect_ratio = aspect_ratio

    def get_matrix(self):
        f = self.far_plane
        n = self.near_plane

        s = 1.0 / np.tan(self.fov / 2.0)

        # Same perspective matrix as in TP4.
        return np.array(
            [
                [s / self.aspect_ratio, 0.0, 0.0, 0.0],
                [0.0, s, 0.0, 0.0],
                [0.0, 0.0, f / (f - n), -(f * n) / (f - n)],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
