import numpy as np


class Camera:
    def __init__(self, position, look_at, up, right):
        self.position = position
        self.look_at = look_at
        self.up = up
        self.right = right

    def get_matrix(self):
        # Rotation matrix from camera basis vectors.
        rotation = np.array(
            [
                [self.right[0], self.right[1], self.right[2], 0.0],
                [self.up[0], self.up[1], self.up[2], 0.0],
                [self.look_at[0], self.look_at[1], self.look_at[2], 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        # Translation that moves the world with respect to the camera position.
        translation = np.array(
            [
                [1.0, 0.0, 0.0, -self.position[0]],
                [0.0, 1.0, 0.0, -self.position[1]],
                [0.0, 0.0, 1.0, -self.position[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        return rotation @ translation
