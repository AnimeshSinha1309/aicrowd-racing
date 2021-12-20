import typing as ty

import numpy as np

from driviiit.interface.vectors import CoordinateTransform


class CameraGroundTransformer:
    """Converts the image coordinates to ground coordinates"""

    def __init__(
            self,
            field_of_view: float,
            image_shape: ty.Tuple[int, int],
            camera_position: CoordinateTransform
    ):
        k = self.get_camera_matrix(field_of_view, image_shape)
        rt = self.get_projection_matrix(camera_position)
        self.p = k @ rt

    @staticmethod
    def get_camera_matrix(field_of_view: float, image_shape: ty.Tuple[int, int]):
        """
        Generate camera intrinsics matrix from the angular field of view and shape of the image
        :type field_of_view: float
        :param field_of_view: angular value of the field of view
        :type image_shape: tuple[int, int]
        :param image_shape: tuple representing the shape of the image
        :rtype: np.array of shape (3, 3)
        :return: the camera intrinsics matrix
        """
        focal_length = image_shape[0] / (2 * np.tan(field_of_view / 2))
        camera_matrix = np.array([
            [focal_length, 0, image_shape[0]],
            [0, focal_length, image_shape[1]],
            [0, 0, 1],
        ])
        return camera_matrix

    @staticmethod
    def get_projection_matrix(coordinate_transform: CoordinateTransform):
        """
        Generates the coordinate transform matrix from linear shift and euler angles
        :type coordinate_transform: CoordinateTransform
        :param coordinate_transform: Object containing values of (x, y, z, pitch, roll, yaw)
        :rtype: np.array of shape (3, 4)
        :return: The coordinate transform matrix
        """
        theta = np.array([coordinate_transform.pitch, coordinate_transform.roll, coordinate_transform.yaw])
        shift = np.array([coordinate_transform.x, coordinate_transform.y, coordinate_transform.z])
        net_rotation = np.eye(3)
        for i in range(3):
            rot = np.array([
                [np.cos(theta[i]), -np.sin(theta[i])],
                [np.sin(theta[i]), np.cos(theta[i])],
            ])
            cur_rotation = np.eye(3)
            for x in range(3):
                for y in range(3):
                    cur_rotation[x, y] = rot[x - (1 if x >= i else 0), y - (1 if y >= i else 0)] \
                        if x != i and y != i else x == y
            net_rotation = cur_rotation @ net_rotation
        transform = np.concatenate([net_rotation, np.expand_dims(shift, axis=1)], axis=1)
        return transform

    def pixel_camera_to_ground(self, x):
        p = self.p
        a = np.array([
            [p[0, 0] - p[2, 0] * x[0], p[0, 1] - p[2, 1] * x[0], p[0, 3] - p[2, 3] * x[0]],
            [p[1, 0] - p[2, 0] * x[1], p[1, 1] - p[2, 1] * x[1], p[1, 3] - p[2, 3] * x[1]],
        ])
        g_x = (a[1, 1] * a[0, 2] - a[0, 1] * a[1, 2]) / (a[0, 1] * a[1, 0] - a[1, 1] * a[0, 0])
        g_y = (a[1, 2] * a[0, 0] - a[0, 2] * a[1, 0]) / (a[1, 0] * a[0, 1] - a[0, 0] * a[1, 1])
        return np.array([g_x, g_y])

    def pixel_ground_to_camera(self, x):
        p = self.p
        x_homogenous = np.expand_dims(np.array([x[0], x[1], 0, 1]), axis=1)
        pix_homogenous = (p @ x_homogenous).reshape(-1)
        return pix_homogenous[:2] / pix_homogenous[2]

    def mask_camera_to_ground(self, image):
        ground_points = np.array([
            self.pixel_camera_to_ground(np.array([i, j]))
            for i in range(image.shape[0])
            for j in range(image.shape[1])
            if image[i][j] > 0
        ])
        return ground_points

    def mask_ground_to_camera(self, ground):
        image_points = np.array([
            self.pixel_camera_to_ground(np.array([i, j]))
            for i in range(ground.shape[0])
            for j in range(ground.shape[1])
            if ground[i][j] > 0
        ])
        return image_points
