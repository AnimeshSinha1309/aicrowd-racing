import typing as ty

import numpy as np

from driviiit.interface.vectors import CoordinateTransform


def camera_details_to_intrinsic_matrix(
    field_of_view: float, image_shape: ty.Tuple[int, int]
):
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
    camera_matrix = np.array(
        [
            [focal_length, 0, image_shape[0] / 2],
            [0, focal_length, image_shape[1] / 2],
            [0, 0, 1],
        ]
    )
    return camera_matrix


def euler_angles_to_transformation_matrix(coordinate_transform: CoordinateTransform):
    """
    Generates the coordinate transform matrix from linear shift and euler angles
    :type coordinate_transform: CoordinateTransform
    :param coordinate_transform: Object containing values of (x, y, z, pitch, roll, yaw)
    :rtype: np.array of shape (3, 4)
    :return: The coordinate transform matrix
    """
    theta = np.array(
        [
            coordinate_transform.pitch,
            coordinate_transform.roll,
            coordinate_transform.yaw,
        ]
    )
    shift = np.array(
        [coordinate_transform.x, coordinate_transform.y, coordinate_transform.z]
    )
    net_rotation = np.eye(3)
    for i in range(3):
        rot = np.array(
            [
                [np.cos(theta[i]), -np.sin(theta[i])],
                [np.sin(theta[i]), np.cos(theta[i])],
            ]
        )
        cur_rotation = np.eye(3)
        for x in range(3):
            for y in range(3):
                cur_rotation[x, y] = (
                    rot[x - (1 if x >= i else 0), y - (1 if y >= i else 0)]
                    if x != i and y != i
                    else x == y
                )
        net_rotation = cur_rotation @ net_rotation
    transform = np.concatenate([net_rotation, np.expand_dims(shift, axis=1)], axis=1)
    return transform


def apply_homogenous_transform(transform_matrix: np.array, points: np.array):
    """
    Applies a homogenous transform matrix and returns the resulting points cloud
    :param transform_matrix: The transformation to be applied
    :param points: Array of all points in the point cloud
    The points array should be of the shape (number_of_points, dimensionality_of_points)
    """
    if transform_matrix.shape[1] != points.shape[1]:
        points = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    points = points @ transform_matrix.T
    points = (
        points[:, :-1] / points[:, -1:]
        if transform_matrix.shape[0] == transform_matrix.shape[1]
        else points
    )
    return points


def invert_affine_orthogonal_matrix(matrix):
    matrix[:, :-1] = matrix[:, :-1].T
    matrix[:, -1] = -matrix[:, -1]
    return matrix
