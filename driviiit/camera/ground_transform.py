import typing as ty

import numpy as np

from driviiit.interface.vectors import CoordinateTransform
from driviiit.sensors.imu import IMUSensorReading
from driviiit.camera.camera_utils import (
    camera_details_to_intrinsic_matrix,
    euler_angles_to_transformation_matrix,
    apply_homogenous_transform,
    invert_affine_orthogonal_matrix,
)


class CameraGroundTransformer:
    """Converts the image coordinates to ground coordinates"""

    def __init__(
        self,
        field_of_view: float,
        image_shape: ty.Tuple[int, int],
        camera_position: CoordinateTransform,
    ):
        k = camera_details_to_intrinsic_matrix(field_of_view, image_shape)
        rt = euler_angles_to_transformation_matrix(camera_position)
        self.p = k @ rt

    def pixel_camera_to_ground(self, x):
        p = self.p
        a = np.array(
            [
                [
                    p[0, 0] - p[2, 0] * x[0],
                    p[0, 1] - p[2, 1] * x[0],
                    p[0, 3] - p[2, 3] * x[0],
                ],
                [
                    p[1, 0] - p[2, 0] * x[1],
                    p[1, 1] - p[2, 1] * x[1],
                    p[1, 3] - p[2, 3] * x[1],
                ],
            ]
        )
        g_x = (a[1, 1] * a[0, 2] - a[0, 1] * a[1, 2]) / (
            a[0, 1] * a[1, 0] - a[1, 1] * a[0, 0]
        )
        g_y = (a[1, 2] * a[0, 0] - a[0, 2] * a[1, 0]) / (
            a[1, 0] * a[0, 1] - a[0, 0] * a[1, 1]
        )
        return np.array([g_x, g_y])

    def pixel_ground_to_camera(self, x):
        p = self.p
        x_homogenous = np.expand_dims(np.array([x[0], x[1], 0, 1]), axis=1)
        pix_homogenous = (p @ x_homogenous).reshape(-1)
        return pix_homogenous[:2] / pix_homogenous[2]

    def mask_camera_to_ground(self, image):
        ground_points = np.array(
            [
                self.pixel_camera_to_ground(np.array([i, j]))
                for i in range(image.shape[0])
                for j in range(image.shape[1])
                if image[i][j] > 0
            ]
        )
        return ground_points

    def mask_ground_to_camera(self, ground):
        image_points = np.array(
            [
                self.pixel_camera_to_ground(np.array([i, j]))
                for i in range(ground.shape[0])
                for j in range(ground.shape[1])
                if ground[i][j] > 0
            ]
        )
        return image_points


def ground_points_to_camera(
    track: np.array,
    imu: IMUSensorReading,
    camera_position: CoordinateTransform,
    camera_intrinsics: np.array
) -> np.array:
    """
    Converts points on the ground to their corresponding points in the image frame
    :param track: The points on the ground frame to convert
    :param imu: The reading from the imu giving us the pose of the car
    :param camera_position: The 3-d pose of the camera
    :param camera_intrinsics: The k matrix containing info on image size and focal length
    """
    # Translate and rotate the points to be in the frame of reference of the car
    pts = track - np.array([imu.position.x, imu.position.y])
    yaw = imu.position.yaw
    pts = pts @ np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    # Come in the pose of the camera with only points in the front and swap axis z correctly
    pts_front = pts[pts[:, 1] > 0, :]
    pts_front = np.concatenate([pts_front, np.zeros((len(pts_front), 1))], axis=1)
    rt = euler_angles_to_transformation_matrix(camera_position)
    pts_cam = apply_homogenous_transform(invert_affine_orthogonal_matrix(rt), pts_front)
    pts_cam = np.stack([pts_cam[:, 0], pts_cam[:, 2], pts_cam[:, 1]], axis=1)
    # Apply the intrinsics to get an image and return the results
    pts_out = apply_homogenous_transform(camera_intrinsics, pts_cam)
    return pts_out
