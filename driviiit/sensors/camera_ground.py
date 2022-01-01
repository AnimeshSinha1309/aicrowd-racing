import typing as ty

import numpy as np

from driviiit.interface.vectors import CoordinateTransform
from driviiit.sensors.imu_readings import IMUSensorReading
from sensors.camera_utils import (
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


def camera_points_to_car(
    points: np.array,
    imu: IMUSensorReading,
    camera_position: CoordinateTransform,
    camera_intrinsics: np.array
) -> np.array:
    """
    Function to map image points from a camera to their actual 3d coordinates
    Only works for a horizontal camera, i.e. z axis perpendicular to the ground normal

    Computing (x, y) given k, X, Y using the relation
        [X, Y, homogeneous] = k @ [x, y, z]

    (k[0, 0] * x + k[0, 1] * y + k[0, 2] * z) / (k[2, 0] * x + k[2, 1] * y + k[2, 2] * z) = X
    (k[1, 0] * x + k[1, 1] * y + k[1, 2] * z) / (k[2, 0] * x + k[2, 1] * y + k[2, 2] * z) = Y

    k[0, 0] * x + k[0, 1] * y + k[0, 2] * z = k[2, 0] * X * x + k[2, 1] * X * y + k[2, 2] * X * z
    k[1, 0] * x + k[1, 1] * y + k[1, 2] * z = k[2, 0] * Y * x + k[2, 1] * Y * y + k[2, 2] * Y * z

    (k[0, 0] - k[2, 0] * X) * x + (k[0, 1] - k[2, 1] * X) * y + (k[0, 2] - k[2, 2] * X) * z = 0
    (k[1, 0] - k[2, 0] * Y) * x + (k[1, 1] - k[2, 1] * Y) * y + (k[1, 2] - k[2, 2] * Y) * z = 0

    Rewriting this as a1 * x + b1 * y + c1 * z = 0 and a2 * x + b2 * y + c2 * z = 0
    a2 * b1 * y + a2 * c1 * z = a1 * b2 * y + a1 * c2 * z
    We can state that
        x = (c1 * b2 - c2 * b1) * y / (c2 * a1 - c1 * a2)
        z = (a1 * b2 - a2 * b1) * y / (a2 * c1 - a1 * c2)
    """
    # Solve linear equations to get the coordinates in the camera frame in 3D (i.e. projected on ground)
    k = camera_intrinsics
    y = camera_position.z  # height from the ground plane, in camera frame it's y

    im_x, im_y = points[:, 0], points[:, 1]
    a1, b1, c1 = k[0, 0] - k[2, 0] * im_x, k[0, 1] - k[2, 1] * im_x, k[0, 2] - k[2, 2] * im_x
    a2, b2, c2 = k[1, 0] - k[2, 0] * im_y, k[1, 1] - k[2, 1] * im_y, k[1, 2] - k[2, 2] * im_y
    x = (c1 * b2 - c2 * b1) * y / (c2 * a1 - c1 * a2)
    z = (a1 * b2 - a2 * b1) * y / (a2 * c1 - a1 * c2)
    # Mask out the low reliability points, only keep the ones we are sure of
    mask = np.abs(z) < 50
    x, z = x[mask], z[mask]
    pts_ground = np.stack([x, z, np.full(shape=len(x), fill_value=y)], axis=1)
    # Rotate and translate using the camera pose and the car pose
    rt = euler_angles_to_transformation_matrix(camera_position)
    rt[2, 3] = 0  # We already lifted the frame for the camera, so z axis should be left as it
    pts_car = apply_homogenous_transform(rt, pts_ground)
    pts_car = pts_car[:, :2]
    return pts_car


def camera_points_to_ground(
    points: np.array,
    imu: IMUSensorReading,
    camera_position: CoordinateTransform,
    camera_intrinsics: np.array
) -> np.array:
    pts_car = camera_points_to_car(points, imu, camera_position, camera_intrinsics)
    yaw = imu.position.yaw
    pts_world = pts_car @ np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]]).T
    pts_world = np.array([imu.position.x, imu.position.y]) - pts_world
    return pts_world
