import matplotlib.pyplot as plt
import numpy as np

import driviiit
from l2r.envs.env import RacingEnv

env = RacingEnv({}, {})
obs = [np.random.random(30), [np.random.random((512, 384, 3)) for _ in range(10)]]

imu = driviiit.sensors.imu.IMUSensorReading(obs[0])
inside_points = (
    env.inside_arr
    @ np.array(
        [
            [np.cos(imu.position.yaw), np.sin(imu.position.yaw)],
            [-np.sin(imu.position.yaw), np.cos(imu.position.yaw)],
        ]
    ).T
    + np.array([imu.position.x, imu.position.y])
)

cg = driviiit.camera.ground_transform.CameraGroundTransformer(
    driviiit.interface.config.FIELD_OF_VIEW,
    driviiit.interface.config.IMAGE_SHAPE,
    driviiit.interface.config.CAMERA_FRONT_POSITION,
)

inside_projected = np.apply_along_axis(
    cg.pixel_ground_to_camera, axis=1, arr=inside_points
)

point_filter = np.all(
    np.logical_and(
        np.array([0, 0]) < inside_projected,
        inside_projected < driviiit.interface.config.IMAGE_SHAPE,
    ),
    axis=1,
)
inside_projected = inside_projected[point_filter, :]

plt.imshow(obs[1][0])
plt.scatter(inside_projected[:, 0], inside_projected[:, 1])
plt.show()
