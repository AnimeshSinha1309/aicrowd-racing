import numpy as np

import driviiit


def test_camera_ground_camera_transform():
    cg = driviiit.camera.ground_transform.CameraGroundTransformer(
        driviiit.interface.config.FIELD_OF_VIEW,
        driviiit.interface.config.IMAGE_SHAPE,
        driviiit.interface.config.CAMERA_FRONT_POSITION
    )
    for _i in range(20):
        random_pixels = np.random.randint(driviiit.interface.config.IMAGE_SHAPE)
        ground_coordinates = cg.pixel_camera_to_ground(random_pixels)
        recovered_pixes = cg.pixel_ground_to_camera(ground_coordinates)
        assert np.allclose(random_pixels, recovered_pixes)
