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


def test_camera_to_road_construction():
    import driviiit
    from matplotlib import pyplot as plt
    import numpy as np

    dat = np.load("data/records/segm_front_0001.000.npy")
    img = np.load("data/records/camera_front_0001.000.npy")

    for target in range(50):

        road_mask = np.all(np.equal(dat[target], np.array([204, 80, 109])), axis=2)
        cg = driviiit.camera.ground_transform.CameraGroundTransformer(
            driviiit.interface.config.FIELD_OF_VIEW,
            driviiit.interface.config.IMAGE_SHAPE,
            driviiit.interface.config.CAMERA_FRONT_POSITION
        )
        pts = cg.mask_camera_to_ground(road_mask)
        print(np.mean)

    plt.imshow(img[target])
    plt.show()
    plt.imshow(dat[target])
    plt.show()
    plt.imshow(road_mask)
    plt.show()
    plt.scatter(pts[:, 0], pts[:, 1], s=0.1)
    plt.show()
