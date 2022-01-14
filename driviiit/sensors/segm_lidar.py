import matplotlib.pyplot as plt
import numpy as np

from driviiit.interface.config import SEGMENTATION_COLORS_MAP


class SegmentationLidarReading:
    def __init__(self, angles=None, center=None, image=None):
        if center is not None:
            self.center = center
        elif image is not None:
            mask = np.all(np.equal(image, SEGMENTATION_COLORS_MAP["CAR"]), axis=2)
            self.center = np.array(tuple(np.mean(points) for points in np.where(mask)))
        else:
            raise ValueError(
                "Either center or image should be specified for the lidar to initialize"
            )

        if angles is None:
            self.angles = [(i / 8) * 2 * np.pi for i in range(8)]
        else:
            self.angles = angles

    def __call__(self, image):
        road_mask = np.logical_or(
            np.all(np.equal(image, SEGMENTATION_COLORS_MAP["ROAD"]), axis=2),
            np.all(np.equal(image, SEGMENTATION_COLORS_MAP["CAR"]), axis=2),
        )
        endpoints, distances = [], []
        for angle in self.angles:
            vector: np.ndarray = np.array([np.cos(angle), np.sin(angle)])
            multiplier: float = 0
            while True:
                queried_point = (self.center - vector * (multiplier + 1)).astype(
                    np.int32
                )
                if not (
                    np.all(queried_point < road_mask.shape)
                    and np.all(queried_point >= (0, 0))
                    and road_mask[queried_point[0], queried_point[1]]
                ):
                    break
                multiplier += 5
            endpoints.append(self.center - vector * multiplier)
            distances.append(multiplier)
        return distances, endpoints

    def plot(self, image, endpoints):
        plt.imshow(image)
        for point in endpoints:
            plt.plot(
                [self.center[1], point[1]], [self.center[0], point[0]], color="black"
            )
        plt.show()
