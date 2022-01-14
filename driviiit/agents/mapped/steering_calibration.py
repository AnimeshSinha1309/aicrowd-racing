import numpy as np


class SteeringCalibration:
    def __init__(self, center_line, inner_line, outer_line):
        self.center_line = center_line
        self.inner_line = inner_line
        self.outer_line = outer_line

    def curvature_angle(self, segment):
        direction_vector = [
            self.center_line[(segment + i + 1) % len(self.center_line)]
            - self.center_line[(segment + i) % len(self.center_line)]
            for i in range(10)
        ]
        rotation_angles = [
            np.dot(
                direction_vector[i] @ np.array([[0, 1], [-1, 0]]),
                direction_vector[i + 1],
            )
            for i in range(len(direction_vector) - 1)
        ]
        return np.mean(rotation_angles)

    def steering_angle(self, segment, imu):
        # angle = self.curvature_angle(segment) * 200
        # return np.clip(angle, -1.0, 1.0)
        # Translate and rotate the points to be in the frame of reference of the car
        points_in_near_future = [
            self.center_line[(segment + i) % len(self.center_line)] for i in range(50)
        ]
        pts = points_in_near_future - np.array([imu.position.x, imu.position.y])
        yaw = imu.position.yaw
        pts = pts @ np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        center_line_distance = -np.clip(np.mean(pts[:, 0]), -1.0, 1.0) / 2
        return center_line_distance
