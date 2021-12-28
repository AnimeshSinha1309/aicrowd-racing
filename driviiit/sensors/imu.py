import numpy as np

from driviiit.interface.vectors import Vector, CoordinateTransform, WheelSensors


class IMUSensorReading:
    def __init__(self, imu_vector):
        self.reading = imu_vector

    @property
    def steering_angle(self):
        return self.reading[0]

    @property
    def gear(self):
        return self.reading[1]

    @property
    def mode(self):
        return self.reading[2]

    @property
    def velocity(self):
        return Vector(x=self.reading[3], y=self.reading[4], z=self.reading[5])

    @property
    def acceleration(self):
        return Vector(x=self.reading[6], y=self.reading[7], z=self.reading[8])

    @property
    def angular_velocity(self):
        return Vector(x=self.reading[9], y=self.reading[10], z=self.reading[11])

    @property
    def speed(self):
        return np.linalg.norm(self.reading[3:6]) * 2

    @property
    def position(self):
        return CoordinateTransform(
            x=self.reading[16],
            y=self.reading[15],
            z=self.reading[17],
            yaw=self.reading[12],
            pitch=self.reading[13],
            roll=self.reading[14],
        )

    @property
    def rpm(self):
        return WheelSensors(
            front_left=self.reading[18],
            front_right=self.reading[19],
            rear_left=self.reading[20],
            rear_right=self.reading[21],
        )

    @property
    def brake(self):
        return WheelSensors(
            front_left=self.reading[22],
            front_right=self.reading[23],
            rear_left=self.reading[24],
            rear_right=self.reading[25],
        )

    @property
    def torq(self):
        return WheelSensors(
            front_left=self.reading[26],
            front_right=self.reading[27],
            rear_left=self.reading[28],
            rear_right=self.reading[29],
        )
