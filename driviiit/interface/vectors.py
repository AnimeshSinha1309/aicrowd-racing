import collections


Vector = collections.namedtuple("Vector", ["x", "y", "z"])

CoordinateTransform = collections.namedtuple(
    "CoordinateTransform", ["x", "y", "z", "yaw", "pitch", "roll"]
)

WheelSensors = collections.namedtuple(
    "WheelSensors", ["front_left", "front_right", "rear_left", "rear_right"]
)
