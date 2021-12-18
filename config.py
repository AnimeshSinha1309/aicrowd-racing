from agents.random_agent import RandomAgent


class SubmissionConfig(object):
    agent = RandomAgent
    pre_eval_time = 100
    eval_episodes = 10


class EnvConfig(object):
    multimodal = True
    eval_mode = True
    n_eval_laps = 1
    max_timesteps = 5000
    obs_delay = 0.1
    not_moving_timeout = 100
    reward_pol = "custom"
    provide_waypoints = False
    reward_kwargs = {
        "oob_penalty": 5.0,
        "min_oob_penalty": 25.0,
        "max_oob_penalty": 125.0,
    }
    controller_kwargs = {
        "sim_version": "ArrivalSim-linux-0.7.1.188691",
        "quiet": False,
        "user": "ubuntu",
        "start_container": False,
        "sim_path": "/home/LinuxNoEditor",
    }
    action_if_kwargs = {
        "max_accel": 6,
        "min_accel": -16,
        "max_steer": .3,
        "min_steer": -.3,
        "ip": "0.0.0.0",
        "port": 7077,
    }
    pose_if_kwargs = {
        "ip": "0.0.0.0",
        "port": 7078,
    }
    camera_if_kwargs = {
        "ip": "0.0.0.0",
        "port": 8008,
    }
    segm_if_kwargs = {
        "ip": 'tcp://127.0.0.1',
        "port": 8009
    }
    birdseye_if_kwargs = {
        "ip": 'tcp://127.0.0.1',
        "port": 8010
    }
    birdseye_segm_if_kwargs = {
        "ip": 'tcp://127.0.0.1',
        "port": 8011
    }
    logger_kwargs = {
        "default": True,
    }
    cameras = {
        "CameraFrontRGB": {
            "Addr": "tcp://0.0.0.0:8008",
            "Format": "ColorBGR8",
            "FOVAngle": 90,
            "Width": 512,
            "Height": 384,
            "bAutoAdvertise": True,
        },
        "CameraFrontSegm": {
            "Addr": "tcp://0.0.0.0:9008",
            "Format": "SegmBGR8",
            "FOVAngle": 90,
            "Width": 512,
            "Height": 384,
            "bAutoAdvertise": True,
        },
        "CameraBirdsEye": {
            "Addr": "tcp://0.0.0.0:10008",
            "Format": "ColorBGR8",
            "FOVAngle": 90,
            "Width": 512,
            "Height": 384,
            "bAutoAdvertise": True,
        },
        "CameraBirdsEyeSegm": {
            "Addr": "tcp://0.0.0.0:10009",
            "Format": "SegmBGR8",
            "FOVAngle": 90,
            "Width": 512,
            "Height": 384,
            "bAutoAdvertise": True,
        }
    }


class SimulatorConfig(object):
    racetrack = "Thruxton"
    active_sensors = [
        "CameraFrontRGB",
        "ImuOxtsSensor",
        "CameraFrontSegm",
        "CameraBirdsEye",
        "CameraBirdsEyeSegm",
    ]
    driver_params = {
        "DriverAPIClass": "VApiUdp",
        "DriverAPI_UDP_SendAddress": "0.0.0.0",
    }
    camera_params = {
        "Format": "ColorBGR8",
        "FOVAngle": 90,
        "Width": 512,
        "Height": 384,
        "bAutoAdvertise": True,
    }
    vehicle_params = False
