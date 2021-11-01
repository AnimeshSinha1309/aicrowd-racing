from agents.random_agent import RandomAgent


class SubmissionConfig:
    agent = RandomAgent
    pre_eval_time = 100
    eval_episodes = 10


class EnvConfig:
    multimodal = False
    max_timesteps = 2000
    obs_delay = 0.1
    not_moving_timeout = 100
    reward_pol = "custom"
    reward_kwargs = {
        "oob_penalty": 5.0,
        "min_oob_penalty": 25.0,
    }
    controller_kwargs = {
        "sim_version": "ArrivalSim-linux-0.7.1.188691",
        "quiet": True,
        "user": "ubuntu",
        "start_container": False,
        "sim_path": "/home/LinuxNoEditor",
    }
    action_if_kwargs = {
        "max_accel": 4,
        "min_accel": -1,
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
    cameras = {
        "CameraFrontRGB": {
            "Addr": "tcp://0.0.0.0:8008",
            "Format": "ColorBGR8",
            "FOVAngle": 90,
            "Width": 192,
            "Height": 144,
            "bAutoAdvertise": True,
        }
    }


class SimulatorConfig:
    racetrack = ["VegasNorthRoad"]
    active_sensors = [
        "CameraFrontRGB",
    ]
    driver_params = {
        "DriverAPIClass": "VApiUdp",
        "DriverAPI_UDP_SendAddress": "0.0.0.0",
    }
