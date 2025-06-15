from gymnasium.envs.registration import register

register(
    id="myrobot-v0",
    entry_point="gym_myrobot.envs:MyRobotEnv",
    max_episode_steps=200,
    kwargs={"xml_path": "/home/ivan/HNURE/Diploma/Test/home/lerobot/lerobot/lerobot/gym_myrobot/gym_myrobot/vx300s.xml"}
)