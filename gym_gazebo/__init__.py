import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# Gazebo
# ----------------------------------------
register(
	id='GazeboCompEnv-v0',
	entry_point='gym_gazebo.envs.competition_env:GazeboCompEnv',
	max_episode_steps=3000,
)

