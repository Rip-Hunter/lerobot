import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv


print("MujocoEnv metadata:", MujocoEnv.metadata)

class MyRobotEnv(MujocoEnv, gym.utils.EzPickle):
	"""
	Custom Robot Environment with MuJoCo based on the provided XML files
	"""

	metadata = {**MujocoEnv.metadata}  # Inherit and then modify as needed
	metadata["render_fps"] = 50  # Add any custom metadata elements


	def __init__(
		self,
		xml_path = './lerobot/lerobot/lib/python3.11/site-packages/gym_myrobot/vx300s.xml',   ### path to the xml file of the robot ###
		render_mode="rgb_array",
		max_episode_steps=600,
		reward_type="sparse", # dense
		task="reach",
		target_position=None,
		camera_ids=None,
		image_width=224,
		image_height=224,
		**kwargs
	):

		self.max_episode_steps = max_episode_steps
		self.reward_type = reward_type
		self.task = task
		self.target_position = target_position or np.array([0.2, 0.0, 0.2])  # Цільова позиція за замовчуванням
		
		### Параметри для камер ###
		### Camera Settings ###
		self.camera_ids = camera_ids or ["gripper_top_camera", "front_left_camera", "front_right_camera"]
		self.image_width = image_width
		self.image_height = image_height

		self.success_steps = 0 
		self.success_threshold = 50   ### How many steps to wait before success ###

		### Перевірка наявності XML файлу ###
		### Checking for the presence of an XML file ###
		if not os.path.exists(xml_path):
			raise IOError(f"XML file does not exist: {xml_path}")
		
		### Ініціалізація MuJoCo середовища ###
		### Initializing the MuJoCo environment ###
		MujocoEnv.__init__(
			self,
			model_path=xml_path,
			frame_skip=10,
			observation_space=None,
			default_camera_config={"trackbodyid": 0, "distance": 1.5},
			render_mode=render_mode,
		)
		
		### Визначення простору спостережень після ініціалізації моделі ###
		### Determining the observation space after model initialization ###
		self.observation_space = self._get_observation_space()
		
		### Визначення простору дій (7 параметрів для маніпулятора з 6 DOF + захоплення) ###
		### Defining the action space (7 parameters for a 6 DOF manipulator + capture) ###
		self.action_space = spaces.Box(
			low=-1.0, high=1.0, shape=(7,), dtype=np.float32
		)

		gym.utils.EzPickle.__init__(
			self, xml_path, render_mode, max_episode_steps, reward_type, task, target_position, 
			camera_ids, image_width, image_height, **kwargs
		)
		
		### Початкова позиція ###
		### Starting position ###
		self.init_qpos = self.data.qpos.copy()
		
		### Лічильник кроків ###
		### Step counter ###
		self.step_count = 0
		
		### Додаємо цільовий об'єкт для візуалізації (якщо потрібно) ###
		### Add a target object for visualization (if necessary) ###
		if hasattr(self, 'model') and self.model is not None:
			self._setup_target_site()

	def _get_observation_space(self):
		""" Визначення простору спостережень на основі стану робота """
		""" Determining the observation space based on the robot state """

		return spaces.Dict({
			"agent_pos": spaces.Box(
				low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32),

			"pixels": spaces.Box(low=0, high=255, shape=( self.image_height, self.image_width, 3), dtype=np.uint8),
		})

	def _get_camera_images(self):
		""" Отримання зображень з камер """
		""" Receiving images from cameras """

		images = {}
		for camera_id in self.camera_ids:
			try:
				if self.render_mode != "rgb_array":
					### Використовуємо заглушку, якщо режим рендерингу не дозволяє отримувати зображення ###
					### Use a stub if the rendering mode does not allow you to get an image ###
					print("ERROR!!!!!! NON  IMG")
					continue
					
				### Використання методу render для отримання зображення (без параметра mode) ###
				### Using the render method to get an image (without the mode parameter) ###
				img = self.render(
					width=self.image_width,
					height=self.image_height,
					camera_name=camera_id
				)
				images[camera_id] = img


			except Exception as e:
				print(f"Error getting image from camera {camera_id}: {e}")
				images[camera_id] = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
		

		return images


	def render(self, width=None, height=None, camera_name='gripper_top_camera'):
		""" Рендеринг зображення з заданої камери """
		""" Rendering an image from a given camera """

		if camera_name is not None:
			### Використовуємо конкретну камеру ###
			### We use a specific camera ###
			return self._render_camera(width, height, camera_name)
		else:
			### Використовуємо стандартний рендер ###
			### We use the standard render ###
			return super().render()

	def _render_camera(self, width, height, camera_name):
		""" Рендеринг зображення з конкретної камери """
		""" Rendering an image from a specific camera """
		if width is None:
			width = self.image_width
		if height is None:
			height = self.image_height
		
		### Отримуємо ID камери ###
		### Get camera ID ###
		cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
		if cam_id < 0:
			print(f"Camera {camera_name} not found in the model")
			return np.zeros((height, width, 3), dtype=np.uint8)
		
		### Оновлюємо дані симуляції ###
		### Updating simulation data ###
		mujoco.mj_forward(self.model, self.data)
		
		### Налаштування камери та рендеринг ###
		### Camera Settings and Rendering ###
		renderer = mujoco.Renderer(self.model, height=height, width=width)
		renderer.update_scene(self.data, camera=camera_name)
		img = renderer.render()

		return img


	def _get_obs(self):
		""" Отримання поточного спостереження за станом робота """
		""" Get current robot health monitoring """

		### Отримати позиції та швидкості суглобів ###
		### Get joint positions and velocities ###
		qpos = self.data.qpos.flat.copy()
		qvel = self.data.qvel.flat.copy()

		### Координати кінцевого ефектора (захоплення) ###
		### End effector coordinates (gripper) ###
		site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
		gripper_pos = self.data.site_xpos[site_id]
		
		### Об'єднуємо всі дані стану ###
		### Combining all state data ###
		robot_state = np.concatenate([
			qpos[-8:],                # Joint positions
			### Can be expanded as needed ###
		])

		### Об'єднуємо всі дані в одне спостереження ###
		### Combining all data into one observation ###	
		obs = {
			"agent_pos": robot_state.astype(np.float32),
			"pixels": None,
			### Can be expanded as needed ###
		}

		### Отримання зображень з камер ###
		### Getting images from cameras ###
		camera_images = self._get_camera_images()

		for camera_id, image in camera_images.items():
			if camera_id == 'gripper_top_camera':
				obs["pixels"] = image.astype(np.uint8)# / 255.0)

		return obs


	def reset(self, seed=None, options=None):
		""" Скидання середовища до початкового стану """
		""" Reset environment to initial state """

		self.success_steps = 0

		### Скидання MujocoEnv ###
		### MujocoEnv Reset ###
		super().reset(seed=seed)
		
		self.step_count = 0
		
		### Встановлення початкового стану на основі ключового кадру "home" з XML ###
		### Setting the initial state based on the "home" keyframe from XML ###
		if hasattr(self.model, 'key_qpos'):
			home_pose_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, 'home')
			if home_pose_idx >= 0:
				qpos = self.model.key_qpos[home_pose_idx].copy()
				qvel = np.zeros_like(self.data.qvel)

				self.set_state(qpos, qvel)
		
		### Отримання спостереження після скидання ###
		### Getting Observation After Reset ###
		obs = self._get_obs()
		info = {}
		
		return obs, info

	def step(self, action):
		""" Виконання кроку в середовищі """
		""" Executing a step in the environment """

		### Застосувати нормалізовану дію до робота (від -1 до 1 перетворити на реальні обмеження суглобів) ###
		### Apply normalized action to robot (convert from -1 to 1 to real joint constraints) ###
		action = self._normalize_action(action)
		
		### Виконати симуляцію ###
		### Run Simulation ###
		self.do_simulation(action, self.frame_skip)
		
		### Отримати спостереження ###
		### Get Observations ###
		obs = self._get_obs()
		
		### Обчислити нагороду ###
		### Calculate reward ###
		reward = self._compute_reward(obs)
		
		### Перевірити чи завдання виконано ###
		### Check if the task is completed ###
		self.step_count += 1
		terminated = self._is_success(obs)
		truncated = self.step_count >= self.max_episode_steps

		info = {
			"is_success": terminated,
			"distance_to_target": np.linalg.norm(
				self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch")] - self.target_position
			),
		}
		
		return obs, reward, terminated, truncated, info


	def _normalize_action(self, action):
		""" Перетворення нормалізованих дій (-1, 1) на реальні значення для суглобів """
		""" Converting normalized actions (-1, 1) to real values ​​for joints """

		### Отримання обмежень для кожного суглоба ###
		### Getting constraints for each joint ###
		ctrl_range = self.model.actuator_ctrlrange
		
		### Масштабування дій до діапазону суглобів ###
		### Scaling actions to a range of joints ###
		scaled_action = np.zeros_like(action)
		for i in range(len(action)):
			if i < len(ctrl_range):
				ctrl_min, ctrl_max = ctrl_range[i]
				scaled_action[i] = 0.5 * (action[i] + 1.0) * (ctrl_max - ctrl_min) + ctrl_min
		
		return scaled_action

	def _compute_reward(self, obs):
		""" Обчислення нагороди на основі поточного стану """
		""" Calculating reward based on current status """

		### Позиція захоплення ###
		### Gripper position ###
		site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
		gripper_pos = self.data.site_xpos[site_id]

		
		### Отримуємо позицію кубика ###
		### Get the position of the cube ###
		cube_pos = self.data.qpos[:3].copy()

		### Отримуємо стан захвату (відкритий/закритий) ###
		### Get the grip state ###
		gripper_state = self.data.qpos[-2].copy()

		### Відстань від захвату до кубика ###
		### Distance from grip to cube ###
		distance_to_cube = np.linalg.norm(gripper_pos - cube_pos)
		
		### Висота кубика над підлогою (підлога на висоті 0) ###
		### Height of the cube above the floor (floor at height 0) ###
		cube_height = cube_pos[2]
		
		### Визначаємо чи тримає робот кубик (кубик піднятий і захват закритий) ###
		### Determine whether the robot is holding a cube (cube is raised and gripper is closed) ###
		holding_cube = cube_height > 0.1 and distance_to_cube < 0.05 and gripper_state < 0.03

		### Цільова позиція перед роботом (можна налаштувати за потребою) ###
		### Target position in front of the robot (can be customized as needed) ###
		target_height = 0.3
		target_position = np.array([0.3, 0, target_height])
		
		### Відстань від кубика до цільової позиції ###
		### Distance from cube to target position ###
		distance_to_target = np.linalg.norm(cube_pos - target_position)
		
		if self.reward_type == "sparse":
			### Повертаємо 1, якщо завдання успішне, інакше 0 ###
			### Return 1 if the task is successful, otherwise 0 ###
			return float(self._is_success(obs))
		else:
			### Щільна нагорода з кількома компонентами ###
			### Dense reward with multiple components ###
			### Can be configured to track more accurate task progress if necessary ###
			reward = 0.0

			reward_reach = 0.8 - np.tanh(3.0 * distance_to_cube)

			reward_grasp = 2.0 if distance_to_cube < 0.04 and gripper_state < 0.03 else 0.0

			reward_lift = 3.0 * cube_height if cube_height > 0.05 else 0.0

			reward_target = 2.0 * (1.0 - np.tanh(2.0 * distance_to_target)) if holding_cube else 0.0
			
			lift_success = 4 if cube_height > 0.5 and cube_height < 0.6 and distance_to_cube < 0.08 else 0.0

			reward_success = 10.0 if self._is_success(obs) else 0.0

			reward = reward_reach + reward_grasp + reward_lift + reward_target + lift_success + reward_success

		return reward


	def _is_success(self, obs):
		""" Перевірка успішності виконання завдання """
		""" Checking the success of the task """

		site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
		gripper_pos = self.data.site_xpos[site_id]

		cube_pos = self.data.qpos[:3].copy()

		gripper_state = self.data.qpos[-2].copy()

		cube_height = cube_pos[2]
		
		target_height = 0.3
		target_position = np.array([0.3, 0, target_height])
		distance_to_cube = np.linalg.norm(gripper_pos - cube_pos)
		
		### Відстань від кубика до цільової позиції ###
		### Distance from cube to target position ###
		distance_to_target = np.linalg.norm(cube_pos - target_position)

		if cube_height > 0.5 and cube_height < 0.6 and distance_to_cube < 0.08:
			self.success_steps += 1
		else:
			self.success_steps = 0

		### Success conditions:
		# 1. The cube is raised high enough
		# 2. The cube is close to the target position
		# 3. The grip is holding the cube (grip is closed)
		return (self.success_steps >= self.success_threshold)
		