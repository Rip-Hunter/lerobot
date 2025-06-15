<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="media/lerobot-logo-thumbnail.png">
    <source media="(prefers-color-scheme: light)" srcset="media/lerobot-logo-thumbnail.png">
    <img alt="LeRobot, Hugging Face Robotics Library" src="media/lerobot-logo-thumbnail.png" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>


<br/>

<h3 align="center">
    <p>List of changes made to the project</p>
</h3>

---

> ⚠️ **Note:** The code is mostly commented to understand what it is responsible for.


<details>
<summary><strong>Сonfigs</strong></summary><br>
<br>

- [**File**](lerobot/common/envs/configs.py)

</details>

<details>
<summary><strong>Modeling_pi0</strong></summary><br>
<br>

- [**File**](lerobot/common/policies/pi0/modeling_pi0.py)

</details>

<details>
<summary><strong>Train</strong></summary><br>
<br>

- [**File**](lerobot/scripts/train.py)

</details>

<details>
<summary><strong>Factory</strong></summary><br>
<br>

- [**File**](lerobot/common/envs/factory.py)

</details>

<details>
<summary><strong>Pi0</strong></summary><br>
<br>

- Added the pi0 folder that was downloaded from [huggingface](https://huggingface.co/lerobot/pi0)
- [**Folder**](pi0/)
---
- Сhanged the model configuration file to suit the characteristics of my robot
- [**File**](pi0/config.json)

</details>


<details>
<summary><strong>Enviroment</strong></summary><br>
<br>

- The [VX300S](https://github.com/google-deepmind/mujoco_menagerie/tree/main/trossen_vx300s) robot was taken as a basis, and it was modified to work with the pi0 model.
1) added multiple cameras
2) added interaction cube
- [**File**](gym_myrobot/vx300s.xml)
---
- Environment created
- [**File**](gym_myrobot/envs/myrobot_env.py)

> ⚠️ **Note:** This environment is a starting point for developing your own projects, and its improvement and optimization are welcomed.

</details>


<!-- 1) Змінено конфіг /lerobot/lerobot/common/envs/configs.py
2) Змінено lerobot/lerobot/common/policies/pi0/modeling_pi0.py
3) Змінено lerobot/lerobot/scripts/train.py
4) Змінено lerobot/lerobot/common/envs/factory.py
5) Додано lerobot/pi0
6) Змінено lerobot/pi0/config.json
 -->



