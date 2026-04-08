# eureka_task_manager.py
import os
import random
import traceback
import multiprocessing
import numpy as np
from datetime import datetime
from typing import Literal

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

try:
    import rl_thor
except ImportError:
    print("❌ rl_thor import error")

AVAILABLE_OBJECT_TYPES = ["Unknown", "Apple", "Bowl", "Bread", "ButterKnife", "Cabinet", "CoffeeMachine", "CounterTop", "Cup", "DishSponge", "Egg", "Faucet", "Floor", "Fork", "Fridge", "GarbageCan",
                "Knife", "Lettuce", "LightSwitch", "Microwave", "Mug", "Pan", "PepperShaker", "Plate", "Pot", "Potato", "SaltShaker", "Sink", "SinkBasin", "SoapBottle",
                "Spatula", "Spoon", "StoveBurner", "StoveKnob", "Toaster", "Tomato"]

OBJECT_TO_ID = {obj: i for i, obj in enumerate(AVAILABLE_OBJECT_TYPES)}

# -------------------------
# Wrapper
# -------------------------
class EurekaThorWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        if hasattr(env.action_space, "spaces"):
            self.action_space = env.action_space["action_index"]
        else:
            self.action_space = env.action_space
            
        orig_obs_space = self.env.observation_space

        self.observation_space = gym.spaces.Dict({
            "env_obs": gym.spaces.Box(0, 255, (300, 300, 3), dtype=np.uint8),

            "center_x": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            "center_y": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            "center_z": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),

            "distance": gym.spaces.Box(0.0, np.inf, shape=(1,), dtype=np.float32),

            "object_type": gym.spaces.Discrete(len(OBJECT_TO_ID)),
            "visible": gym.spaces.Discrete(2),
        })

        # print(orig_obs_space)
        
        # if isinstance(orig_obs_space, gym.spaces.Dict):
        #     new_spaces = dict(orig_obs_space.spaces)
        #     new_spaces["goal_index"] = gym.spaces.Discrete(len(AVAILABLE_OBJECT_TYPES))
        #     self.observation_space = gym.spaces.Dict(new_spaces)
        #     self._is_obs_dict = True
        # else:
        #     self.observation_space = gym.spaces.Dict({
        #         "center_position": self.controller.last_event.metadata["objects"]["axisAlignedBoundingBox"]["center"],
        #         "distance": self.controller.last_event.metadata["objects"]["distance"],
        #         "target_object": {
        #             "object_type": self.controller.last_event.metadata["objects"]["axisAlignedBoundingBox"]["center"]
        #         }
        #     })
        #     self._is_obs_dict = False

        self._eureka_episode_sums = {"eureka_total_rewards": 0.0}
        self._get_rewards_eureka = None

    @property
    def controller(self):
        return self.env.unwrapped.controller

    @property
    def target_object_type(self):
        return getattr(self.env.unwrapped, "target_object_type", AVAILABLE_OBJECT_TYPES)

    def action(self, act):
        return {"action_index": int(act)}
    
    def set_target_object(self, target_object_type):
        for obs in self.controller.last_event.metadata['objects']:
            if obs['objectType'] == target_object_type:
                self.target_object = obs
                break
    
    def build_observation(self, target=None):
        obj_type = target["objectType"]
        obj_id = OBJECT_TO_ID.get(obj_type, OBJECT_TO_ID["Unknown"])

        center = target["axisAlignedBoundingBox"]["center"]
        distance = target["distance"]
        visible = target["visible"]

        return {
            "env_obs": self.env_obs if hasattr(self, "env_obs") else np.zeros((300,300,3), dtype=np.uint8),

            # 🔥 scalar → vector
            "center_x": np.array([center["x"]], dtype=np.float32),
            "center_y": np.array([center["y"]], dtype=np.float32),
            "center_z": np.array([center["z"]], dtype=np.float32),

            "distance": np.array([distance], dtype=np.float32),

            # Discrete는 그대로 OK
            "object_type": np.int64(obj_id),
            "visible": np.int64(visible),
        }

    def get_observation(self):
        self.set_target_object(self.target_object['objectType'])

        target = self.target_object
        obs = self.build_observation(target)

        # print(obs)
        return obs
    
    def _inject_goal_to_obs(self, obs):
        target_name = self.target_object_type

        goal_idx = OBJECT_TO_ID.get(target_name, 0)
        goal_array = np.int64(goal_idx)

        # 반드시 dict 보장
        if not isinstance(obs, dict):
            raise ValueError("Observation must be dict for SB3 MultiInputPolicy")

        obs = dict(obs)  # safety copy

        obs["goal_index"] = goal_array

        return obs

    def step(self, action):
        # 🔥 SB3 → env format 변환
        if not isinstance(action, dict):
            action = {"action_index": int(action)}

        obs, reward, terminated, truncated, info = self.env.step(action)

        self.set_target_object(self.target_object_type)
        obs = self.build_observation(self.target_object)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.env.reset(**kwargs)

        target_type = self.target_object_type
        self.set_target_object(target_type)

        obs = self.build_observation(self.target_object)
        info = {}

        return obs, info

def clean_code(code):
    while isinstance(code, list) and len(code) > 0:
        code = code
    if not isinstance(code, str):
        code = str(code)
        
    code = code.strip()

    if code.startswith("```"):
        parts = code.split("```")
        if len(parts) > 1:
            code = parts
            if code.startswith("python\n") or code.startswith("python"):
                code = code.replace("python", "", 1).strip()
    if code.endswith("```"):
        code = code[:-3]

    return code.strip()

import re

def fix_function_name(code: str):
    match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code)

    if match:
        original_name = match.group(1)
        if original_name != "_get_rewards_eureka":
            print(f"⚠️ Fixing function name: {original_name} → _get_rewards_eureka")
            code = re.sub(
                r"def\s+" + original_name + r"\s*\(",
                "def _get_rewards_eureka(",
                code
            )

    return code

# -------------------------
# Task Manager
# -------------------------
class EurekaTaskManager:
    def __init__(
        self,
        env: str = "rl_thor/ITHOREnv-v0.1",
        num_processes: int = 1,
        device: str = "cuda",
        max_training_iterations: int = 2048,
        config_path: str = None,
        category: str = None
    ):
        self._env = env
        self._num_processes = num_processes
        self._device = device
        self._max_training_iterations = max_training_iterations
        self._config_path = config_path or os.path.expanduser("~/Documents/rl_thor/config/environment_config.yaml")

        raw_env = gym.make(
                    self._env,
                    config_path=self._config_path,
                    config_override={"max_episode_steps": 500}
        )
        self.thor_env = EurekaThorWrapper(raw_env)
        
        target_object_type = self.set_random_target(category)
        self.set_target_object(target_object_type)
        self.thor_env.get_observation()

        self._rewards_queues = [multiprocessing.Queue() for _ in range(self._num_processes)]
        self._results_queue = multiprocessing.Queue()
        self.termination_event = multiprocessing.Event()

        self._processes = {}
        for idx in range(self._num_processes):
            p = multiprocessing.Process(target=self._worker, args=(idx, self._rewards_queues[idx]))
            p.daemon = True 
            self._processes[idx] = p
            p.start()

    def set_random_target(self, category):
        available_target_list = self.get_available_target_list(category)
        return available_target_list[random.randint(0, len(available_target_list)-1)]

    def train(self, reward_data_list):
        for idx, data in enumerate(reward_data_list):
            self._rewards_queues[idx].put(data)

        results = [None] * self._num_processes
        for _ in range(self._num_processes):
            idx, res = self._results_queue.get()
            results[idx] = res

        return results
    
    def get_available_target_list(self, category):
        proper_objects = []
        for obj in self.thor_env.unwrapped.controller.last_event.metadata['objects']:
            if category == 'navigation':
                proper_objects.append(obj['objectType'])
            elif obj[category] and obj['objectType'] in AVAILABLE_OBJECT_TYPES:
                proper_objects.append(obj['objectType'])
        
        return proper_objects
    
    def set_target_object(self, target_object_type):
        self._target_object_type = target_object_type
        self.thor_env.set_target_object(target_object_type)
        
    def _worker(self, idx, queue):
        while not self.termination_event.is_set():
            data = queue.get()
            if data == "Stop":
                break

            reward_code = data["reward_code"]
            success_code = data["success_code"]
            precondition_code = "True"

            try:
                ns = {}
                reward_code = clean_code(reward_code)
                reward_code = fix_function_name(reward_code)

                exec(reward_code, ns)

                if "_get_rewards_eureka" not in ns:
                    raise ValueError("LLM did not define _get_rewards_eureka")

                self.thor_env._get_rewards_eureka = ns["_get_rewards_eureka"]

                MAX_RESET_TRY = 20
                found = False

                for _ in range(MAX_RESET_TRY):
                    obs, _ = self.thor_env.reset()
                    metadata = self.thor_env.unwrapped.controller.last_event.metadata
                    current_target = getattr(self.thor_env.unwrapped, "target_object_type", AVAILABLE_OBJECT_TYPES)
                    
                    try:
                        # 🔥 수정됨: precondition 평가 시에도 TARGET_TYPE 주입
                        if eval(precondition_code, {"metadata": metadata, "TARGET_TYPE": current_target}):
                            found = True
                            break
                    except Exception as e:
                        print(f"⚠️ precondition eval error: {e}")
                        break

                # 🔥 수정됨: 전제 조건을 만족하지 못했을 때 강제 학습 진행 방지 경고
                if not found:
                    print("⚠️ [경고] 최대 Reset 시도에도 precondition을 만족하는 초기 상태를 찾지 못했습니다! (에이전트가 빈 손일 확률이 높음)")

                model = PPO("MultiInputPolicy", self.thor_env, verbose=0, device=self._device)
                model.learn(total_timesteps=self._max_training_iterations)
                model_state_dict = model.policy.state_dict()

                success_count = 0
                episodes = 10

                for _ in range(episodes):
                    for _ in range(MAX_RESET_TRY):
                        obs, _ = self.thor_env.reset()
                        metadata = self.thor_env.unwrapped.controller.last_event.metadata
                        current_target = getattr(self.thor_env.unwrapped, "target_object_type", AVAILABLE_OBJECT_TYPES)
                        
                        try:
                            # 🔥 수정됨: 평가 루프 초기화 시에도 TARGET_TYPE 주입
                            if eval(precondition_code, {"metadata": metadata, "TARGET_TYPE": current_target}):
                                break
                        except:
                            break

                    done = False
                    while not done:
                        action, _ = model.predict(obs)
                        obs, _, terminated, truncated, _ = self.thor_env.step(action)
                        done = terminated or truncated

                        metadata = self.thor_env.unwrapped.controller.last_event.metadata
                        current_target = getattr(self.thor_env.unwrapped, "target_object_type", AVAILABLE_OBJECT_TYPES)

                        try:
                            if eval(success_code, {"metadata": metadata, "TARGET_TYPE": current_target}):
                                success_count += 1
                                break
                        except Exception as e:
                            print(f"⚠️ success eval error: {e}")
                            break

                success_rate = success_count / episodes

                result = {
                    "success": True,
                    "reward_mean": self.thor_env._eureka_episode_sums["eureka_total_rewards"],
                    "success_rate": success_rate,
                    "model_state_dict": model_state_dict
                }

            except Exception as e:
                print(traceback.format_exc())
                result = {"success": False, "exception": str(e)}

            self._results_queue.put((idx, result))

    def close(self):
        self.termination_event.set()
        for q in self._rewards_queues:
            q.put("Stop")
        for p in self._processes.values():
            p.join(timeout=2)