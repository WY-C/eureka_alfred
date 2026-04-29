# eureka_task_manager.py
import os
import random
import traceback
import multiprocessing
import numpy as np
from datetime import datetime
from typing import Literal, Tuple, Dict

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

def normalize_obs(obs):
    for k, v in obs.items():
        if k == "env_obs":
            continue

        # scalar → (1,)
        if np.isscalar(v):
            obs[k] = np.array([v], dtype=np.float32)

        # () → (1,)
        elif isinstance(v, np.ndarray) and v.shape == ():
            obs[k] = v.reshape(1).astype(np.float32)

        # dtype 통일
        elif isinstance(v, np.ndarray):
            obs[k] = v.astype(np.float32)

    return obs

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

        self.observation_space = gym.spaces.Dict({
            "env_obs": gym.spaces.Box(0, 255, (300, 300, 3), dtype=np.uint8),

            "center_x": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            "center_y": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            "center_z": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),

            "distance": gym.spaces.Box(0.0, np.inf, shape=(1,), dtype=np.float32),

            "object_type": gym.spaces.Box(0, len(OBJECT_TO_ID), shape=(1,), dtype=np.int64),
            "visible": gym.spaces.Box(0, 1, shape=(1,), dtype=np.int64),
        })

        self._get_rewards_eureka = None
        self._eureka_episode_sums = {"eureka_total_rewards": 0.0}

        self._eureka_components_history = {}
        self._reward_components_per_epoches = {}

        self.last_obs = None

    @property
    def controller(self):
        return self.env.unwrapped.controller

    @property
    def target_object_type(self):
        return getattr(self.env.unwrapped, "target_object_type", None)

    # -------------------------
    # 🔥 target 찾기 (핵심)
    # -------------------------
    def find_target_object(self):
        target_type = self.target_object_type
        if target_type is None:
            return None

        for obj in self.controller.last_event.metadata["objects"]:
            if obj["objectType"] == target_type:
                return obj

        return None
    
    def get_interacted_objects(self):
        metadata = self.controller.last_event.metadata

        interacted = {
            "inventory": metadata.get("inventoryObjects", []),
            "opened": [],
            "toggled": [],
            "broken": []
        }

        for obj in metadata.get("objects", []):
            if obj.get("isOpen", False):
                interacted["opened"].append(obj)

            if obj.get("isToggled", False):
                interacted["toggled"].append(obj)

            if obj.get("isBroken", False):
                interacted["broken"].append(obj)

        return interacted

    # -------------------------
    # 🔥 observation 생성
    # -------------------------
    def build_observation(self, target):
        if target is None:
            obs = {
                "env_obs": np.zeros((300,300,3), dtype=np.uint8),
                "center_x": np.array([0.0], dtype=np.float32),
                "center_y": np.array([0.0], dtype=np.float32),
                "center_z": np.array([0.0], dtype=np.float32),
                "distance": np.array([0.0], dtype=np.float32),
                "object_type": np.array([0], dtype=np.float32),
                "visible": np.array([0], dtype=np.float32),
            }
            self.last_obs = obs
            return obs

        obj_id = OBJECT_TO_ID.get(target["objectType"], 0)
        center = target["axisAlignedBoundingBox"]["center"]

        obs = {
            "env_obs": np.zeros((300,300,3), dtype=np.uint8),

            "center_x": np.array([center["x"]], dtype=np.float32),
            "center_y": np.array([center["y"]], dtype=np.float32),
            "center_z": np.array([center["z"]], dtype=np.float32),

            "distance": np.array([target["distance"]], dtype=np.float32),

            "object_type": np.int64(obj_id),
            "visible": np.array([target["visible"]], dtype=np.int64),
        }

        # 🔥 핵심: 저장
        self.last_obs = obs

        return obs

    # -------------------------
    # step
    # -------------------------
    def step(self, action):
        if not isinstance(action, dict):
            action = {"action_index": int(action)}

        obs, reward, terminated, truncated, info = self.env.step(action)

        target = self.find_target_object()
        obs = self.build_observation(target)
        obs = normalize_obs(obs)

        self.last_obs = obs


        

        # 🔥 reward shaping
        if self._get_rewards_eureka is not None:
            #print(self._get_rewards_eureka(self))
            r, reward_dict = self._get_rewards_eureka(self)
            for reward_component in reward_dict:
                if reward_component not in self._eureka_components_history:
                    self._eureka_components_history[reward_component] = []
                self._eureka_components_history[reward_component].append(reward_dict[reward_component])
            reward += r
            self._eureka_episode_sums["eureka_total_rewards"] += r

        # print(self.controller.last_event.metadata.get('inventoryObjects'))
        if self.controller.last_event.metadata.get('inventoryObjects') and self.controller.last_event.metadata.get('inventoryObjects')[0]['objectType'] == self.target_object_type:
            terminated = True
            print('상황종료됨')
            # print('상황종료됨')
            # print('상황종료됨')
        # 🔥🔥🔥 핵심 추가
        if terminated or truncated:
            for reward_component in reward_dict:
                if reward_component not in self._reward_components_per_epoches:
                    self._reward_components_per_epoches[reward_component] = []
                self._reward_components_per_epoches[reward_component].append(
                    sum(self._eureka_components_history.get(reward_component, [0.0]))
                )
            print("self._reward_components_per_epoches:", self._reward_components_per_epoches)
            info["terminal_observation"] = normalize_obs(obs)



        return obs, reward, terminated, truncated, info

    # -------------------------
    # reset
    # -------------------------
    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)

        # 🔥 에피소드 reward 초기화
        self._eureka_episode_sums["eureka_total_rewards"] = 0.0

        target = self.find_target_object()
        obs = self.build_observation(target)
        obs = normalize_obs(obs)

        self._eureka_components_history = {}
        #self._reward_components_per_epoches = {}

        return obs, info
    
    def reset_reward_components_per_epoches(self):
        self._reward_components_per_epoches = {}


def clean_code(code):
    # 1. list면 첫 번째 요소 사용
    if isinstance(code, list):
        code = code[0]

    # 2. 무조건 string으로 변환
    code = str(code)

    code = code.strip()

    # 3. ``` 제거
    if code.startswith("```"):
        code = code.strip("`")

        # python prefix 제거
        if code.startswith("python"):
            code = code[len("python"):].strip()

    # 4. 끝에 ``` 제거
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
        target_object_type = 'Mug'
        self.set_target_object(target_object_type)

        # 🔥 반드시 reset 이후에 object 잡힘
        self.thor_env.reset()

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
        self.thor_env.unwrapped.target_object_type = target_object_type
        
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
                    current_target = getattr(self.thor_env.unwrapped, "target_object_type", None)
                    
                    try:
                        # 🔥 수정됨: precondition 평가 시에도 TARGET_TYPE 주입
                        if eval(precondition_code, {"metadata": metadata, "TARGET_TYPE": current_target, "env": self.thor_env}):
                            found = True
                            break
                    except Exception as e:
                        print(f"⚠️ precondition eval error: {e}")
                        break

                # 🔥 수정됨: 전제 조건을 만족하지 못했을 때 강제 학습 진행 방지 경고
                if not found:
                    print("⚠️ [경고] 최대 Reset 시도에도 precondition을 만족하는 초기 상태를 찾지 못했습니다! (에이전트가 빈 손일 확률이 높음)")

                model = PPO("MultiInputPolicy", self.thor_env, ent_coef=0.01, verbose=0, device=self._device)
                model.learn(total_timesteps=self._max_training_iterations)
                model_state_dict = model.policy.state_dict()

                success_count = 0
                episodes = 10

                episode_rewards = []

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
                        #print(f'action: {action}')
                        obs, _, terminated, truncated, _ = self.thor_env.step(action)
                        done = terminated or truncated

                        metadata = self.thor_env.unwrapped.controller.last_event.metadata
                        current_target = getattr(self.thor_env.unwrapped, "target_object_type", AVAILABLE_OBJECT_TYPES)

                        try:
                            if eval(success_code, {"metadata": metadata, "TARGET_TYPE": current_target, "env": self.thor_env}):
                                success_count += 1
                                print(f'success count increased: {success_count}')
                                break
                        except Exception as e:
                            print(f"⚠️ success eval error: {e}")
                            break
                    episode_rewards.append(
                        self.thor_env._eureka_episode_sums["eureka_total_rewards"]
                    )

                print(f'final success count: {success_count}')
                print(f'final episode: {episodes}')
                success_rate = success_count / episodes
                reward_mean = np.mean(episode_rewards)

                result = {
                    "success": True,
                    "reward_mean": reward_mean,
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