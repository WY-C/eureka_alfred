# eureka_task_manager.py
import os
import random
import traceback
import multiprocessing
import numpy as np
import torch
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
    """관측치 딕셔너리의 타입을 안정화하고 shape을 맞춤"""
    for k, v in obs.items():
        if k == "env_obs":
            # 시각 정보는 uint8 유지가 일반적 (SB3 내부에서 스케일링 가능)
            obs[k] = v.astype(np.uint8)
            continue

        # 스칼라 혹은 빈 shape 방지 -> (1,)
        if np.isscalar(v):
            obs[k] = np.array([v], dtype=np.float32)
        elif isinstance(v, np.ndarray):
            if v.shape == ():
                obs[k] = v.reshape(1).astype(np.float32)
            else:
                obs[k] = v.astype(np.float32)
        else:
            obs[k] = np.array([v], dtype=np.float32)
    return obs

def _eureka_worker(idx, queue, results_queue, termination_event, env_name, config_path, max_training_iterations, device, target_object_type):
    raw_env = gym.make(env_name, config_path=config_path, config_override={"max_episode_steps": 250})
    thor_env = EurekaThorWrapper(raw_env)
    thor_env.unwrapped.target_object_type = target_object_type

    while not termination_event.is_set():
        data = queue.get()
        if data == "Stop":
            break

        thor_env._eureka_components_history = {}
        thor_env._reward_components_per_epoches = {}
        thor_env.count_try = 0
        thor_env.count_success = 0

        # 🔥 1. [추가] 새로운 학습마다 과거의 성공 상태 리스트를 비워줍니다!
        thor_env.success_states = []

        reward_code = data["reward_code"]
        success_code = data["success_code"]
        precondition_code = "True"

        training_steps = data.get("training_steps", max_training_iterations)

        try:
            
            ns = {  
                "np": np, 
                    "torch": torch,
                    "AVAILABLE_OBJECT_TYPES": AVAILABLE_OBJECT_TYPES,
                    "OBJECT_TO_ID": OBJECT_TO_ID
            }
            reward_code = clean_code(reward_code)
            reward_code = fix_function_name(reward_code)

            exec(reward_code, ns)

            if "_get_rewards_eureka" not in ns:
                raise ValueError("LLM did not define _get_rewards_eureka")

            thor_env._get_rewards_eureka = ns["_get_rewards_eureka"]

            MAX_RESET_TRY = 50
            found = False

            for _ in range(MAX_RESET_TRY):
                obs, _ = thor_env.reset()
                metadata = thor_env.unwrapped.controller.last_event.metadata
                current_target = getattr(thor_env.unwrapped, "target_object_type", None)

                try:
                    if eval(precondition_code, {"metadata": metadata, "TARGET_TYPE": current_target, "env": thor_env}):
                        found = True
                        break
                except Exception as e:
                    print(f"⚠️ precondition eval error: {e}")
                    break

            if not found:
                print("⚠️ [경고] 최대 Reset 시도에도 precondition을 만족하는 초기 상태를 찾지 못했습니다! (에이전트가 빈 손일 확률이 높음)")
            thor_env.count_try = 0
            thor_env.count_success = 0
            model = PPO("MultiInputPolicy", thor_env, ent_coef=0.01, verbose=0, device=device)
            model.learn(total_timesteps=training_steps, progress_bar=True)
            
            raw_state_dict = model.policy.state_dict()
            model_state_dict = {k: v.cpu().clone() for k, v in raw_state_dict.items()}


            train_tries = thor_env.count_try
            train_successes = thor_env.count_success
            train_success_rate = train_successes / max(1, train_tries) # 0으로 나누기 방지
            print(f"[Training Phase] Success: {train_successes}/{train_tries} ({train_success_rate * 100:.2f}%)")


            success_count = 0
            episodes = data.get("eval_episodes", 50)

            episode_rewards = []

            for _ in range(episodes):
                for _ in range(MAX_RESET_TRY):
                    obs, _ = thor_env.reset()
                    metadata = thor_env.unwrapped.controller.last_event.metadata
                    current_target = getattr(thor_env.unwrapped, "target_object_type", AVAILABLE_OBJECT_TYPES)

                    try:
                        # 🔥 수정됨: 평가 루프 초기화 시에도 TARGET_TYPE 주입
                        if eval(precondition_code, {"metadata": metadata, "TARGET_TYPE": current_target}):
                            break
                    except:
                        break

                # Subtask 성공 상태에서 시작하는 로직
                # if thor_env.success_states:
                #     for state in thor_env.success_states[random.randint(0, len(thor_env.success_states) - 1)]:
                #         if state['objectType'] == self._target_object_type:
                #             precondition = state
                #             break

                #     thor_env.unwrapped.controller.step(
                #         action='SetObjectPoses',
                #         objectPoses=[
                #             {
                #                 'objectName': precondition['name'],
                #                 'rotation': {
                #                     'y': precondition['rotation']['y'],
                #                     'x': precondition['rotation']['x'],
                #                     'z': precondition['rotation']['z']
                #                 },
                #                 'position': {
                #                     'y': precondition['position']['y'],
                #                     'x': precondition['position']['x'],
                #                     'z': precondition['position']['z']
                #                 }
                #             }
                #         ]
                #     )
                #     target = thor_env.find_target_object()
                #     obs = thor_env.build_observation(target)
                #     obs = normalize_obs(obs)

                # else:
                #     for _ in range(MAX_RESET_TRY):
                #         obs, _ = thor_env.reset()
                #         metadata = thor_env.unwrapped.controller.last_event.metadata
                #         current_target = getattr(thor_env.unwrapped, "target_object_type", AVAILABLE_OBJECT_TYPES)

                #         try:
                #             # 🔥 수정됨: 평가 루프 초기화 시에도 TARGET_TYPE 주입
                #             if eval(precondition_code, {"metadata": metadata, "TARGET_TYPE": current_target}):
                #                 break
                #         except:
                #             break

                done = False
                while not done:
                    action, _ = model.predict(obs)
                    #print(f'action: {action}')
                    obs, _, terminated, truncated, _ = thor_env.step(action)
                    done = terminated or truncated

                    metadata = thor_env.unwrapped.controller.last_event.metadata
                    current_target = getattr(thor_env.unwrapped, "target_object_type", AVAILABLE_OBJECT_TYPES)

                    try:
                        if eval(success_code, {"metadata": metadata, "TARGET_TYPE": current_target, "env": thor_env}):
                            success_count += 1
                            # print(f'success count increased: {success_count}')
                            thor_env.success_states.append(metadata['objects'])
                            if len(thor_env.success_states) > 200:
                                thor_env.success_states.pop(0)
                            break
                    except Exception as e:
                        print(f"⚠️ success eval error: {e}")
                        break
                episode_rewards.append(
                    thor_env._eureka_episode_sums["eureka_total_rewards"]
                )

            # print(f'final success count: {success_count}')
            # print(f'final episode: {episodes}')
            success_rate = success_count / episodes
            reward_mean = np.mean(episode_rewards)

            result = {
                "success": True,
                "reward_mean": reward_mean,
                "success_rate": success_rate,
                "model_state_dict": model_state_dict,
                "train_success_rate": train_success_rate, # 🔥 학습 도중 성공률
                "train_tries": train_tries,               # 🔥 학습 도중 시도 횟수
                "train_successes": train_successes,       # 🔥 학습 도중 성공 횟수
                "reward_components": thor_env._reward_components_per_epoches,
                "reward_code": reward_code,
                "saved_success_states": thor_env.success_states
            }

        except Exception as e:
            print(traceback.format_exc())
            result = {"success": False, "exception": str(e)}

        results_queue.put((idx, result))

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
            "relative_target_pos": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "distance": gym.spaces.Box(0.0, np.inf, shape=(1,), dtype=np.float32), # 추가
            "visible": gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "is_holding_target": gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            # "agent_yaw_sc"는 일반성을 위해 제거하셨으므로 여기서도 뺍니다.
            "camera_horizon": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
        })
        self._get_rewards_eureka = None
        self._eureka_episode_sums = {"eureka_total_rewards": 0.0}

        self._eureka_components_history = {}
        self._reward_components_per_epoches = {}
        self.count_try = 0
        self.count_success = 0
        self.last_obs = None
        self.prev_obs = None

        self.success_states = []

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
    # -------------------------
    # 🔥 observation 생성
    # -------------------------
    def build_observation(self, target):
        """
        Builds ego-centric observations. 
        relative_target_pos indices:
        [0]: Left/Right vector (x)
        [1]: Up/Down vector (y)
        [2]: Forward/Backward vector (z) -> The direction the agent is facing.
        """
        meta = self.controller.last_event.metadata
        agent = meta["agent"]
        
        # 1. 회전각 계산
        yaw_rad = np.deg2rad(agent["rotation"]["y"])
        sin_yaw = np.sin(yaw_rad)
        cos_yaw = np.cos(yaw_rad)

        camera_pitch = agent["cameraHorizon"] / 90


        if target is None:
            ego_rel_pos = np.zeros(3, dtype=np.float32)
            dist, vis = 0.0, 0.0
        else:
            t_pos = target["axisAlignedBoundingBox"]["center"]
            dx = t_pos["x"] - agent["position"]["x"]
            dy = t_pos["y"] - agent["position"]["y"]
            dz = t_pos["z"] - agent["position"]["z"]
            
            # 🔥 Ego-centric 변환 적용
            ego_x =  cos_yaw * dx + sin_yaw * dz
            ego_z = -sin_yaw * dx + cos_yaw * dz
            ego_y = dy
            
            ego_rel_pos = np.array([ego_x, ego_y, ego_z], dtype=np.float32)
            dist, vis = target["distance"], float(target["visible"])

        is_holding = any(obj["objectType"] == self.target_object_type for obj in meta.get("inventoryObjects", []))

        # 🔥 반환하는 딕셔너리의 키가 observation_space와 1:1 대응되어야 함
        obs = {
            "relative_target_pos": ego_rel_pos,
            "distance": np.array([dist], dtype=np.float32), # 배열화
            "visible": np.array([vis], dtype=np.float32),   # 배열화
            "is_holding_target": np.array([1.0 if is_holding else 0.0], dtype=np.float32), # 배열화
            "camera_horizon": np.array([camera_pitch], dtype=np.float32)
            
        }
        return obs # normalize_obs는 나중에 step/reset에서 호출됨

    # -------------------------
    # step
    # -------------------------
    def step(self, action):
        if not isinstance(action, dict):
            action = {"action_index": int(action)}
        self.prev_obs = self.last_obs

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
            self.count_success += 1
            # print('subtask성공')
            self.success_states.append(self.controller.last_event.metadata['objects'])
            # 2번 태스크를 위해 모아두되, 최대 200개까지만 유지 (OOM 완벽 방어)
            if len(self.success_states) > 200:
                self.success_states.pop(0)
            # print(f'success_states added: {self.success_states[-1]}')

        if terminated or truncated:
            if self._get_rewards_eureka is not None:
                for reward_component in reward_dict:
                    if reward_component not in self._reward_components_per_epoches:
                        self._reward_components_per_epoches[reward_component] = []
                    
                    self._reward_components_per_epoches[reward_component].append(
                        sum(self._eureka_components_history.get(reward_component, [0.0]))
                )
            #print("self._reward_components_per_epoches:", self._reward_components_per_epoches)
            info["terminal_observation"] = normalize_obs(obs)


        return obs, reward, terminated, truncated, info

    # -------------------------
    # reset
    # -------------------------
    def reset(self, **kwargs):
        self.count_try += 1
        _, info = self.env.reset(**kwargs)

        # 🔥 에피소드 reward 초기화
        self._eureka_episode_sums["eureka_total_rewards"] = 0.0

        target = self.find_target_object()
        obs = self.build_observation(target)
        obs = normalize_obs(obs)

        self._eureka_components_history = {}
        #self._reward_components_per_epoches = {}

        self.last_obs = obs
        self.prev_obs = obs

        return obs, info
    
    def reset_eureka_components_history(self):
        self._eureka_components_history = {}

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


        temp_env = gym.make(self._env, config_path=self._config_path, config_override={"max_episode_steps": 250})
        temp_wrapper = EurekaThorWrapper(temp_env)
        temp_wrapper.reset()
        
        proper_objects = []
        for obj in temp_wrapper.unwrapped.controller.last_event.metadata['objects']:
            if category == 'navigate': proper_objects.append(obj['objectType'])
            elif obj.get(category) and obj['objectType'] in AVAILABLE_OBJECT_TYPES: proper_objects.append(obj['objectType'])
        
        # self._target_object_type = proper_objects[random.randint(0, max(0, len(proper_objects)-1))] if proper_objects else 'Mug'
        # self._target_object_type = 'Mug' # 🔥 일단 고정 (나중에 랜덤으로 바꿀 수 있음)
        temp_wrapper.close() # 임시 환경 종료

        raw_env = gym.make(
                    self._env,
                    config_path=self._config_path,
                    config_override={
                                        "max_episode_steps": 250,
                                     }
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
            p = multiprocessing.Process(
                target=_eureka_worker, 
                args=(
                    idx, 
                    self._rewards_queues[idx], 
                    self._results_queue,
                    self.termination_event,
                    self._env,
                    self._config_path,
                    self._max_training_iterations,
                    self._device,
                    self._target_object_type # 필요한 데이터만 쏙쏙 골라서 전달
                )
            )
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
        

    def close(self):
        self.termination_event.set()
        for q in self._rewards_queues:
            q.put("Stop")
        for p in self._processes.values():
            p.join(timeout=2)