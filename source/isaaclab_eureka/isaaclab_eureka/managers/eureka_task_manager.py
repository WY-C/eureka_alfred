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

# 🔥 에이전트가 학습할 타겟 객체들의 전체 사전
TARGET_VOCAB = [
    "Mug", "Apple", "Tomato", "Bowl", "Laptop", 
    "Book", "CellPhone", "Pen", "Pencil", "Remote",
    "Statue", "Vase", "Cup", "Plate", "SoapBottle"
]

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
        
        if isinstance(orig_obs_space, gym.spaces.Dict):
            new_spaces = dict(orig_obs_space.spaces)
            new_spaces["goal_index"] = gym.spaces.Discrete(len(TARGET_VOCAB))
            self.observation_space = gym.spaces.Dict(new_spaces)
            self._is_obs_dict = True
        else:
            self.observation_space = gym.spaces.Dict({
                "vision": orig_obs_space,
                "goal_index": gym.spaces.Discrete(len(TARGET_VOCAB))
            })
            self._is_obs_dict = False

        self._eureka_episode_sums = {"eureka_total_rewards": 0.0}
        self._get_rewards_eureka = None

    @property
    def controller(self):
        return self.env.unwrapped.controller

    @property
    def target_object_type(self):
        return getattr(self.env.unwrapped, "target_object_type", TARGET_VOCAB)

    def action(self, act):
        return {"action_index": int(act)}
    
    def get_interacted_objects(self):
        metadata = self.env.unwrapped.controller.last_event.metadata
        objects = metadata.get("objects", [])
        
        interactable_state = {
            "inventory": metadata.get("inventoryObjects", []),
            "pickupable_types": list(set(obj.get("objectType") for obj in objects if obj.get("pickupable"))),
            "openable_types": list(set(obj.get("objectType") for obj in objects if obj.get("openable"))),
            "toggleable_types": list(set(obj.get("objectType") for obj in objects if obj.get("toggleable"))),
            "receptacle_types": list(set(obj.get("objectType") for obj in objects if obj.get("receptacle")))
        }
        return interactable_state

    def _inject_goal_to_obs(self, obs):
        target_name = self.target_object_type
        
        if target_name in TARGET_VOCAB:
            goal_idx = TARGET_VOCAB.index(target_name)
        else:
            goal_idx = 0 
            
        goal_array = np.array(goal_idx, dtype=np.int64)

        if self._is_obs_dict:
            obs["goal_index"] = goal_array
            return obs
        else:
            return {
                "vision": obs,
                "goal_index": goal_array
            }

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(self.action(action))
        obs = self._inject_goal_to_obs(obs)

        reward = 0
        rewards_dict = {}

        if self._get_rewards_eureka:
            try:
                result = self._get_rewards_eureka(self)

                if isinstance(result, tuple) and len(result) == 2:
                    reward, rewards_dict = result
                else:
                    print("⚠️ Invalid reward return format. Using fallback.")
                    reward = float(result) if isinstance(result, (int, float)) else 0.0
                    rewards_dict = {}
            except Exception as e:
                print(f"⚠️ reward error: {e}")
                raise RuntimeError(f"Reward function failed: {e}")

        self._eureka_episode_sums["eureka_total_rewards"] += float(reward)
        return obs, float(reward), terminated, truncated, info

    def reset(self, **kwargs):
        self._eureka_episode_sums = {"eureka_total_rewards": 0.0}
        obs, info = self.env.reset(**kwargs)
        
        metadata = self.env.unwrapped.controller.last_event.metadata
        objects = metadata.get("objects", [])
        
        available_targets = list(set(
            obj.get("objectType") for obj in objects 
            if obj.get("pickupable") and obj.get("objectType") in TARGET_VOCAB
        ))
        
        if available_targets:
            chosen_target = random.choice(available_targets)
            self.env.unwrapped.target_object_type = chosen_target
        else:
            self.env.unwrapped.target_object_type = TARGET_VOCAB
            
        obs = self._inject_goal_to_obs(obs)
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
        config_path: str = None
    ):
        self._env = env
        self._num_processes = num_processes
        self._device = device
        self._max_training_iterations = max_training_iterations
        self._config_path = config_path or os.path.expanduser("~/Documents/rl_thor/config/environment_config.yaml")

        self._rewards_queues = [multiprocessing.Queue() for _ in range(self._num_processes)]
        self._results_queue = multiprocessing.Queue()
        self.termination_event = multiprocessing.Event()

        self._processes = {}
        for idx in range(self._num_processes):
            p = multiprocessing.Process(target=self._worker, args=(idx, self._rewards_queues[idx]))
            p.daemon = True 
            self._processes[idx] = p
            p.start()

    def train(self, reward_data_list):
        for idx, data in enumerate(reward_data_list):
            self._rewards_queues[idx].put(data)

        results = [None] * self._num_processes
        for _ in range(self._num_processes):
            idx, res = self._results_queue.get()
            results[idx] = res

        return results

    def _worker(self, idx, queue):
        raw_env = gym.make(
            self._env,
            config_path=self._config_path,
            config_override={"max_episode_steps": 500}
        )
        env = EurekaThorWrapper(raw_env)

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

                env._get_rewards_eureka = ns["_get_rewards_eureka"]

                MAX_RESET_TRY = 20
                found = False

                for _ in range(MAX_RESET_TRY):
                    obs, _ = env.reset()
                    metadata = env.unwrapped.controller.last_event.metadata
                    current_target = getattr(env.unwrapped, "target_object_type", TARGET_VOCAB)
                    
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

                model = PPO("MultiInputPolicy", env, verbose=0, device=self._device)
                model.learn(total_timesteps=self._max_training_iterations)
                model_state_dict = model.policy.state_dict()

                success_count = 0
                episodes = 10

                for _ in range(episodes):
                    for _ in range(MAX_RESET_TRY):
                        obs, _ = env.reset()
                        metadata = env.unwrapped.controller.last_event.metadata
                        current_target = getattr(env.unwrapped, "target_object_type", TARGET_VOCAB)
                        
                        try:
                            # 🔥 수정됨: 평가 루프 초기화 시에도 TARGET_TYPE 주입
                            if eval(precondition_code, {"metadata": metadata, "TARGET_TYPE": current_target}):
                                break
                        except:
                            break

                    done = False
                    while not done:
                        action, _ = model.predict(obs)
                        obs, _, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated

                        metadata = env.unwrapped.controller.last_event.metadata
                        current_target = getattr(env.unwrapped, "target_object_type", TARGET_VOCAB)

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
                    "reward_mean": env._eureka_episode_sums["eureka_total_rewards"],
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