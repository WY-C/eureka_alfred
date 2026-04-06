# eureka_task_manager.py
import os
import traceback
import multiprocessing
from datetime import datetime
from typing import Literal

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

try:
    import rl_thor
except ImportError:
    print("❌ rl_thor import error")

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

        self._eureka_episode_sums = {"eureka_total_rewards": 0.0}
        self._get_rewards_eureka = None

    def action(self, act):
        return {"action_index": int(act)}

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(self.action(action))

        reward = 0
        rewards_dict = {}

        if self._get_rewards_eureka:
            try:
                result = self._get_rewards_eureka(self.env.unwrapped)
                # print(f"[DEBUG reward result]: {result}")

                # 🔥 안전 처리
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
        return self.env.reset(**kwargs)

def clean_code(code: str):
    code = code.strip()

    # ``` 제거
    if code.startswith("```"):
        code = code.split("```")[1]  # python 코드 부분
    if code.endswith("```"):
        code = code[:-3]

    return code

import re

def fix_function_name(code: str):
    # 함수 정의 찾기
    match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code)

    if match:
        original_name = match.group(1)

        # 이름이 다르면 교체
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
        task: str = "rl_thor/ITHOREnv-v0.1",
        num_processes: int = 1,
        device: str = "cuda",
        max_training_iterations: int = 2048,
        config_path: str = None
    ):
        self._task = task
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
            self._processes[idx] = p
            p.start()

    # -------------------------
    def train(self, reward_data_list):
        for idx, data in enumerate(reward_data_list):
            self._rewards_queues[idx].put(data)

        results = [None] * self._num_processes
        for _ in range(self._num_processes):
            idx, res = self._results_queue.get()
            results[idx] = res

        return results

    # -------------------------
    def _worker(self, idx, queue):
        raw_env = gym.make(
            self._task,
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

            try:
                # compile reward
                ns = {}
                reward_code = clean_code(reward_code)
                reward_code = fix_function_name(reward_code)
                exec(f"{reward_code}", ns)
                if "_get_rewards_eureka" not in ns:
                    raise ValueError("LLM did not define _get_rewards_eureka")
                env._get_rewards_eureka = ns["_get_rewards_eureka"]

                # train
                model = PPO("MultiInputPolicy", env, verbose=0, device=self._device)
                model.learn(total_timesteps=self._max_training_iterations)

                # -------------------------
                # 🔥 success evaluation
                # -------------------------
                success_count = 0
                episodes = 10

                for _ in range(episodes):
                    obs, _ = env.reset()
                    done = False

                    while not done:
                        action, _ = model.predict(obs)
                        obs, _, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated

                        metadata = env.unwrapped.controller.last_event.metadata

                        try:
                            if eval(success_code, {"metadata": metadata}):
                                success_count += 1
                                break
                        except Exception as e:
                            print(f"⚠️ success eval error: {e}")
                            break

                success_rate = success_count / episodes

                result = {
                    "success": True,
                    "reward_mean": env._eureka_episode_sums["eureka_total_rewards"],
                    "success_rate": success_rate
                }

            except Exception as e:
                print(traceback.format_exc())
                result = {"success": False, "exception": str(e)}

            self._results_queue.put((idx, result))

    # -------------------------
    def close(self):
        self.termination_event.set()
        for q in self._rewards_queues:
            q.put("Stop")
        for p in self._processes.values():
            p.join()