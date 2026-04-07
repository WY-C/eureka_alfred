# eureka_task_manager.py
import os
import traceback
import multiprocessing
import random

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

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
        self.target_object = "Mug"
        self._success_code = None

        if hasattr(env.action_space, "spaces"):
            self.action_space = env.action_space["action_index"]
        else:
            self.action_space = env.action_space

        self._get_rewards_eureka = None

    def action(self, act):
        return {"action_index": int(act)}

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(self.action(action))

        reward = 0.0
        metadata = self.env.unwrapped.controller.last_event.metadata

        # -------------------------
        # 1. reward shaping
        # -------------------------
        if self._get_rewards_eureka:
            try:
                result = self._get_rewards_eureka(self.env.unwrapped)
                reward = float(result[0] if isinstance(result, tuple) else result)
            except Exception as e:
                raise RuntimeError(f"Reward function failed: {e}")

        # -------------------------
        # 2. SUCCESS (subtask1: pick)
        # -------------------------
        # inventory = metadata.get("inventoryObjects", [])
        # target = self.target_object

        info["is_success"] = False

        if self._success_code:
            try:
                success = eval(self._success_code, {"metadata": metadata})
                if success:
                    reward += 10.0
                    terminated = True
                    info["is_success"] = True
            except Exception as e:
                print(f"⚠️ success eval error: {e}")

        return obs, reward, terminated, truncated, info

    
    def reset(self, **kwargs):
        MAX_TRY = 10
        last_obs, last_info = None, None

        for _ in range(MAX_TRY):
            obs, info = self.env.reset(**kwargs)

            metadata = self.env.unwrapped.controller.last_event.metadata
            objects = metadata.get("objects", [])

            valid_targets = [
                obj for obj in objects
                if obj.get("objectType") == self.target_object
                and obj.get("pickupable", False)
                and obj.get("visible", False)
            ]

            if len(valid_targets) > 0:
                return obs, info

            last_obs, last_info = obs, info

        print(f"⚠️ Failed to find visible & pickable {self.target_object}")
        return last_obs, last_info


# -------------------------
# utils
# -------------------------
def clean_code(code: str):
    code = code.strip()
    if code.startswith("```"):
        code = code.split("```")[1]
    if code.endswith("```"):
        code = code[:-3]
    return code


import re

def fix_function_name(code: str):
    match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code)
    if match:
        name = match.group(1)
        if name != "_get_rewards_eureka":
            code = re.sub(
                r"def\s+" + name + r"\s*\(",
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
        task="rl_thor/ITHOREnv-v0.1",
        num_processes=1,
        device="cuda",
        max_training_iterations=20000,
        config_path=None
    ):
        self._task = task
        self._num_processes = num_processes
        self._device = device
        self._max_training_iterations = max_training_iterations
        self._config_path = config_path or os.path.expanduser(
            "~/Documents/rl_thor/config/environment_config.yaml"
        )

        self._rewards_queues = [multiprocessing.Queue() for _ in range(num_processes)]
        self._results_queue = multiprocessing.Queue()
        self.termination_event = multiprocessing.Event()

        self._processes = {}
        for idx in range(num_processes):
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
            config_override={"max_episode_steps": 300}
        )

        env = EurekaThorWrapper(raw_env)

        while not self.termination_event.is_set():
            data = queue.get()
            #print("DATA:", data)    
            if data == "Stop":
                break

            reward_code = data["reward_code"]
            env._success_code = data["success_code"]

            try:
                # -------------------------
                # reward compile
                # -------------------------
                ns = {}
                reward_code = clean_code(reward_code)
                reward_code = fix_function_name(reward_code)
                # print("====== REWARD CODE ======")
                # print(reward_code)
                # print("=========================")
                exec(reward_code, ns)
                # print("ns: ", ns)

                if "_get_rewards_eureka" not in ns:
                    raise ValueError("Reward function missing")

                env._get_rewards_eureka = ns["_get_rewards_eureka"]

                # -------------------------
                # train
                # -------------------------
                model = PPO(
                    "MultiInputPolicy",
                    env,
                    verbose=0,
                    device=self._device,
                    n_steps=512,
                    batch_size=64,
                    ent_coef=0.01
                )

                model.learn(total_timesteps=self._max_training_iterations)

                # -------------------------
                # evaluation
                # -------------------------
                success_count = 0
                episodes = 10

                for _ in range(episodes):
                    obs, _ = env.reset()
                    done = False

                    while not done:
                        action, _ = model.predict(obs)
                        obs, _, terminated, truncated, info = env.step(action)
                        done = terminated or truncated

                        if info.get("is_success", False):
                            success_count += 1
                            break

                success_rate = success_count / episodes

                # -------------------------
                # reward mean
                # -------------------------
                rewards = []

                for _ in range(5):
                    obs, _ = env.reset()
                    done = False
                    total_r = 0

                    while not done:
                        action, _ = model.predict(obs)
                        obs, r, terminated, truncated, _ = env.step(action)
                        total_r += r
                        done = terminated or truncated

                    rewards.append(total_r)

                reward_mean = sum(rewards) / len(rewards)

                result = {
                    "success": True,
                    "reward_mean": reward_mean,
                    "success_rate": success_rate,
                    "model_state_dict": model.policy.state_dict()
                }

            except Exception as e:
                print(traceback.format_exc())
                result = {"success": False, "exception": str(e)}

            self._results_queue.put((idx, result))

    # -------------------------
    def finalize_training(self, reward_code, success_code=None):
        raw_env = gym.make(
            self._task,
            config_path=self._config_path,
            config_override={"max_episode_steps": 300}
        )

        env = EurekaThorWrapper(raw_env)

        ns = {}
        reward_code = clean_code(reward_code)
        reward_code = fix_function_name(reward_code)
        exec(reward_code, ns)

        env._get_rewards_eureka = ns["_get_rewards_eureka"]

        if success_code:
            env._success_code = success_code

        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            device=self._device,
            n_steps=512,
            batch_size=64,
            ent_coef=0.01
        )

        # -------------------------
        # ✅ checkpoint callback
        # -------------------------
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,                      # 🔥 10000 step마다 저장
            save_path="./checkpoints/",           # 저장 경로
            name_prefix="ppo_eureka",             # 파일 이름 prefix
            save_replay_buffer=False,
            save_vecnormalize=False,
        )

        # -------------------------
        # train
        # -------------------------
        model.learn(
            total_timesteps=100000,
            callback=checkpoint_callback
        )

        return model

    # -------------------------
    def close(self):
        self.termination_event.set()
        for q in self._rewards_queues:
            q.put("Stop")
        for p in self._processes.values():
            p.join()