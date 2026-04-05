# eureka_task_manager.py
import os
import sys
import traceback
import types
import multiprocessing
import math
from datetime import datetime
from typing import Literal

import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

# rl_thor 임포트 확인
try:
    import rl_thor
except ImportError:
    print("❌ [오류] rl_thor를 찾을 수 없습니다.")

# --- Eureka 전용 통합 Wrapper ---
class EurekaThorWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # SB3 호환용 행동 공간 설정 (Discrete 14)
        if hasattr(env.action_space, "spaces"):
            self.action_space = env.action_space["action_index"]
        else:
            self.action_space = env.action_space
            
        self._eureka_episode_sums = {"eureka_total_rewards": 0.0, "oracle_total_rewards": 0.0}
        self._get_rewards_eureka = None

    def action(self, act):
        return {"action_index": int(act)}

    def step(self, action):
        # 1. 기존 환경의 step 실행
        obs, oracle_reward, terminated, truncated, info = self.env.step(self.action(action))
        
        eureka_reward = oracle_reward
        rewards_dict = {}

        # 2. LLM 보상 함수 실행
        if self._get_rewards_eureka is not None:
            try:
                # env.unwrapped를 통해 메타데이터에 접근
                eureka_reward, rewards_dict = self._get_rewards_eureka(self.env.unwrapped)
            except Exception as e:
                if not hasattr(self, "_error_logged"):
                    print(f"⚠️ [보상 함수 실행 에러] {e}")
                    self._error_logged = True
                eureka_reward = oracle_reward

        # 3. 통계 기록
        self._eureka_episode_sums["eureka_total_rewards"] += float(eureka_reward)
        self._eureka_episode_sums["oracle_total_rewards"] += float(oracle_reward)
        if isinstance(rewards_dict, dict):
            for key, val in rewards_dict.items():
                if key not in self._eureka_episode_sums:
                    self._eureka_episode_sums[key] = 0.0
                self._eureka_episode_sums[key] += float(val)

        return obs, float(eureka_reward), terminated, truncated, info

    def reset(self, **kwargs):
        self._eureka_episode_sums = {k: 0.0 for k in self._eureka_episode_sums}
        return self.env.reset(**kwargs)

class EurekaTaskManager:
    def __init__(
        self,
        task: str = "rl_thor/ITHOREnv-v0.1",
        rl_library: Literal["sb3"] = "sb3",
        num_processes: int = 1,
        device: str = "cuda",
        max_training_iterations: int = 2048,
        config_path: str = None
    ):
        self._task = task
        self._rl_library = rl_library
        self._num_processes = num_processes
        self._device = device
        self._max_training_iterations = max_training_iterations
        self._config_path = config_path or os.path.expanduser("~/Desktop/rl_thor/config/environment_config.yaml")

        self._processes = dict()
        self._rewards_queues = [multiprocessing.Queue() for _ in range(self._num_processes)]
        self._observations_queue = multiprocessing.Queue()
        self._results_queue = multiprocessing.Queue()
        self.termination_event = multiprocessing.Event()

        for idx in range(self._num_processes):
            p = multiprocessing.Process(target=self._worker, args=(idx, self._rewards_queues[idx]))
            self._processes[idx] = p
            p.start()

        self._obs_as_string = self._observations_queue.get()

    @property
    def get_observations_method_as_string(self) -> str:
        return self._obs_as_string

    def train(self, reward_func_list: list[str]) -> list[dict]:
        for idx, reward_str in enumerate(reward_func_list):
            self._rewards_queues[idx].put(reward_str)

        results = [None] * self._num_processes
        for _ in range(self._num_processes):
            idx, res = self._results_queue.get()
            results[idx] = res
        return results

    def _worker(self, idx: int, rewards_queue: multiprocessing.Queue):
        raw_env = gym.make(self._task, config_path=self._config_path, 
                           config_override={"max_episode_steps": 500, "action_modifiers": {"discrete_actions": True}})
        env = EurekaThorWrapper(raw_env)
        
        if idx == 0:
            env_info = "AI2-THOR Metadata: env.controller.last_event.metadata (agent, objects)"
            self._observations_queue.put(env_info)

        while not self.termination_event.is_set():
            reward_func_code = rewards_queue.get()
            if reward_func_code == "Stop": break

            try:
                # 보상 함수 컴파일 (SyntaxError 방지를 위해 멀티라인 문자열 사용)
                reward_ns = {}
                full_code = f"import torch\nimport math\n{reward_func_code}"
                exec(full_code, reward_ns)
                env._get_rewards_eureka = reward_ns["_get_rewards_eureka"]
                
                log_dir = self._run_training(env)
                result = {
                    "success": True, 
                    "log_dir": log_dir, 
                    "reward_mean": env._eureka_episode_sums["eureka_total_rewards"]
                }
            except Exception as e:
                print(traceback.format_exc())
                result = {"success": False, "exception": str(e)}

            self._results_queue.put((idx, result))

    def _run_training(self, env):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = f"./thor_tensorboard/run_{timestamp}"
        new_logger = configure(log_path, ["stdout", "tensorboard"])
        
        model = PPO("MultiInputPolicy", env, verbose=1, device=self._device)
        model.set_logger(new_logger)
        model.learn(total_timesteps=self._max_training_iterations)
        return log_path

    def close(self):
        self.termination_event.set()
        for q in self._rewards_queues: q.put("Stop")
        for p in self._processes.values(): p.join()