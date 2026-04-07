import gymnasium as gym
import torch
from stable_baselines3 import PPO
import rl_thor
def visualize_policy(policy_path='./outputs/policies/place_mug_on_countertop.pt', task="rl_thor/ITHOREnv-v0.1", config_path=None):
    env = gym.make(
    "rl_thor/ITHOREnv-v0.1",
    config_path="/home/cau/Documents/rl_thor/config/environment_config.yaml",  # 🔥 절대경로
    config_override={"max_episode_steps": 30000}
    )

    # wrapper 적용
    from eureka_task_manager import EurekaThorWrapper
    env = EurekaThorWrapper(env)

    # 모델 생성
    model = PPO("MultiInputPolicy", env)

    # 🔥 저장된 policy 로드
    state_dict = torch.load(policy_path)
    model.policy.load_state_dict(state_dict)

    # rollout
    obs, _ = env.reset()

    done = False
    step = 0

    while not done:
        action, _ = model.predict(obs, deterministic=False)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 🔥 핵심: 화면 보기
        try:
            env.render()  # GUI 뜨는 경우
        except:
            pass

        print(f"Step {step} | Reward: {reward}")
        step += 1

visualize_policy()