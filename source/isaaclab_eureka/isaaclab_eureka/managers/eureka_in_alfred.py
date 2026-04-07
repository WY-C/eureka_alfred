# temp.py
import os
import sys
import torch

# 현재 폴더 및 상위 폴더 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from eureka_task_manager import EurekaTaskManager
    print("✅ [성공] eureka_task_manager.py를 정상적으로 로드했습니다.")
except ImportError as e:
    print(f"❌ [오류] 임포트 실패: {e}")
    sys.exit(1)

def main():
    print("\n" + "="*50)
    print("🚀 Eureka + AI2-THOR 통합 연동 테스트 (RTX 5070 Ti)")
    print("="*50)

    manager = EurekaTaskManager(
        env="rl_thor/ITHOREnv-v0.1",
        num_processes=1,
        device="cuda",
        max_training_iterations=2000000
    )

    # _get_rewards_eureka(env) 형태로 작성
    mock_reward_code = """
def _get_rewards_eureka(env):
    rewards_dict = {}
    metadata = env.controller.last_event.metadata
    agent_pos = metadata['agent']['position']
    
    objs = metadata['objects']
    apple = next((o for o in objs if o['objectType'] == 'Apple'), None)
    
    if apple:
        # 3D 거리 계산
        dist = math.sqrt(
            (apple['position']['x'] - agent_pos['x'])**2 +
            (apple['position']['y'] - agent_pos['y'])**2 +
            (apple['position']['z'] - agent_pos['z'])**2
        )
        rewards_dict['dist_bonus'] = math.exp(-dist)
    else:
        rewards_dict['dist_bonus'] = 0.0
        
    return rewards_dict['dist_bonus'], rewards_dict
"""

    print("\n[진행] LLM 보상 주입 및 학습 시작...")
    results = manager.train([mock_reward_code])

    print("\n" + "="*50)
    for i, res in enumerate(results):
        if res["success"]:
            print(f"✅ 결과: 학습 성공 (로그: {res['log_dir']})")
            print(f"   - 최종 보상 합계: {res['reward_mean']:.4f}")
        else:
            print(f"❌ 결과: 학습 실패 (사유: {res['exception']})")
    print("="*50)

    manager.close()

if __name__ == "__main__":
    main()
# temp.py
import os
import sys
import torch

# 현재 폴더 및 상위 폴더 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from eureka_task_manager import EurekaTaskManager
    print("✅ [성공] eureka_task_manager.py를 정상적으로 로드했습니다.")
except ImportError as e:
    print(f"❌ [오류] 임포트 실패: {e}")
    sys.exit(1)

def main():
    print("\n" + "="*50)
    print("🚀 Eureka + AI2-THOR 통합 연동 테스트 (RTX 5070 Ti)")
    print("="*50)

    manager = EurekaTaskManager(
        task="rl_thor/ITHOREnv-v0.1",
        num_processes=1,
        device="cuda",
        max_training_iterations=2000000
    )

    # _get_rewards_eureka(env) 형태로 작성
    mock_reward_code = """
def _get_rewards_eureka(env):
    rewards_dict = {}
    metadata = env.controller.last_event.metadata
    agent_pos = metadata['agent']['position']
    
    objs = metadata['objects']
    apple = next((o for o in objs if o['objectType'] == 'Apple'), None)
    
    if apple:
        # 3D 거리 계산
        dist = math.sqrt(
            (apple['position']['x'] - agent_pos['x'])**2 +
            (apple['position']['y'] - agent_pos['y'])**2 +
            (apple['position']['z'] - agent_pos['z'])**2
        )
        rewards_dict['dist_bonus'] = math.exp(-dist)
    else:
        rewards_dict['dist_bonus'] = 0.0
        
    return rewards_dict['dist_bonus'], rewards_dict
"""

    print("\n[진행] LLM 보상 주입 및 학습 시작...")
    results = manager.train([mock_reward_code])

    print("\n" + "="*50)
    for i, res in enumerate(results):
        if res["success"]:
            print(f"✅ 결과: 학습 성공 (로그: {res['log_dir']})")
            print(f"   - 최종 보상 합계: {res['reward_mean']:.4f}")
        else:
            print(f"❌ 결과: 학습 실패 (사유: {res['exception']})")
    print("="*50)

    manager.close()

if __name__ == "__main__":
    main()