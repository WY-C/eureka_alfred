#좀 더 수정해야함. 기존 코드에서 더 찾아볼것 OUTPUT에 리워드도 적어두고....

# eureka_main.py
import os
import sys
from datetime import datetime
from llm_manager import LLMManager
from eureka_task_manager import EurekaTaskManager

# --- 1. Configurations ---
GPT_MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"
NUM_SUGGESTIONS = 1  
TEMPERATURE = 0.7
MAX_ITERATIONS = 5   
TRAINING_STEPS = 10000 

SYSTEM_PROMPT = """
You are a world-class reward engineer for reinforcement learning. 
Task: 'Place an Apple on a CounterTop' in the AI2-THOR environment.

[Instructions]
1. Provide your step-by-step 'Reasoning' for the reward design strictly in ENGLISH.
2. Provide the reward function in a python code block starting with ```python.
3. Function signature: def _get_rewards_eureka(env):
4. Return: (total_reward, rewards_dict)
"""

def run_eira_loop():
    # 2. Initialize Managers
    llm = LLMManager(
        gpt_model=GPT_MODEL,
        num_suggestions=NUM_SUGGESTIONS,
        temperature=TEMPERATURE,
        system_prompt=SYSTEM_PROMPT
    )
    
    task_manager = EurekaTaskManager(
        num_processes=NUM_SUGGESTIONS,
        max_training_iterations=TRAINING_STEPS
    )

    last_feedback = "This is the first iteration. Focus on moving the agent toward the Apple."

    # 3. Eureka Main Loop
    for i in range(MAX_ITERATIONS):
        print(f"\n🔄 [Iteration {i+1}/{MAX_ITERATIONS}] Requesting reward design from {GPT_MODEL}...")
        
        # LLM 호출
        response = llm.prompt(f"Design the reward function. Feedback: {last_feedback}")
        reward_strings = response["reward_strings"]
        raw_outputs = response["raw_outputs"] # 이 녀석은 리스트 ['내용'] 입니다.

        # 📄 [수정 포인트 1] 파일 저장 시으로 문자열 추출
        os.makedirs("outputs/eira_logs", exist_ok=True)
        with open(f"outputs/eira_logs/reasoning_iter_{i+1}.txt", "w", encoding="utf-8") as f:
            if isinstance(raw_outputs, list) and len(raw_outputs) > 0:
                # 보따리(list)에서 첫 번째 내용물(str)을 꺼내서 씁니다.
                f.write(raw_outputs[0]) 
            else:
                f.write(str(raw_outputs))
        print(f"📄 Reasoning log saved: outputs/eira_logs/reasoning_iter_{i+1}.txt")

        # 4. 학습 시작
        print(f"🚀 Training started (Target: {TRAINING_STEPS} steps)...")
        
        try:
            # task_manager.train은 결과 딕셔너리가 든 '리스트'를 반환합니다.
            results = task_manager.train(reward_strings)
            
            # 📊 [수정 포인트 2] 리스트에서 0번 인덱스 추출
            if isinstance(results, list) and len(results) > 0:
                best_result = results 
            else:
                best_result = results
            
            # 점수 추출 및 피드백 생성
            if isinstance(best_result, dict):
                score = best_result.get("reward_mean", 0)
                print(f"📊 Iteration {i+1} Finished. Mean Reward: {score:.4f}")
                last_feedback = f"The previous reward function got a score of {score:.4f}. Improve it to be more dense."
            else:
                print(f"⚠️ Warning: Invalid result format: {type(best_result)}")
                score = 0
                last_feedback = "The training completed but results were not formatted correctly."
                
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️ Training Error: {error_msg}")
            last_feedback = f"The previous reward function failed with error: {error_msg}. Please fix the code."

    task_manager.close()
    print("\n🏁 All Eureka iterations finished. Check your EIRA data!")

if __name__ == "__main__":
    run_eira_loop()