#TODO reward 정규화
#마지막 학습을 1개의 reward function 으로 하지말고 5개의 순위 매겨서 따로 학습해보기
# eureka_main.py
import os
import re
import multiprocessing as mp
import inspect

from llm_manager import LLMManager
from eureka_task_manager import EurekaTaskManager, EurekaThorWrapper
from policy_manager import PolicyManager
from datetime import datetime
# --- Config ---
GPT_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"

NUM_SUGGESTIONS = 5
TEMPERATURE = 1.0
MAX_ITERATIONS = 10
TRAINING_STEPS = 50000

# NUM_SUGGESTIONS = 2
# TEMPERATURE = 1.0
# MAX_ITERATIONS = 1
# TRAINING_STEPS = 10

TASK_DESCRIPTION = "Place an Mug on a CounterTop"

SYSTEM_PROMPT = f"""
You are a reward engineer trying to write reward functions to solve reinforcement learning tasks as effective as possible.
Your goal is to write a reward function for the environment that will help the agent learn the task described in text.
Your reward function should use useful variables from the environment as inputs.

The environment source code is:
{inspect.getsource(EurekaThorWrapper)}

** CRITICAL: Observation Coordinate System **
The `relative_target_pos` is provided in an EGO-CENTRIC coordinate system (the agent is at the origin, looking towards +Z).
- `relative_target_pos[0]`: Left (-x) / Right (+x) relative to the agent's view.
- `relative_target_pos[1]`: Below (-y) / Above (+y) relative to the agent's view.
- `relative_target_pos[2]`: Behind (-z) / In Front (+z) relative to the agent's view. (Increase this to move closer to the target).

You have access to the previous step's observation via `env.prev_obs` and the current step's observation via `env.last_obs`. 
DO NOT use `metadata` to calculate distances or visibility. ONLY use `env.prev_obs` and `env.last_obs` to ensure consistency.

When writing rewards, use these indices to guide the agent. For example, to make the agent move toward the target, you should encourage maximizing `relative_target_pos[2]` while minimizing its distance to zero.

As an example, the reward function signature can be:
def _get_rewards_eureka(env):
    rewards_dict = dict()
    ...
    return total_reward, rewards_dict

** IMPORTANT ** 
You have to write that reward components is the key and its reward value is the value of the rewards_dict. 

Below is a example of the reward function using prev_obs and last_obs:
def _get_rewards_eureka(env):
    rewards_dict = dict()
    
    # 1. Distance Reward (Potential-based)
    # Encourages getting closer. Scale should be moderate.
    prev_dist = env.prev_obs["distance"][0]
    curr_dist = env.last_obs["distance"][0]
    rewards_dict["dist_progress"] = (prev_dist - curr_dist) * 2.0

    # 2. Precision Alignment (Centering & Pitch)
    # Use exp(-error) to provide high pressure when very close to center.
    rel_x = env.last_obs["relative_target_pos"][0]
    rel_y = env.last_obs["relative_target_pos"][1]
    
    # Centering (Yaw) alignment
    rewards_dict["align_x"] = np.exp(-abs(rel_x) * 5.0) * 0.2
    
    # Vertical (Pitch) alignment: Agent's camera_horizon should match the target's Y direction
    # This is critical for objects on the floor or high shelves.
    curr_pitch = env.last_obs["camera_horizon"][0]
    target_pitch_needed = -rel_y # Heuristic: if rel_y is -0.5, look down.
    rewards_dict["align_y"] = np.exp(-abs(curr_pitch - target_pitch_needed) * 5.0) * 0.2

    # 3. Task Success (The most important part)
    # Check the success condition provided in the subtask label.
    # If success: rewards_dict["success_bonus"] = 50.0
    
    # 4. Step Penalty
    rewards_dict["step_penalty"] = -0.01

    total_reward = sum(rewards_dict.values())
    return total_reward, rewards_dict

"""

# -------------------------------
# ✅ Subtask + Success Code 생성
# -------------------------------
def generate_subtasks(llm, task_desc):
    prompt = f"""
Task: {task_desc}

For each subtask, generate:

1. Subtask: ...
   PreconditionCode: <python expression>
   SuccessCode: <python expression>

Rules:
- Both must be valid ONE LINE python expressions
- Use variable: metadata
- metadata = env.controller.last_event.metadata
- DO NOT use env.xxx

Example:

1. Subtask: Pick up apple
   PreconditionCode: True
   SuccessCode: any(obj["objectType"]=="Apple" for obj in metadata["inventoryObjects"])

2. Subtask: Place apple on plate
   PreconditionCode: any(obj["objectType"]=="Apple" for obj in metadata["inventoryObjects"])
   SuccessCode: any(obj["objectType"]=="Apple" and obj["parentReceptacles"] and any("Plate" in p for p in obj["parentReceptacles"]) for obj in metadata["objects"])

STRICT:
- No explanation
"""

    response = llm.prompt(prompt)

    if isinstance(response, dict) and "raw_outputs" in response:
        return response["raw_outputs"][0]
    return str(response)

import json
import random

def validate_output(data):
    label = data.get("label", "").strip()
    category = data.get("category", "").strip()

    # label 기본 처리
    if not label:
        label = "unknown_policy"

    # snake_case 보정
    label = label.lower().replace(" ", "_")

    # category 검증
    if category not in CATEGORIES:
        category = "Moveable"  # fallback (적당한 default)

    return {
        "label": label,
        "category": category
    }

def parse_policy_output(text):
    text = text.strip()

    # 1. JSON 직접 파싱 시도
    try:
        data = json.loads(text)
        return validate_output(data)
    except:
        pass

    # 2. JSON 일부만 있는 경우 추출
    try:
        json_str = re.search(r"\{.*\}", text, re.DOTALL).group()
        data = json.loads(json_str)
        return validate_output(data)
    except:
        pass

    # 3. fallback (line parsing)
    label = None
    category = None

    for line in text.split("\n"):
        if "label" in line.lower():
            label = line.split(":")[-1].strip().replace('"', '')
        if "category" in line.lower():
            category = line.split(":")[-1].strip().replace('"', '')

    return validate_output({
        "label": label,
        "category": category
    })

CATEGORIES = [
    "breakable", "cookable", "dirtyable", "fillable",
    "moveable", "openable", "pickupable", "receptacle",
    "sliceable", "toggleable", "usedUp", "navigate"
]

def generate_policy_label_and_category(llm, subtask):
    prompt = f"""  
Subtask: {subtask}

You are generating a GENERAL REUSABLE POLICY LABEL for a robotics agent.

IMPORTANT GOAL:
- The label must describe a reusable SKILL
- NOT a specific object instance

--------------------------------------------------
STEP 1: Identify action type
Examples:
- pick up
- navigate
- place
- open
- toggle
- slice

STEP 2: Abstract object into CATEGORY LEVEL
NEVER use specific objects like:
❌ mug, cup, apple, plate, knife, door handle

Instead use:
✔ object
✔ receptacle
✔ container
✔ tool
✔ target_object

--------------------------------------------------
RULES:
- label MUST be snake_case
- label MUST be (verb + abstract_noun)
- label MUST NOT contain any specific object name
- label MUST be reusable across environments
- category MUST be one from the list

--------------------------------------------------
CATEGORY LIST:
{CATEGORIES}

--------------------------------------------------
BAD EXAMPLES:
❌ pick_up_mug
❌ pick_up_red_cup
❌ navigate_to_kitchen_sink

GOOD EXAMPLES:
✔ pick_up_object
✔ navigate_to_object
✔ place_object_on_receptacle
✔ open_receptacle
✔ toggle_object

--------------------------------------------------
OUTPUT FORMAT (STRICT JSON):
{{
    "label": "...",
    "category": "..."
}}
"""

    res = llm.prompt(prompt)

    if isinstance(res, dict):
        text = res["raw_outputs"][0]
    else:
        text = str(res)

    return parse_policy_output(text)

# -------------------------------
# ✅ 파싱
# -------------------------------
def parse_subtasks(text):
    lines = text.split("\n")

    subtasks = []
    current = {}

    for line in lines:
        line = line.strip()

        if re.match(r"^\d+\.\s*Subtask:", line):
            if current:
                subtasks.append(current)
            current = {"subtask": "", "pre": "", "success": ""}

            current["subtask"] = re.sub(r"^\d+\.\s*Subtask:\s*", "", line)

        elif line.startswith("PreconditionCode:"):
            current["pre"] = line.replace("PreconditionCode:", "").strip()

        elif line.startswith("SuccessCode:"):
            current["success"] = line.replace("SuccessCode:", "").strip()

    if current:
        subtasks.append(current)

    return subtasks

def run_train_loop():
    llm = LLMManager(
        gpt_model=GPT_MODEL,
        num_suggestions=NUM_SUGGESTIONS,
        temperature=TEMPERATURE,
        system_prompt=SYSTEM_PROMPT
    )

    policy_manager = PolicyManager()

    os.makedirs("outputs/reward_shaping_logs", exist_ok=True)

    # -------------------------------
    # Subtask 생성
    # -------------------------------
    subtask_plan = generate_subtasks(llm, TASK_DESCRIPTION)
    subtasks = parse_subtasks(subtask_plan)

    print("🧩 Subtasks:", subtasks)

    # -------------------------------
    # Subtask loop
    # -------------------------------
    for s_idx, s in enumerate(subtasks):
        subtask = s["subtask"]
        success_code = s["success"]
        pre_code = s["pre"]

        print(f"\n🚀 [Subtask {s_idx+1}] {subtask}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = f"outputs/reward_shaping_logs/subtask_{s_idx+1}_{timestamp}.txt"

        # 🔥 LLM으로 policy label 생성
        subtask_info = generate_policy_label_and_category(llm, subtask) # {'label': ..., 'category': ...}
        print(f"🏷 Subtask Label(generalized): {subtask_info['label']}")
        print(f"🏷 Subtask Category: {subtask_info['category']}")

        # 파일 초기화
        with open(log_path, "w") as f:
            f.write(f"Subtask: {subtask}\n")
            f.write(f"Subtask label: {subtask_info['label']}\n")
            f.write(f"Category: {subtask_info['category']}\n")
            f.write(f"Success code: {success_code}\n")
            f.write("="*50 + "\n\n")

        task_manager = EurekaTaskManager(
            num_processes=NUM_SUGGESTIONS,
            max_training_iterations=TRAINING_STEPS,
            category=subtask_info['category']
        )
        target_object_type = task_manager._target_object_type

        print(f'Target object: {target_object_type}')

        best_score = -float("inf")
        best_success_rate = -1.0
        best_reward_code = None

        components_feedback = ""
        last_feedback = ""
        i = 0
        all_successful_results = []
        while i < MAX_ITERATIONS:
            print(f"\n🔄 Iter {i+1}")
            
            reward_components_feedback = components_feedback if components_feedback else "No feedback available yet (First Iteration)."
            reward_prompt = f"""
            We are currently focusing ONLY on this specific subtask: {subtask}
We trained a RL policy using the provided reward function code and tracked the values of the
individual components in the reward function as well as global policy metrics such as
success rates and episode lengths after all epochs and the maximum, mean,
minimum values encountered:
{reward_components_feedback}

[Qualitative Analysis & Correction]
{last_feedback}

Please carefully analyze the policy feedback and provide a new, improved reward function that can better solve the task. 
Some helpful tips for analyzing the policy feedback:
    (1) If the success rates are always near zero, then you must rewrite the entire reward function
    (2) If the values for a certain reward component are near identical throughout, then this means RL is not able to optimize this component as it is written. 
        You may consider
            (a) Changing its scale or the value of its temperature parameter
            (b) Re-writing the reward component
            (c) Discarding the reward component
    (3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range
    (4) PREVENT REWARD HACKING: Do NOT give continuous positive rewards every step for being in a good state (e.g., close to the object). The agent will just stay there to accumulate infinite rewards. Instead, use POTENTIAL-BASED shaping: Reward the agent ONLY when the distance DECREASES compared to the previous step (e.g., `reward = last_distance - current_distance`), or give a one-time reward when crossing a distance threshold.
    (5) STEP PENALTY: Add a small negative step penalty (e.g., -0.01 per step) to encourage the agent to finish the task quickly and prevent it from milking positive rewards without completing the task.
    (6) REWARD SCALING (CRITICAL): PPO learning is highly unstable if rewards are too large. Ensure the maximum possible reward per episode is tightly bounded. Scale down all components so that achieving the final interaction goal gives a moderate reward like 10.0 to 50.0 (NEVER 500.0 or 1000.0), and dense step rewards are proportionally small fractions (e.g., 0.1 to 1.0).
Please analyze each existing reward component in the suggested manner above first, and then write the reward function code.
"""
            # print(reward_prompt)
            # task_manager.thor_env.reset_reward_components_per_epoches()

            
            response = llm.prompt(reward_prompt)
            #print("response:", response)
            reward_strings = response["reward_strings"]

            reward_data = []
            for reward_code in reward_strings:
                reward_data.append({
                    "reward_code": reward_code,
                    "success_code": success_code,
                    "precondition_code": s["pre"]
                })

            results = task_manager.train(reward_data)

            
            with open(log_path, "a") as f:
                f.write(f"\n--- [Iter {i+1} All Candidate Results] ---\n")
                for r_idx, r in enumerate(results): # successful_results 대신 전체 results 순회
                    f.write(f"--- Candidate {r_idx+1} ---\n")
                    if r.get("success", False):
                        c_code = r.get("reward_code", "")
                        c_train_sr = r.get("train_success_rate", 0)
                        c_eval_sr = r.get("success_rate", 0)
                        c_mean = r.get("reward_mean", 0)
                        f.write(f"{c_code}\n")
                        f.write(f"> [SUCCESS] Train SR: {c_train_sr:.4f} | Eval SR: {c_eval_sr:.4f} | Mean Reward: {c_mean:.4f}\n\n")
                    else:
                        # 🔥 실패한 경우 에러 메시지를 로그에 남깁니다.
                        f.write(f"> [FAILED] Exception: {r.get('exception', 'Unknown error')}\n\n")

            # 이후 로직은 동일하게 successful_results로 진행
            successful_results = [r for r in results if r is not None and r.get("success", False)]
            all_successful_results.extend(successful_results) # 전체 성공 결과 누적

        
            if not successful_results:
                last_feedback = "All reward suggestions failed to execute. Please check the syntax and environment API."
                continue
            best_iter_result = max(successful_results, key=lambda x: (x["success_rate"], x["train_success_rate"], x["reward_mean"]))
            score = best_iter_result["reward_mean"]
            success_rate = best_iter_result["success_rate"]
            train_success_rate = best_iter_result["train_success_rate"]
            reward_code = best_iter_result["reward_code"]

            print(f"Training Success Rate: {train_success_rate:.4f}")

            raw_components = best_iter_result.get("reward_components", {})
            
            feedback_lines = []
            feedback_lines.append("[Global Policy Metrics]")
            feedback_lines.append(f"- Success Rate: {success_rate:.2f}")
            feedback_lines.append(f"- Mean Total Reward: {score:.4f}")
            feedback_lines.append("\n[Reward Components (Min, Mean, Max)]")

            if raw_components:
                for comp_name, values in raw_components.items():
                    if values:
                        # 리스트 안의 값이 어떤 타입이든 순수 파이썬 float으로 통일
                        clean_values = [float(v) for v in values]
                        
                        comp_min = min(clean_values)
                        comp_max = max(clean_values)
                        comp_mean = sum(clean_values) / len(clean_values)
                        
                        # 원본 리스트의 값들도 보기 편하게 포맷팅 (소수점 4자리)
                        #raw_list_str = "[" + ", ".join([f"{v:.4f}" for v in clean_values]) + "]"
                        
                        feedback_lines.append(f"- {comp_name}:")
                        feedback_lines.append(f"  * Stats: Min={comp_min:.4f}, Mean={comp_mean:.4f}, Max={comp_max:.4f}")
                        #feedback_lines.append(f"  * Raw  : {raw_list_str}")
                    else:
                        feedback_lines.append(f"- {comp_name}: No data")
            else:
                feedback_lines.append("- No component data available.")
                
            components_feedback = "\n".join(feedback_lines)


            with open(log_path, "a") as f:
                f.write(f"Iteration {i+1}, TrainSuccessRate: {train_success_rate}, SuccessRate: {success_rate}\n")

            # 🔥 success_rate 기반 feedback
            # 수정된 코드
            if success_rate < 0.1:
                if score > 100: # 점수는 높은데 성공을 못함 = 보상 해킹 중
                    last_feedback = (
                        "WARNING: The agent is getting very high rewards but failing the task (Reward Hacking). "
                        "It is exploiting dense rewards (like proximity or visibility) by just standing near the object without actually interacting. "
                        "DO NOT give positive rewards every step for just being near the object. "
                        "Use potential-based rewards (e.g., based on the change in distance: last_dist - current_dist) or strictly cap dense rewards per episode."
                    )
                else: # 점수도 낮고 성공도 못함 = 진짜 학습 안됨
                    last_feedback = (
                        "The agent almost never succeeds and has low rewards. "
                        "Provide careful potential-based intermediate rewards to guide it closer."
                    )

            elif success_rate < 0.5:
                last_feedback = (
                    "The agent sometimes succeeds but is unstable. "
                    "Improve reward shaping to guide behavior more consistently. "
                    "Use distance-based shaping and intermediate milestones."
                )

            elif success_rate < 0.9:
                last_feedback = (
                    "The agent often succeeds but not reliably. "
                    "Refine the reward to reduce randomness and improve consistency. "
                    "Penalize unnecessary actions and encourage efficient behavior."
                )

            else:
                last_feedback = (
                    "The agent succeeds reliably. "
                    "Now refine the reward to improve efficiency and avoid reward hacking."
                )

            print(f"Score: {score}, SuccessRate: {success_rate}")
            with open(log_path, "a") as f:
                f.write(f"Score: {score}, SuccessRate: {success_rate}\n")

                # f.write(f"Last Feedback: {last_feedback}\n")
                # f.write(f"Components Feedback: {components_feedback}\n")

            i += 1

            # 🔥 여기 핵심: model 가져오기
            # best_state_dict = result["model_state_dict"]
            # score아니고 success_rate로 해야함.
            if best_success_rate is None or success_rate > best_success_rate:
                best_success_rate = success_rate
                best_reward_code = reward_code

            last_feedback += f"\nLast Score: {score}, Last Success Rate: {success_rate}\nLast Reward Function Code: \n{reward_code}"
            # print(last_feedback)

         # 🔥 best reward 기록
        with open(log_path, "a") as f:
            f.write(f"🔥 BEST (score={score})\n")
            f.write(reward_code + "\n\n")

        print(f"✅ Best short-term success rate: {best_success_rate:.4f}")
        print("\n🔥 Extracting Top-K Reward Functions for Final Training...")

        # 1. 중복 코드 제거 (같은 코드면 가장 높은 점수 기록만 유지)
        unique_results = {}
        for res in all_successful_results:
            code = res["reward_code"]
            if code not in unique_results:
                unique_results[code] = res
            else:
                existing = unique_results[code]
                # Eval SR -> Train SR -> Mean Reward 순으로 비교하여 갱신
                if (res["success_rate"], res["train_success_rate"], res["reward_mean"]) > \
                   (existing["success_rate"], existing["train_success_rate"], existing["reward_mean"]):
                    unique_results[code] = res

        # 2. 정렬 (1순위: Eval SR, 2순위: Train SR, 3순위: Reward)
        sorted_unique_results = sorted(
            unique_results.values(),
            key=lambda x: (x["success_rate"], x["train_success_rate"], x["reward_mean"]),
            reverse=True
        )

        # 3. 상위 5개(NUM_SUGGESTIONS) 추출
        top_k_results = sorted_unique_results[:NUM_SUGGESTIONS]

        final_reward_data = []
        with open(log_path, "a") as f:
            f.write("\n=== 🏆 Top-K Selected Reward Functions for Final Training ===\n")
            for rank, res in enumerate(top_k_results):
                print(f"🏅 Top {rank+1} Selected (Eval SR: {res['success_rate']:.4f}, Train SR: {res['train_success_rate']:.4f})")
                f.write(f"🏅 Top {rank+1} Selected (Eval SR: {res['success_rate']:.4f})\n")
                
                final_reward_data.append({
                    "reward_code": res["reward_code"],
                    "success_code": success_code,
                    "precondition_code": s["pre"],
                    "training_steps": TRAINING_STEPS * 10,
                    "eval_episodes": 100
                })

        # 혹시라도 고유한 코드가 5개가 안 될 경우를 대비해 1등 코드로 빈자리 채우기
        while len(final_reward_data) < NUM_SUGGESTIONS:
            final_reward_data.append(final_reward_data[0].copy())

        print("\n🔥 Final training starting with diverse Top candidates...")

        # 🔥 최종 학습 실행
        final_results = task_manager.train(final_reward_data)
        print("final model 학습 완료")

        valid_final_results = [r for r in final_results if r is not None and r.get("success", False)]
        if valid_final_results:
            # 5개의 워커 중 가장 결과가 좋은 놈의 모델을 최종 선택
            best_final_result = max(valid_final_results, key=lambda x: (x["success_rate"], x["train_success_rate"], x["reward_mean"]))
            final_model = best_final_result["model_state_dict"]

            with open(log_path, "a") as f:
                f.write(f"\n--- Final Evaluation of BEST Reward Function across all workers ---\n")
                for r_idx, r in enumerate(final_results): # successful_results 대신 전체 results 순회
                    f.write(f"--- Candidate {r_idx+1} ---\n")
                    if r and r.get("success", False):
                        c_code = r.get("reward_code", "")
                        c_train_sr = r.get("train_success_rate", 0)
                        c_eval_sr = r.get("success_rate", 0)
                        c_mean = r.get("reward_mean", 0)
                        f.write(f"{c_code}\n")
                        f.write(f"> [SUCCESS] Train SR: {c_train_sr:.4f} | Eval SR: {c_eval_sr:.4f} | Mean Reward: {c_mean:.4f}\n\n")
                    else:
                        # 🔥 실패한 경우 에러 메시지를 로그에 남깁니다.
                        error_msg = r.get('exception', 'Unknown error') if r else 'Worker Process No Response (None)'
                        f.write(f"> [FAILED] Exception: {error_msg}\n\n")

            policy_manager.save_policy(final_model, subtask_info['label'])
            print(f"✅ Final model saved (score={best_final_result['reward_mean']}, success={best_final_result['success_rate']})")
        else:
            print("❌ Final training failed for all workers.")
        task_manager.close()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    run_train_loop()