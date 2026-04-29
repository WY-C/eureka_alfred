# eureka_main.py
import os
import re
import multiprocessing as mp
import inspect

from llm_manager import LLMManager
from eureka_task_manager import EurekaTaskManager, EurekaThorWrapper
from policy_manager import PolicyManager

# --- Config ---
GPT_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"
NUM_SUGGESTIONS = 1
TEMPERATURE = 1.2
MAX_ITERATIONS = 10
# TRAINING_STEPS = 100
TRAINING_STEPS = 70000

TASK_DESCRIPTION = "Place an Mug on a CounterTop"

SYSTEM_PROMPT = f"""
You are a reward engineer trying to write reward functions to solve reinforcement learning tasks as effective as possible.
Your goal is to write a reward function for the environment that will help the agent learn the task described in text.
Your reward function should use useful variables from the environment as inputs.

The environment source code is:
{inspect.getsource(EurekaThorWrapper)}

As an example, the reward function signature can be:
def _get_rewards_eureka(object_pos: torch.Tensor, goal_pos: torch.Tensor):
    rewards_dict = dict()
    ...
    return total_reward, rewards_dict

** IMPORTANT ** 
You have to write that reward components is the key and its reward value is the value of the rewards_dict. 

Below is one example of the reward function:
def _get_rewards_eureka(object_rot: torch.Tensor, goal_rot: torch.Tensor):
    rot_diff = torch.abs(torch.sum(object_rot * goal_rot, dim=1) - 1) / 2
    rotation_reward = torch.exp(-20 * rot_diff)

    # Scaling factor for the rotation reward
    rotation_temp = 20.0
    total_reward = rotation_reward

    rewards_dict = {{
        "rotation_reward": rotation_reward
    }}
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

        print(f"\n🚀 [Subtask {s_idx+1}] {subtask}")

        log_path = f"outputs/reward_shaping_logs/subtask_{s_idx+1}.txt"

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
Please analyze each existing reward component in the suggested manner above first, and then write the reward function code.
"""
            # print(reward_prompt)
            # task_manager.thor_env.reset_reward_components_per_epoches()

            
            response = llm.prompt(reward_prompt)
            reward_strings = response["reward_strings"]

            reward_code = reward_strings[0]

            # 🔥 로그 저장
            with open(log_path, "a") as f:
                f.write(f"[Iter {i+1}]\n")
                f.write(reward_code + "\n\n")

            reward_data = [{
                "reward_code": reward_strings[0],
                "success_code": success_code,
                "precondition_code": s["pre"]
            }]

            results = task_manager.train(reward_data)
            result = results[0]


            if not result["success"]:
                last_feedback = f"Error: {result['exception']}"
                continue
            score = result["reward_mean"]
            success_rate = result["success_rate"]
            train_success_rate = result["train_success_rate"]
            print(f"Training Success Rate: {train_success_rate:.4f}")

            raw_components = result.get("reward_components", {})
            
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
                        raw_list_str = "[" + ", ".join([f"{v:.4f}" for v in clean_values]) + "]"
                        
                        feedback_lines.append(f"- {comp_name}:")
                        feedback_lines.append(f"  * Stats: Min={comp_min:.4f}, Mean={comp_mean:.4f}, Max={comp_max:.4f}")
                        feedback_lines.append(f"  * Raw  : {raw_list_str}")
                    else:
                        feedback_lines.append(f"- {comp_name}: No data")
            else:
                feedback_lines.append("- No component data available.")
                
            components_feedback = "\n".join(feedback_lines)


            with open(log_path, "a") as f:
                f.write(f"Score: {score}, SuccessRate: {success_rate}\n\n")  

            # 🔥 success_rate 기반 feedback
            if success_rate < 0.1:
                last_feedback = (
                    "The agent almost never succeeds. "
                    "Add strong and dense intermediate rewards. "
                    "Guide the agent step-by-step (visibility, distance, interaction)."
                )

            elif success_rate < 0.5:
                last_feedback += (
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
            
            i += 1

            # 🔥 여기 핵심: model 가져오기
            # best_state_dict = result["model_state_dict"]
            # score아니고 success_rate로 해야함.
            if best_success_rate is None or success_rate > best_success_rate:
                best_success_rate = success_rate
                best_reward_code = reward_strings[0]

            last_feedback += f"\nLast Score: {score}, Last Success Rate: {success_rate}\nLast Reward Function Code: \n{reward_code}"
            # print(last_feedback)

         # 🔥 best reward 기록
        with open(log_path, "a") as f:
            f.write(f"🔥 BEST (score={score})\n")
            f.write(reward_code + "\n\n")

        print(f"✅ Best success rate: {best_success_rate:.4f}")
        print("\n🔥 Final training with BEST reward function...")

        final_reward_data = [{
            "reward_code": best_reward_code,
            "success_code": success_code,
            "precondition_code": s["pre"]
        }]

        # 🔥 더 길게 학습 (중요)
        task_manager._max_training_iterations = TRAINING_STEPS * 4

        final_results = task_manager.train(final_reward_data)
        final_result = final_results[0]

        if final_result["success"]:
            final_model = final_result["model_state_dict"]
            policy_manager.save_policy(final_model, subtask_info['label'])

            print(f"✅ Final model saved (score={final_result['reward_mean']}, success={final_result['success_rate']})")
        else:
            print("❌ Final training failed:", final_result["exception"])

    task_manager.close()


if __name__ == "__main__":
    # mp.set_start_method("spawn", force=True)
    run_train_loop()