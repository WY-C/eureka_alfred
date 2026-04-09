# eureka_main.py
import os
import re
import multiprocessing as mp

from llm_manager import LLMManager
from eureka_task_manager import EurekaTaskManager
from policy_manager import PolicyManager

# --- Config ---
GPT_MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"
NUM_SUGGESTIONS = 1  
TEMPERATURE = 1.2
MAX_ITERATIONS = 5   
TRAINING_STEPS = 10000 

TASK_DESCRIPTION = "Place an Mug on a CounterTop"

SYSTEM_PROMPT = """
You are a world-class reward engineer.

CRITICAL OUTPUT FORMAT:

You MUST return EXACTLY:

def _get_rewards_eureka(env):
    total_reward = 0
    rewards_dict = {}

    ...

    return total_reward, rewards_dict

RULES:
- NEVER return a single value
- ALWAYS return TWO values
- Second value MUST be dict
- If you violate this → code will crash

IMPORTANT:

- The target object type is available as:
    TARGET_TYPE

- DO NOT use:
    targetObjectType
    obj["targetObjectType"]

- Always compare using:
    obj["objectType"] == TARGET_TYPE

NEVER hardcode object names.

    
- All observation values are 1D arrays of size 1
- ALWAYS use index [0]

Correct:
    env.last_obs["distance"][0]
    env.last_obs["center_x"][0]

WRONG:
    env.last_obs["distance"][1]
    env.last_obs["center_x"][1]

ENV & GENERALIZATION:
- The target object changes every episode.
- Use `env.target_object_type` to get the target object's name as a string. NEVER hardcode object names like "Mug" or "Apple".
- Use `env.controller.last_event.metadata` for full state.
- Use `interacted = env.get_interacted_objects()` to easily get lists of objects currently in 'inventory', 'opened', 'toggled', or 'broken'.
  Example: `any(obj["objectType"] == env.target_object_type for obj in interacted["inventory"])`

CRITICAL:

You MUST define EXACTLY this function:

def _get_rewards_eureka(env):

If you fail, the code will crash.

IMPORTANT (VERY IMPORTANT):

The agent ONLY sees the observation.

You MUST design reward using:

env.last_obs

Available fields:

- env.last_obs["distance"] → distance to target
- env.last_obs["visible"] → 0 or 1
- env.last_obs["center_x"], ["center_y"], ["center_z"]

DO NOT rely only on metadata.
Use observation-based reward shaping.

Example:

if env.last_obs["visible"]:
    total_reward += 0.5

total_reward += -env.last_obs["distance"][0]

Example Reward Function:

def _get_rewards_eureka(env):
    total_reward = 0
    rewards_dict = {}

    current_distance = env.last_obs["distance"][0]

    if not hasattr(env, "prev_distance"):
        env.prev_distance = current_distance

    # progress reward
    progress = env.prev_distance - current_distance
    total_reward += progress * 10.0

    env.prev_distance = current_distance

    # visibility reward
    if env.last_obs["visible"][0]:
        total_reward += 1.0

    # proximity reward
    if current_distance < 1.0:
        total_reward += 2.0

    if current_distance < 0.5:
        total_reward += 5.0

    # success reward
    metadata = env.controller.last_event.metadata
    TARGET_TYPE = env.target_object_type

    if any(obj["objectType"] == TARGET_TYPE for obj in metadata["inventoryObjects"]):
        total_reward += 100.0

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
            f.write("="*50 + "\n\n")

        task_manager = EurekaTaskManager(
            num_processes=NUM_SUGGESTIONS,
            max_training_iterations=TRAINING_STEPS,
            category=subtask_info['category']
        )
        target_object_type = task_manager._target_object_type

        print(f'Target object: {target_object_type}')

        best_score = -float("inf")
        best_reward_code = None

        last_feedback = f"Focus ONLY on subtask: {subtask}"

        for i in range(MAX_ITERATIONS):
            print(f"\n🔄 Iter {i+1}")

            reward_prompt = f"""
Main Task:
{TASK_DESCRIPTION}

Current Subtask:
{subtask}

Previous Reward Function:
{best_reward_code if i > 0 else "None"}

Feedback:
{last_feedback}

Improve the reward.

Improve the previous reward function.
DO NOT repeat the same logic.

CRITICAL:

You MUST define EXACTLY this function:

def _get_rewards_eureka(env):

If you fail, the code will crash.
"""

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

            # 🔥 여기 핵심: model 가져오기
            # best_state_dict = result["model_state_dict"]
            if score > best_score:
                best_score = score
                best_reward_code = reward_strings[0]

            last_feedback += f"\nLast Score: {score}, Last Success Rate: {success_rate}\nLast Reward Function Code: \n{reward_code}"
            # print(last_feedback)

         # 🔥 best reward 기록
        with open(log_path, "a") as f:
            f.write(f"🔥 BEST (score={score})\n")
            f.write(reward_code + "\n\n")

        print(f"✅ Best Score: {best_score:.4f}")
        print("\n🔥 Final training with BEST reward function...")

        final_reward_data = [{
            "reward_code": best_reward_code,
            "success_code": success_code,
            "precondition_code": s["pre"]
        }]

        # 🔥 더 길게 학습 (중요)
        task_manager._max_training_iterations = TRAINING_STEPS * 3

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