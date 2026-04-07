# eureka_main.py
import os
import re
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

def generate_policy_label(llm, subtask):
    prompt = f"""
Subtask: {subtask}

Generate a short, general policy name.

Rules:
- snake_case
- verb + object
- generalizable

Examples:
pick_up_object
navigate_to_object
place_object_on_receptacle

ONLY return label.
"""
    res = llm.prompt(prompt)

    if isinstance(res, dict):
        return res["raw_outputs"][0].strip()
    return str(res).strip()

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

# -------------------------------
# ✅ success eval 함수
# -------------------------------
def check_success(env, code):
    metadata = env.unwrapped.controller.last_event.metadata

    try:
        return eval(code, {"metadata": metadata})
    except Exception as e:
        print(f"⚠️ Success eval error: {e}")
        return False


def run_train_loop():
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

        # 🔥 LLM으로 policy label 생성
        policy_label = generate_policy_label(llm, subtask)
        print(f"🏷 Policy Label: {policy_label}")

        best_score = -float("inf")
        best_reward_code = None
        best_model = None

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

            if score < 1:
                last_feedback += "Reward rarely triggers. Add more dense intermediate rewards."
            elif score < 5:
                last_feedback += "Reward exists but not guiding behavior. Add distance-based reward."
            else:
                last_feedback += "Refine success condition and avoid reward hacking."

            # 🔥 여기 핵심: model 가져오기
            best_state_dict = result["model_state_dict"]
            if score > best_score:
                best_score = score
                best_reward_code = reward_strings[0]
                best_model = best_state_dict  # 베스트 모델 갱신
                policy_manager.save_policy(best_state_dict, policy_label)
                print(f'Policy updated(score: {score})')

            last_feedback = f"Score: {score}"
            print(last_feedback)

        # -------------------------------
        # ✅ policy 저장
        # -------------------------------
        if best_model is not None:
            policy_manager.save_policy(best_model, policy_label)

        print(f"✅ Best Score: {best_score:.4f}")

    task_manager.close()


if __name__ == "__main__":
    run_train_loop()