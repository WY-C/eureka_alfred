# eureka_main.py
import os
import re
from llm_manager import LLMManager
from eureka_task_manager import EurekaTaskManager

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

ENV:
Use env.controller.last_event.metadata

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

# -------------------------------
# ✅ main loop
# -------------------------------
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

    os.makedirs("outputs/eira_logs", exist_ok=True)

    # --- Subtasks 생성 ---
    subtask_plan = generate_subtasks(llm, TASK_DESCRIPTION)
    subtasks = parse_subtasks(subtask_plan)
    print(subtask_plan)

    print("🧩 Subtasks:", subtasks)

    # --- Subtask loop ---
    for s_idx, s in enumerate(subtasks):
        subtask = s["subtask"]
        success_code = s["success"]

        print(f"\n🚀 [Subtask {s_idx+1}] {subtask}")
        print(f"🎯 SuccessCode: {success_code}")

        subtask_log_path = f"outputs/reward_shaping_logs/subtask_{s_idx+1}_log.txt"

        with open(subtask_log_path, "w") as f:
            f.write(f"Subtask: {subtask}\n")
            f.write(f"SuccessCode: {success_code}\n")
            f.write("="*50 + "\n")

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
{best_reward_code if i > 0 else "None(It is the first iteration.)"}

Feedback:
{last_feedback}

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

            # 🔥 실패 케이스
            if not result["success"]:
                error_msg = result["exception"]

                last_feedback = f"""
        Your previous code failed with this error:

        {error_msg}

        Fix the reward function.
        """

                continue
                # ✅ 성공 케이스
            score = result["reward_mean"]

            last_feedback = f"""
        Score: {score}
        """

            if score < 1:
                last_feedback += "Reward rarely triggers. Add more dense intermediate rewards."
            elif score < 5:
                last_feedback += "Reward exists but not guiding behavior. Add distance-based reward."
            else:
                last_feedback += "Refine success condition and avoid reward hacking."

            best_score = max(best_score, score)

            # 로그 저장
            with open(subtask_log_path, "a") as f:
                f.write(f"\n--- Iter {i+1} ---\n")
                f.write(f"Score: {score}\n")
                f.write("\n[Reward]\n")
                f.write(reward_strings[0] + "\n")

        # -----------------------
        # ✅ success rate 측정
        # -----------------------
        print("📊 Evaluating success rate...")

        env = task_manager._processes[0]  # 실제론 worker에서 가져오는 구조 필요 (간단히 설명용)

        # ⚠️ 실제론 env 접근 구조 바꿔야 함 (아래 참고)
        success_rate = 0  # placeholder

        with open(subtask_log_path, "a") as f:
            f.write("\n🔥 BEST REWARD 🔥\n")
            f.write(best_reward_code or "")
            f.write(f"\nBest Score: {best_score:.4f}\n")

        print(f"✅ Best Score: {best_score:.4f}")

    task_manager.close()
    print("\n🏁 Done!")


if __name__ == "__main__":
    run_train_loop()