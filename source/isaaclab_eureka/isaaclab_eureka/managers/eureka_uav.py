# eureka_main.py
import os
import re
import multiprocessing as mp
import inspect

from llm_manager import LLMManager
from eureka_task_manager import EurekaTaskManager, EurekaThorWrapper
from llmsf.env_rl import GridWorldMultiAgentEnv
from policy_manager import PolicyManager
from datetime import datetime

import ray
from ray.rllib.algorithms.ppo import PPOConfig

ray.init()

# --- Config ---
GPT_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"
NUM_SUGGESTIONS = 5
TEMPERATURE = 1.0
MAX_ITERATIONS = 10
# TRAINING_STEPS = 100
TRAINING_STEPS = 50000

TASK_DESCRIPTION = "3 UAVs must visit start and then reach end as fast as possible"

SYSTEM_PROMPT = f"""
You are a reward engineer trying to write reward functions to solve reinforcement learning tasks as effective as possible.
Your goal is to write a reward function for the environment that will help the agent learn the task described in text.
Your reward function should use useful variables from the environment as inputs.

The environment is a GridWorld with multiple UAVs.
The environment source code is:
{inspect.getsource(GridWorldMultiAgentEnv)}

There are several enemies which UAVs have to avoid.

Each UAV has:
- location: (x, y)
- goal: (x, y)
- battery
- finished (True if reached goal)

Global:
- env.start
- env.end
- env.uavs (list)

Task:
- UAVs must FIRST visit start point
- THEN reach end point
- As fast as possible

IMPORTANT:
- Use env.uavs[i].location
- Use env.start, env.end
- Use distance-based rewards (Manhattan distance)
- Add step penalty

Return format:

def _get_rewards_eureka(env):
    ...
    ret

"""

def check_success(env):
    # 모든 UAV가 start 방문 + goal 도착
    for i in range(env.uav_num):
        if not env.uavs[i].finished:
            return False
    return True

import json
import random

def generate_skills(llm, max_skills=5):
    prompt = f"""
You are a robotics skill designer.

Your job is to design reusable SKILLS for a multi-agent reinforcement learning system.

--------------------------------------------------
ENVIRONMENT CODE:
{inspect.getsource(GridWorldMultiAgentEnv)}

--------------------------------------------------
TASK:
{TASK_DESCRIPTION}

--------------------------------------------------
GOAL:
Generate a list of reusable SKILLS that agents can use.

Each skill must:
- Be reusable across environments
- Be behavior-level (NOT low-level actions)
- Represent a meaningful strategy

--------------------------------------------------
GOOD SKILLS:
- safe_move
- fast_move
- explore_area
- avoid_enemy
- lure_enemy
- coordinate_move

BAD SKILLS:
- move_left
- move_right
- go_to_3_4
- step_forward

--------------------------------------------------
RULES:
- Each skill MUST be abstract
- Each skill MUST be useful for solving the task
- Maximum {max_skills} skills

--------------------------------------------------
OUTPUT FORMAT (STRICT JSON ONLY):
[
  {{
    "label": "snake_case_name",
    "description": "what this skill does",
    "use_when": "when this skill should be used"
  }}
]

DO NOT OUTPUT ANYTHING ELSE.
"""

    res = llm.prompt(prompt)

    # -------------------------------
    # 🔥 Robust parsing
    # -------------------------------
    if isinstance(res, dict):
        text = res.get("raw_outputs", [""])[0]
    else:
        text = str(res)

    text = text.strip()

    # 1. direct json
    try:
        skills = json.loads(text)
        return validate_skills(skills)
    except:
        pass

    # 2. extract json
    try:
        json_str = re.search(r"\[.*\]", text, re.DOTALL).group()
        skills = json.loads(json_str)
        return validate_skills(skills)
    except:
        pass

    print("⚠️ Skill parsing failed, fallback to default skills")
    return []


# -------------------------------
# ✅ validation
# -------------------------------
def validate_skills(skills):
    validated = []

    for s in skills:
        label = s.get("label", "").strip().lower().replace(" ", "_")
        desc = s.get("description", "").strip()
        use_when = s.get("use_when", "").strip()

        if not label:
            continue

        validated.append({
            "label": label,
            "description": desc,
            "use_when": use_when
        })

    return validated

def parse_rewards_for_config(llm_outputs):
    parsed_rewards = []
    
    for output in llm_outputs:
        if 'reward_code' in output:
            # LLM 출력 문자열에 포함된 Non-breaking space(\xa0)를 일반 공백으로 치환
            clean_code = output['reward_code'].replace('\xa0', ' ')
            parsed_rewards.append(clean_code)
            
    return parsed_rewards

def run_train_loop():
    llm = LLMManager(
        gpt_model=GPT_MODEL,
        num_suggestions=NUM_SUGGESTIONS,
        temperature=TEMPERATURE,
        system_prompt=SYSTEM_PROMPT
    )

    policy_manager = PolicyManager()

    os.makedirs("outputs/reward_shaping_logs_uavs", exist_ok=True)

    skills = generate_skills(llm)
    print(skills)

    # -------------------------------
    # Subtask loop
    # -------------------------------
    for s_idx, s in enumerate(skills):
        skill_name = s["label"]
        skill_desc = s["description"]

        print(f"\n🚀 [Skill {s_idx+1}] {skill_name}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = f"outputs/reward_shaping_logs_uavs/{skill_name}_{timestamp}.txt"

        # 파일 초기화
        with open(log_path, "w") as f:
            f.write(f"Task: {TASK_DESCRIPTION}\n")
            f.write(f"Skill name: {skill_name}\n")
            f.write("="*50 + "\n\n")

        # task_manager = EurekaTaskManager(
        #     num_processes=NUM_SUGGESTIONS,
        #     max_training_iterations=TRAINING_STEPS,
        #     category='navigate'
        # )
        best_reward_code = None
        best_success_rate = -1

        for i in range(MAX_ITERATIONS):
            print(f"\n🔄 Iter {i+1}")
            
            # reward_components_feedback = components_feedback if components_feedback else "No feedback available yet (First Iteration)."
            reward_prompt = f"""
You are designing a reward function for the following skill:

Skill: {skill_name}
Description: {skill_desc}

Environment:
- Multiple UAVs
- Each UAV has location, battery, goal
- Enemies exist (avoid them)

IMPORTANT:
- Input parameter must be uav, enemy_locs
- Add step penalty

Return:

def _get_rewards_eureka(uav, enemy_locs):
    ...
"""
            # print(reward_prompt)
            # task_manager.thor_env.reset_reward_components_per_epoches()

            response = llm.prompt(reward_prompt)
            #print("response:", response)
            reward_strings = response["reward_strings"]
            reward_data = [{"reward_code": r} for r in reward_strings]
            print(reward_data)
            print()

            config = (
            PPOConfig()
            .environment(
                env=GridWorldMultiAgentEnv,
                env_config={
                    'reward_function': parse_rewards_for_config(reward_data)
                    }
                )
                .api_stack(
                    enable_rl_module_and_learner=False,
                    enable_env_runner_and_connector_v2=False
                )
                .framework("torch")
                .env_runners(num_env_runners=2)
                .training(
                    train_batch_size=1000,
                    gamma=0.99,
                    lr=5e-4
                )
                .multi_agent(
                    policies={"shared_policy"},
                    policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
                )
            )

            algo = config.build()

            results = algo.train()

            successful = [r for r in results if r.get("success", False)]

            if not successful:
                continue

            best_iter = max(successful, key=lambda x: x["success_rate"])

            if best_iter["success_rate"] > best_success_rate:
                best_success_rate = best_iter["success_rate"]
                best_reward_code = best_iter["reward_code"]

            
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
            successful_results = [r for r in results if r.get("success", False)]

        
            if not successful_results:
                last_feedback = "All reward suggestions failed to execute. Please check the syntax and environment API."
                continue
            best_iter_result = max(successful_results, key=lambda x: x["train_success_rate"])
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

        print(f"✅ Best success rate: {best_success_rate:.4f}")
        print("\n🔥 Final training with BEST reward function...")

        final_reward_data = [{
            "reward_code": best_reward_code,
        }]

        # 🔥 더 길게 학습 (중요)
        task_manager._max_training_iterations = TRAINING_STEPS * 10

        final_results = task_manager.train(final_reward_data)
        final_result = final_results[0]

        if final_result["success"]:
            final_model = final_result["model_state_dict"]
            policy_manager.save_policy(final_model, skill_name)

            print(f"✅ Final model saved (score={final_result['reward_mean']}, success={final_result['success_rate']})")
        else:
            print("❌ Final training failed:", final_result["exception"])

    task_manager.close()


if __name__ == "__main__":
    # mp.set_start_method("spawn", force=True)
    run_train_loop()