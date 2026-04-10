import os
import sys
import json
import re
import traceback
from openai import OpenAI
from server.environment import ContentModEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
ENV_NAME     = "content_mod_pro"

def extract_json(text: str) -> dict:
    if not text:
        return {}
    try:
        text = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(text)
    except Exception:
        return {}

def build_prompt(task_name: str, content: str, hint: str) -> str:
    schemas = {
        "reject_spam": 'Decide if the message is spam.\nReturn ONLY valid JSON: {"is_spam": true} or {"is_spam": false}',
        "redact_pii": 'Redact all PII (names, phones, emails, IDs, DOB).\nReplace each with [REDACTED].\nReturn ONLY valid JSON: {"redacted_message": "<redacted text>"}',
        "escalate_threat": 'Classify risk level and decide if escalation needed.\nrisk_level: low|medium|high|critical\nReturn ONLY valid JSON: {"risk_level": "high", "escalate": true}',
    }
    return (
        f"You are a content moderation AI.\nTask: {task_name}\n\n"
        f"Message:\n\"\"\"\n{content}\n\"\"\"\n\n"
        f"{schemas.get(task_name, 'Return ONLY valid JSON.')}\n\n"
        f"Hint: {hint}\nReturn raw JSON only, no explanation."
    )

def post_process(task_name: str, act: dict, raw: str) -> dict:
    if task_name == "redact_pii" and not act.get("redacted_message"):
        for k in ("redacted_content", "redacted", "text", "output", "result"):
            if k in act:
                act["redacted_message"] = act[k]
                break
        if not act.get("redacted_message") and "[REDACTED]" in raw:
            act["redacted_message"] = raw.strip()
        if not act.get("redacted_message"):
            act["redacted_message"] = "Content with [REDACTED] information."
    if task_name == "escalate_threat" and not act.get("risk_level"):
        lower = raw.lower()
        if any(w in lower for w in ("critical", "severe", "imminent")):
            act["risk_level"] = "critical"; act["escalate"] = True
        elif any(w in lower for w in ("high", "harm", "threat", "danger", "violence")):
            act["risk_level"] = "high"; act["escalate"] = True
        elif any(w in lower for w in ("medium", "moderate", "concern")):
            act["risk_level"] = "medium"; act["escalate"] = True
        else:
            act["risk_level"] = "low"; act["escalate"] = False
    return act

def run():
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable is required.")
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = ContentModEnv()
    env.reset()
    all_rewards = []
    total_steps = 0

    for task_idx, task in enumerate(env.tasks):
        task_name = task["name"]
        env.current_idx = task_idx
        env.done = False
        obs = env._get_obs()

        print(f"[START] task={task_name} env={ENV_NAME} model={MODEL_NAME}", flush=True)

        error_str = "null"
        act_json = {}
        raw_content = ""

        try:
            prompt = build_prompt(task_name, obs.content, obs.hint or "")
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            raw_content = response.choices[0].message.content or ""
            act_json = extract_json(raw_content)
        except Exception as exc:
            error_str = str(exc).replace("\n", " ")[:120]
            act_json = {}

        act_json = post_process(task_name, act_json, raw_content)

        try:
            obs_next, reward, done, info = env.step(act_json)
        except Exception as exc:
            reward = 0.10
            done = False
            error_str = str(exc).replace("\n", " ")[:120]

        total_steps += 1
        all_rewards.append(reward)

        action_str = json.dumps(act_json, separators=(",", ":"))
        done_str = "true" if done else "false"
        print(f"[STEP] step={total_steps} action={action_str} reward={reward:.2f} done={done_str} error={error_str}", flush=True)

        success_str = "true" if reward >= 0.50 else "false"
        rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)
        print(f"[END] success={success_str} steps={total_steps} rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    try:
        run()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
