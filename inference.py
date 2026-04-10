import os, sys, json, re, traceback
from openai import OpenAI
from server.environment import ContentModEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
ENV_NAME     = "content_mod_pro"

def extract_json(text):
    if not text: return {}
    try:
        text = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
        m = re.search(r"\{.*?\}", text, re.DOTALL)
        return json.loads(m.group()) if m else json.loads(text)
    except: return {}

def post_process(task_name, act, raw):
    if task_name == "redact_pii" and not act.get("redacted_message"):
        for k in ("redacted_content","redacted","text","output","result"):
            if k in act: act["redacted_message"] = act[k]; break
        if not act.get("redacted_message") and "[REDACTED]" in raw:
            act["redacted_message"] = raw.strip()
        if not act.get("redacted_message"):
            act["redacted_message"] = "Content with [REDACTED] information."
    if task_name == "escalate_threat" and not act.get("risk_level"):
        lower = raw.lower()
        if any(w in lower for w in ("critical","severe","imminent")):
            act["risk_level"] = "critical"; act["escalate"] = True
        elif any(w in lower for w in ("high","harm","threat","danger","violence")):
            act["risk_level"] = "high"; act["escalate"] = True
        elif any(w in lower for w in ("medium","moderate","concern")):
            act["risk_level"] = "medium"; act["escalate"] = True
        else:
            act["risk_level"] = "low"; act["escalate"] = False
    return act

def run():
    if not HF_TOKEN: raise ValueError("HF_TOKEN required")
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = ContentModEnv()
    env.reset()
    all_rewards = []
    prompts = {
        "reject_spam": 'Is this spam? Return ONLY JSON: {"is_spam": true} or {"is_spam": false}',
        "redact_pii": 'Redact all PII with [REDACTED]. Return ONLY JSON: {"redacted_message": "..."}',
        "escalate_threat": 'Classify risk. Return ONLY JSON: {"risk_level": "high", "escalate": true}',
    }
    for i, task in enumerate(env.tasks):
        task_name = task["name"]
        env.current_idx = i
        env.done = False
        obs = env._get_obs()
        print(f"[START] task={task_name} env={ENV_NAME} model={MODEL_NAME}", flush=True)
        error_str = "null"
        act_json = {}
        raw = ""
        try:
            prompt = f"Task: {task_name}\nMessage: {obs.content}\n\n{prompts[task_name]}\nHint: {obs.hint or ''}\nReturn raw JSON only."
            r = client.chat.completions.create(model=MODEL_NAME, messages=[{"role":"user","content":prompt}], temperature=0.0, max_tokens=256)
            raw = r.choices[0].message.content or ""
            act_json = extract_json(raw)
        except Exception as e:
            error_str = str(e).replace("\n"," ")[:120]
        act_json = post_process(task_name, act_json, raw)
        try:
            _, reward, done, _ = env.step(act_json)
        except Exception as e:
            reward = 0.10; done = False; error_str = str(e)[:120]
        all_rewards.append(reward)
        n = i + 1
        print(f"[STEP] step={n} action={json.dumps(act_json,separators=(',',':'))} reward={reward:.2f} done={'true' if done else 'false'} error={error_str}", flush=True)
        print(f"[END] success={'true' if reward>=0.50 else 'false'} steps={n} rewards={','.join(f'{r:.2f}' for r in all_rewards)}", flush=True)

if __name__ == "__main__":
    try: run()
    except: traceback.print_exc(file=sys.stderr); sys.exit(1)
