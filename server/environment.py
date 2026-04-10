import re
from pydantic import BaseModel, ConfigDict
from typing import Optional, Any, Tuple, List, Dict


class ModAction(BaseModel):
    model_config = ConfigDict(extra="ignore")
    is_spam: Optional[bool] = None
    redacted_message: Optional[str] = None
    risk_level: Optional[str] = None
    escalate: Optional[bool] = None


class ModObservation(BaseModel):
    content: str
    task_id: int
    task_name: str
    difficulty: str
    done: bool
    hint: Optional[str] = None


class ModState(BaseModel):
    current_idx: int
    cumulative_reward: float
    done: bool
    scores: List[float]


TASK_DATA: List[Dict] = [
    {
        "id": 1, "name": "reject_spam", "difficulty": "easy",
        "messages": ["Congratulations!!! You have been selected to win 10 lakh rupees! Click NOW!"],
        "hint": 'Return JSON: {"is_spam": true}',
    },
    {
        "id": 2, "name": "redact_pii", "difficulty": "medium",
        "messages": ["Hi, my name is Ravi Kumar and my phone number is 9876543210."],
        "hint": 'Return JSON: {"redacted_message": "Hi, my name is [REDACTED] and my phone number is [REDACTED]."}',
    },
    {
        "id": 3, "name": "escalate_threat", "difficulty": "hard",
        "messages": ["I am going to seriously harm someone at the office tomorrow morning."],
        "hint": 'Return JSON: {"risk_level": "high", "escalate": true}',
    },
]

PII_PATTERN = re.compile(
    r"(\b\d{10}\b|\b\d{4}\s?\d{4}\s?\d{4}\b"
    r"|[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    r"|\+?\d[\d\s\-]{8,14}\d|\b\d{2}/\d{2}/\d{4}\b"
    r"|(?i)\b(ravi|john|kumar|doe)\b)",
    re.IGNORECASE,
)


def _clamp(v: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return max(lo, min(hi, float(v)))


def _grade_reject_spam(action: ModAction) -> float:
    if action.is_spam is True:  return 0.85
    if action.is_spam is False: return 0.15
    return 0.10


def _grade_redact_pii(action: ModAction) -> float:
    val = action.redacted_message or ""
    if not val.strip(): return 0.10
    score = 0.30
    if re.search(r"\[REDACTED\]|REDACTED", val, re.IGNORECASE): score += 0.30
    if not PII_PATTERN.search(val): score += 0.20
    if 10 <= len(val) <= 500: score += 0.10
    return score


def _grade_escalate_threat(action: ModAction) -> float:
    if action.escalate is True:
        rl = (action.risk_level or "").lower().strip()
        if rl in ("high", "critical"): return 0.85
        if rl == "medium": return 0.60
        return 0.40
    return 0.10


class ContentModEnv:
    def __init__(self):
        self.tasks = TASK_DATA
        self.current_idx: int = 0
        self.done: bool = False
        self.scores: List[float] = []

    def reset(self) -> ModObservation:
        self.current_idx = 0
        self.done = False
        self.scores = []
        return self._get_obs()

    def state(self) -> ModState:
        return ModState(
            current_idx=self.current_idx,
            cumulative_reward=sum(self.scores),
            done=self.done,
            scores=list(self.scores),
        )

    def step(self, action: Any) -> Tuple[ModObservation, float, bool, dict]:
        try:
            parsed = ModAction(**(action if isinstance(action, dict) else {}))
        except Exception:
            parsed = ModAction()

        idx = max(0, min(self.current_idx, len(self.tasks) - 1))
        task_id = self.tasks[idx]["id"]

        if task_id == 1:   raw = _grade_reject_spam(parsed)
        elif task_id == 2: raw = _grade_redact_pii(parsed)
        elif task_id == 3: raw = _grade_escalate_threat(parsed)
        else:              raw = 0.10

        reward = _clamp(raw)
        self.scores.append(reward)

        if reward >= 0.50:
            self.current_idx += 1
        if self.current_idx >= len(self.tasks):
            self.done = True

        return self._get_obs(), reward, self.done, {"task_id": task_id}

    def _get_obs(self) -> ModObservation:
        i = max(0, min(self.current_idx, len(self.tasks) - 1))
        t = self.tasks[i]
        return ModObservation(
            content=t["messages"][0], task_id=t["id"],
            task_name=t["name"], difficulty=t["difficulty"],
            done=self.done, hint=t.get("hint"),
        )
