import re
from pydantic import BaseModel, ConfigDict
from typing import Optional, Any, Tuple, List, Dict


# ---------------------------------------------------------------------------
# Action / Observation models
# ---------------------------------------------------------------------------

class ModAction(BaseModel):
    model_config = ConfigDict(extra="ignore")

    is_spam: Optional[bool] = None
    redacted_message: Optional[str] = None
    risk_level: Optional[str] = None   # "low" | "medium" | "high" | "critical"
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


# ---------------------------------------------------------------------------
# Dataset — more realistic messages per task
# ---------------------------------------------------------------------------

TASK_DATA: List[Dict] = [
    {
        "id": 1,
        "name": "reject_spam",
        "difficulty": "easy",
        "messages": [
            "Congratulations!!! You have been selected to win 10 lakh rupees! Click NOW!",
            "FREE iPhone 15 Pro — claim your prize before midnight. Limited offer!!!",
            "URGENT: Your account has been suspended. Verify immediately at bit.ly/scam",
        ],
        "expected_label": True,   # is_spam should be True
        "hint": "Return JSON: {\"is_spam\": true}",
    },
    {
        "id": 2,
        "name": "redact_pii",
        "difficulty": "medium",
        "messages": [
            "Hi, my name is Ravi Kumar and my phone number is 9876543210.",
            "Please contact me at john.doe@gmail.com or call +1-800-555-0199.",
            "My Aadhaar number is 1234 5678 9012 and DOB is 15/08/1990.",
        ],
        "expected_label": "[REDACTED]",
        "hint": "Return JSON: {\"redacted_message\": \"Hi, my name is [REDACTED] and my phone number is [REDACTED].\"}",
    },
    {
        "id": 3,
        "name": "escalate_threat",
        "difficulty": "hard",
        "messages": [
            "I am going to seriously harm someone at the office tomorrow morning.",
            "If they fire me I will make sure everyone regrets it. I know where they live.",
            "This is not a joke. I have a plan and I will execute it this Friday.",
        ],
        "expected_label": {"risk_level": "high", "escalate": True},
        "hint": "Return JSON: {\"risk_level\": \"high\", \"escalate\": true}",
    },
]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

PII_PATTERN = re.compile(
    r"(\b\d{10}\b"                        # 10-digit phone
    r"|\b\d{4}\s?\d{4}\s?\d{4}\b"         # Aadhaar
    r"|[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"  # email
    r"|\+?\d[\d\s\-]{8,14}\d"             # international phone
    r"|\b\d{2}/\d{2}/\d{4}\b"             # date of birth
    r"|(?i)\b(ravi|john|kumar|doe)\b)",    # proper names from dataset
    re.IGNORECASE,
)


def _clamp(v: float, lo: float = 0.01, hi: float = 0.99) -> float:
    """Strictly clamp score to open interval (0, 1)."""
    return max(lo, min(hi, float(v)))


def _grade_reject_spam(action: ModAction, task: Dict) -> float:
    """Binary: correct detection = 0.85, wrong = 0.15."""
    if action.is_spam is True:
        return 0.85
    if action.is_spam is False:
        return 0.15
    # No answer provided
    return 0.10


def _grade_redact_pii(action: ModAction, original: str) -> float:
    """
    Partial-credit grading:
      - Base score 0.30 if redacted_message is present at all
      - +0.30 if '[REDACTED]' or 'REDACTED' appears in response
      - +0.20 if no raw PII pattern remains in the response
      - +0.10 if the overall length is plausible (not too short / too long)
    Max possible: 0.90
    """
    val = action.redacted_message or ""
    if not val.strip():
        return 0.10

    score = 0.30

    # Redaction markers present
    if re.search(r"\[REDACTED\]|REDACTED", val, re.IGNORECASE):
        score += 0.30

    # No raw PII remaining
    if not PII_PATTERN.search(val):
        score += 0.20

    # Length sanity check (between 10 and 500 chars)
    if 10 <= len(val) <= 500:
        score += 0.10

    return score  # already in [0.30, 0.90] range


def _grade_escalate_threat(action: ModAction) -> float:
    """
    Partial credit:
      - 0.10  no useful output
      - 0.40  escalate=True but wrong/missing risk_level
      - 0.60  risk_level='medium' and escalate=True (partially correct)
      - 0.85  risk_level='high' or 'critical' AND escalate=True
    """
    if action.escalate is True:
        rl = (action.risk_level or "").lower().strip()
        if rl in ("high", "critical"):
            return 0.85
        if rl == "medium":
            return 0.60
        # escalate=True but unknown level
        return 0.40
    # Missed escalation entirely
    return 0.10


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ContentModEnv:
    def __init__(self):
        self.tasks = TASK_DATA
        self.current_idx: int = 0
        self.done: bool = False
        self.scores: List[float] = []

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

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
        # Parse action
        try:
            if isinstance(action, dict):
                parsed = ModAction(**action)
            elif isinstance(action, ModAction):
                parsed = action
            else:
                parsed = ModAction()
        except Exception:
            parsed = ModAction()

        idx = max(0, min(self.current_idx, len(self.tasks) - 1))
        task = self.tasks[idx]
        task_id = task["id"]
        original_msg = task["messages"][0]  # use first message as canonical

        # Grade
        if task_id == 1:
            raw_score = _grade_reject_spam(parsed, task)
        elif task_id == 2:
            raw_score = _grade_redact_pii(parsed, original_msg)
        elif task_id == 3:
            raw_score = _grade_escalate_threat(parsed)
        else:
            raw_score = 0.10

        # Strict clamp — NEVER exactly 0.0 or 1.0
        reward = _clamp(raw_score)
        self.scores.append(reward)

        # Advance
        success = reward >= 0.50
        if success:
            self.current_idx += 1

        if self.current_idx >= len(self.tasks):
            self.done = True

        return self._get_obs(), reward, self.done, {"task_id": task_id, "success": success}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> ModObservation:
        safe_idx = max(0, min(self.current_idx, len(self.tasks) - 1))
        task = self.tasks[safe_idx]
        return ModObservation(
            content=task["messages"][0],
            task_id=task["id"],
            task_name=task["name"],
            difficulty=task["difficulty"],
            done=self.done,
            hint=task.get("hint"),
        )
