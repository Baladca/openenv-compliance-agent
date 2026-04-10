from pydantic import BaseModel, ConfigDict
from typing import Optional, List


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
