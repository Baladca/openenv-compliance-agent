from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ModAction(BaseModel):
    action_type: str 
    target_text: Optional[str] = None
    reason: Optional[str] = None

class ModObservation(BaseModel):
    content: str
    metadata: Dict[str, Any]
    task_id: int
    current_score: float
    logs: List[str]

class ModState(BaseModel):
    current_idx: int
    cumulative_reward: float
    done: bool