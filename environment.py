from models import ModAction, ModObservation, ModState

class ContentModEnv:
    def __init__(self):
        self.tasks = [
            {"id": 1, "text": "WIN A FREE IPHONE AT EVIL-LINK.COM!!!", "target": "reject", "difficulty": "easy"},
            {"id": 2, "text": "My email is john.doe@email.com", "target": "redact", "val": "john.doe@email.com", "difficulty": "medium"},
            {"id": 3, "text": "I am going to bypass the security firewall now.", "target": "escalate", "difficulty": "hard"}
        ]
        self.reset()

    def reset(self):
        self.current_idx = 0
        self.cumulative_reward = 0.0
        self.done = False
        return self._get_obs("Session Started")

    def state(self) -> ModState:
        return ModState(
            current_idx=self.current_idx,
            cumulative_reward=self.cumulative_reward,
            done=self.done
        )

    def step(self, action_dict: dict):
        reward = 0.0
        idx = min(self.current_idx, len(self.tasks)-1)
        task = self.tasks[idx]
        
        action_type = action_dict.get("action_type")
        target_text = action_dict.get("target_text")

        if action_type == task["target"]:
            if task["target"] == "redact":
                if target_text == task["val"]:
                    reward = 1.0
                    msg = f"Task {task['id']} Success: PII isolated."
                    self.current_idx += 1
                else:
                    reward = 0.3 
                    msg = f"Task {task['id']} Partial: Substring mismatch."
            else:
                reward = 1.0
                msg = f"Task {task['id']} Success."
                self.current_idx += 1
        else:
            reward = -0.4
            msg = f"Incorrect action for Task {task['id']}."

        if self.current_idx >= len(self.tasks):
            self.done = True
                
        self.cumulative_reward += reward
        return self._get_obs(msg), reward, self.done, {"last_action_error": None}

    def _get_obs(self, log_msg):
        idx = min(self.current_idx, len(self.tasks)-1)
        task = self.tasks[idx]
        return ModObservation(
            content=task["text"],
            metadata={"difficulty": task["difficulty"]},
            task_id=task.get("id", 0),
            current_score=round(self.current_idx / len(self.tasks), 2),
            logs=[log_msg]
        )