from .models import ModAction, ModObservation, ModState

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
        self.cumulative_reward = 0.05
        self.done = False
        return self._get_obs("Session Started")

    def state(self) -> ModState:
        return ModState(
            current_idx=self.current_idx,
            cumulative_reward=self.cumulative_reward,
            done=self.done
        )

    def step(self, action_dict: dict):
        if hasattr(action_dict, 'dict'):
            action_dict = action_dict.dict()
            
        reward = 0.01 
        idx = min(self.current_idx, len(self.tasks) - 1)
        task = self.tasks[idx]
        
        action_type = action_dict.get("action_type")
        target_text = action_dict.get("target_text")

        if action_type == task["target"]:
            if task["target"] == "redact":
                if target_text == task["val"]:
                    reward = 0.99 
                    self.current_idx += 1
                    msg = f"Task {task['id']} Success."
                else:
                    reward = 0.3 
                    msg = f"Task {task['id']} Partial."
            else:
                reward = 0.99 
                self.current_idx += 1
                msg = f"Task {task['id']} Success."
        else:
            reward = 0.05 
            msg = f"Incorrect action for Task {task['id']}."

        if self.current_idx >= len(self.tasks):
            self.done = True
                
        self.cumulative_reward = reward
        return self._get_obs(msg), reward, self.done, {"last_action_error": None}

    def _get_obs(self, log_msg):
        display_idx = min(self.current_idx, len(self.tasks) - 1)
        task = self.tasks[display_idx]
        
        # Safe score map: 0.1, 0.33, 0.66, 0.99
        progress_map = {0: 0.10, 1: 0.33, 2: 0.66, 3: 0.99}
        clamped_score = progress_map.get(self.current_idx, 0.99)
        
        return ModObservation(
            content=task["text"],
            metadata={"difficulty": task["difficulty"]},
            task_id=task["id"],
            current_score=float(clamped_score),
            logs=[log_msg]
        )