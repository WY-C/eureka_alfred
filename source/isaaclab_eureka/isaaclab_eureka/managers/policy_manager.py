# policy_manager.py
import os
import torch

class PolicyManager:
    def __init__(self, save_dir="outputs/policies"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save_policy(self, state_dict, label):
        path = os.path.join(self.save_dir, f"{label}.pt")
        torch.save(state_dict, path)
        print(f"💾 Saved policy: {label}")

    def load_policy(self, model, label):
        path = os.path.join(self.save_dir, f"{label}.pt")
        state_dict = torch.load(path)
        model.policy.load_state_dict(state_dict)
        return model