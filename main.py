from lerobot.policies import SmolVLAConfig
import torch
import gymnasium as gym


device = "cuda" if torch.cuda.is_available() else "cpu"
policy = SmolVLAConfig.from_pretrained("hamzabouajila/franka_smolvla", force_download=True)
policy.to(device)

env = gym.make("panda_pick_and_place", obs_type="...")

obs = env.reset()
policy.reset()

frames = [env.render()]
rewards = []

while True:
    # prepare input dict based on env observation format
    action = policy.select_action({...})
    obs, r, terminated, truncated, _ = env.step(action.cpu().numpy())
    frames.append(env.render())
    rewards.append(r)
    if terminated or truncated: break

# save video likewise
