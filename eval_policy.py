"""Оценка обученной политики."""
import torch
import numpy as np
from tqdm import trange

from ocean.env import OceanEnv
from ocean.config import SimConfig
from train_dqn import QNet


def eval_policy(model_path="dqn.pt", episodes=10):
    """Оценка обученной политики."""
    cfg = SimConfig()
    env = OceanEnv(cfg)
    
    P = cfg.patch_size
    nA = 5
    
    net = QNet(P, nA)
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()
    
    def to_t(x):
        return torch.as_tensor(x, dtype=torch.float32)
    
    episode_rewards = []
    episode_lengths = []
    
    for ep in trange(episodes, desc="Оценка"):
        obs, _ = env.reset(seed=ep)
        total_reward = 0.0
        steps = 0
        
        while True:
            with torch.no_grad():
                q = net(to_t(obs["obs_map"][None]), to_t(obs["obs_vec"][None]))
                action = int(q.argmax(1).item())
            
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
    
    print(f"\nРезультаты оценки ({episodes} эпизодов):")
    print(f"Средняя награда: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Средняя длина: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Максимальная награда: {np.max(episode_rewards):.2f}")
    print(f"Минимальная награда: {np.min(episode_rewards):.2f}")


if __name__ == "__main__":
    eval_policy()

