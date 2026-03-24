"""Обучение DQN агента."""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import trange

from ocean.env import OceanEnv
from ocean.config import SimConfig
from ocean.dataset import ReplayBuffer


class QNet(nn.Module):
    """Q-сеть для DQN."""
    
    def __init__(self, P, n_actions):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        flat = 32 * P * P
        self.head = nn.Sequential(
            nn.Linear(flat + 4, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )
    
    def forward(self, obs_map, obs_vec):
        """Прямой проход."""
        x = self.conv(obs_map)
        x = torch.cat([x, obs_vec], dim=1)
        return self.head(x)


def train_dqn(
    steps=50000,
    lr=1e-3,
    batch_size=64,
    buffer_size=100000,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=20000,
    save_path="dqn.pt"
):
    """Обучение DQN агента."""
    cfg = SimConfig()
    env = OceanEnv(cfg)
    obs, _ = env.reset()
    
    P = cfg.patch_size
    nA = 5
    
    net = QNet(P, nA)
    tgt = QNet(P, nA)
    tgt.load_state_dict(net.state_dict())
    
    opt = optim.Adam(net.parameters(), lr=lr)
    
    rb = ReplayBuffer(buffer_size, (3, P, P), (4,))
    
    def to_t(x):
        return torch.as_tensor(x, dtype=torch.float32)
    
    global_step = 0
    o = obs
    
    for _ in trange(steps, desc="Обучение"):
        eps = eps_end + (eps_start - eps_end) * np.exp(-global_step / eps_decay)
        
        with torch.no_grad():
            q = net(to_t(o["obs_map"][None]), to_t(o["obs_vec"][None]))
            a = int(q.argmax(1).item()) if np.random.rand() > eps else np.random.randint(nA)
        
        no, r, d, _, _ = env.step(a)
        rb.add(o, a, r, d, no)
        
        o = no if not d else env.reset()[0]
        global_step += 1
        
        # Обучение
        if rb.full or rb.ptr > batch_size:
            idx = np.random.randint(0, rb.cap if rb.full else rb.ptr, size=batch_size)
            
            om = to_t(rb.obs_map[idx])
            ov = to_t(rb.obs_vec[idx])
            a_t = torch.as_tensor(rb.act[idx], dtype=torch.int64)
            r_t = to_t(rb.rew[idx])
            d_t = to_t(rb.done[idx])
            nom = to_t(rb.next_obs_map[idx])
            nov = to_t(rb.next_obs_vec[idx])
            
            q = net(om, ov).gather(1, a_t[:, None]).squeeze(1)
            
            with torch.no_grad():
                q_next = tgt(nom, nov).max(1)[0]
                target = r_t + gamma * (1.0 - d_t) * q_next
            
            loss = nn.functional.smooth_l1_loss(q, target)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            if global_step % 1000 == 0:
                tgt.load_state_dict(net.state_dict())
    
    torch.save(net.state_dict(), save_path)
    print(f"Модель сохранена в {save_path}")


if __name__ == "__main__":
    train_dqn()

