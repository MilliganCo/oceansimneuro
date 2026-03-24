"""Тесты Gymnasium среды."""
import pytest
import numpy as np
from ocean.env import OceanEnv
from ocean.config import SimConfig


def test_env_reset():
    """Тест сброса среды."""
    cfg = SimConfig(width=32, height=32, seed=42)
    env = OceanEnv(cfg)
    
    obs, info = env.reset()
    
    assert "obs_map" in obs
    assert "obs_vec" in obs
    assert obs["obs_map"].shape == (3, cfg.patch_size, cfg.patch_size)
    assert obs["obs_vec"].shape == (4,)


def test_env_step():
    """Тест шага среды."""
    cfg = SimConfig(width=32, height=32, seed=42)
    env = OceanEnv(cfg)
    
    obs, _ = env.reset()
    
    action = env.action_space.sample()
    next_obs, reward, done, truncated, info = env.step(action)
    
    assert "obs_map" in next_obs
    assert "obs_vec" in next_obs
    assert isinstance(reward, (float, np.floating))
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)


def test_env_multiple_steps():
    """Тест нескольких шагов."""
    cfg = SimConfig(width=32, height=32, seed=42)
    env = OceanEnv(cfg)
    
    obs, _ = env.reset()
    
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        if done:
            obs, _ = env.reset()
        
        assert -1.0 <= reward <= 1.5  # Проверка ограничения награды

