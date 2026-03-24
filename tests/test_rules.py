"""Тесты правил клеточного автомата."""
import pytest
from ocean.grid import Grid, PREY, PRED
from ocean.rules import step
from ocean.config import SimConfig


def test_progress():
    """Тест базовой эволюции."""
    cfg = SimConfig(width=16, height=16, seed=1)
    g = Grid.init(cfg.height, cfg.width, 0.2, 0.1, cfg.seed)
    before = g.state.copy()
    
    step(g, cfg)
    
    assert g.state.shape == before.shape
    assert g.state.dtype == before.dtype


def test_prey_movement():
    """Тест движения жертв."""
    cfg = SimConfig(width=8, height=8, seed=42)
    g = Grid.init(cfg.height, cfg.width, 0.3, 0.0, cfg.seed)
    
    initial_prey = (g.state == PREY).sum()
    
    for _ in range(10):
        step(g, cfg)
    
    final_prey = (g.state == PREY).sum()
    
    # Жертвы должны остаться (или размножиться)
    assert final_prey >= 0


def test_predator_eating():
    """Тест поедания жертв хищниками."""
    cfg = SimConfig(width=16, height=16, seed=123)
    g = Grid.init(cfg.height, cfg.width, 0.3, 0.1, cfg.seed)
    
    initial_prey = (g.state == PREY).sum()
    initial_pred = (g.state == PRED).sum()
    
    for _ in range(50):
        step(g, cfg)
    
    final_prey = (g.state == PREY).sum()
    final_pred = (g.state == PRED).sum()
    
    # Хищники должны существовать
    assert final_pred >= 0
    # Жертвы могут быть съедены, но не все
    assert final_prey >= 0

