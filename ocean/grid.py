"""Модель данных сетки."""
from dataclasses import dataclass
import numpy as np


EMPTY, PREY, PRED, FOOD = 0, 1, 2, 3


@dataclass
class Grid:
    """Состояние сетки клеточного автомата."""
    
    state: np.ndarray      # (H, W) uint8
    age: np.ndarray        # (H, W) uint16
    energy: np.ndarray     # (H, W) int16
    rng: np.random.Generator
    
    @classmethod
    def init(cls, H, W, fish_p, shark_p, food_p, seed):
        """Инициализация сетки с случайным размещением жертв, хищников и травы."""
        rng = np.random.default_rng(seed)
        state = np.zeros((H, W), dtype=np.uint8)
        mask = rng.random((H, W))
        
        # Размещение жертв
        state[mask < fish_p] = PREY
        # Размещение хищников
        state[(mask >= fish_p) & (mask < fish_p + shark_p)] = PRED
        # Размещение травы на оставшихся пустых клетках
        empty_mask = (state == EMPTY)
        food_mask = empty_mask & (rng.random((H, W)) < food_p)
        state[food_mask] = FOOD
        
        age = np.zeros_like(state, dtype=np.uint16)
        energy = np.zeros_like(state, dtype=np.int16)
        energy[state == PRED] = 5
        energy[state == PREY] = 3  # Начальная энергия жертв
        energy[state == FOOD] = 1  # Энергия травы (для визуализации/отслеживания)
        
        return cls(state, age, energy, rng)

