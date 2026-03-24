"""Конфигурация симуляции."""
from pydantic import BaseModel


class SimConfig(BaseModel):
    """Конфигурация симуляции Wa-Tor."""
    
    width: int = 64
    height: int = 64
    fish_init_density: float = 0.2
    shark_init_density: float = 0.01  # Уменьшено с 0.05 до 0.01 (1% вместо 5%)
    food_init_density: float = 0.3  # Начальная плотность травы
    food_growth_rate: float = 0.02  # Вероятность роста травы на пустой клетке за тик
    fish_breed_age: int = 3
    shark_breed_age: int = 8  # Увеличено с 6 до 8 для усложнения размножения
    fish_eat_energy: int = 2  # Энергия, получаемая жертвой при поедании травы
    fish_move_cost: int = 1  # Стоимость движения для жертв
    fish_start_energy: int = 3  # Начальная энергия жертв
    eat_energy: int = 5
    move_cost: int = 1
    breed_energy: int = 10  # Увеличено с 8 до 10 для усложнения размножения
    child_energy: int = 4
    max_age: int = 128
    max_steps: int = 10_000
    patch_size: int = 11
    seed: int = 42

