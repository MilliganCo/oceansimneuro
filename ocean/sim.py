"""Симулятор клеточного автомата."""
import time
from .grid import Grid
from .rules import step
from .stats import counts


def run_sim(cfg, steps, hook=None, delay=0.0):
    """Запуск симуляции на заданное количество шагов.
    
    Args:
        cfg: SimConfig конфигурация
        steps: количество шагов
        hook: опциональная функция callback(t, grid, counts)
        delay: задержка между шагами в секундах (для визуализации)
    
    Returns:
        (grid, hist): финальная сетка и история популяций
    """
    g = Grid.init(
        cfg.height,
        cfg.width,
        cfg.fish_init_density,
        cfg.shark_init_density,
        cfg.food_init_density,
        cfg.seed
    )
    
    hist = []
    
    for t in range(steps):
        step(g, cfg)
        c = counts(g.state)
        hist.append(c)
        
        if hook:
            hook(t, g, c)
        
        if delay > 0:
            time.sleep(delay)
    
    return g, hist
