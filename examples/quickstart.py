"""Быстрый старт с симуляцией."""
from ocean.config import SimConfig
from ocean.sim import run_sim
from ocean.plotting import plot_counts, RealtimeVisualizer
import matplotlib.pyplot as plt


def example_basic():
    """Базовый пример без GUI."""
    cfg = SimConfig(width=64, height=64, seed=7)
    _, hist = run_sim(cfg, 3000)
    plot_counts(hist)


def example_gui():
    """Пример с графическим интерфейсом в реальном времени."""
    cfg = SimConfig(width=64, height=64, seed=7)
    
    visualizer = RealtimeVisualizer(cfg)
    visualizer.show()
    
    def update_hook(t, grid, counts_dict):
        visualizer.update(t, grid, counts_dict)
        plt.pause(0.001)
    
    try:
        _, hist = run_sim(cfg, 3000, hook=update_hook, delay=0.01)
        print("\nСимуляция завершена. Нажмите Enter для закрытия...")
        input()
    finally:
        visualizer.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "gui":
        example_gui()
    else:
        example_basic()
