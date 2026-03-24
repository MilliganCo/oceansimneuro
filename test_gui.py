"""Тестовый скрипт для проверки GUI визуализации."""
from ocean.config import SimConfig
from ocean.sim import run_sim
from ocean.plotting import RealtimeVisualizer
import matplotlib.pyplot as plt


if __name__ == "__main__":
    print("Запуск тестовой симуляции с GUI...")
    print("Нажмите Ctrl+C для остановки")
    
    cfg = SimConfig(width=32, height=32, seed=42)
    
    visualizer = RealtimeVisualizer(cfg)
    visualizer.show()
    
    def update_hook(t, grid, counts_dict):
        visualizer.update(t, grid, counts_dict)
        plt.pause(0.001)
    
    try:
        _, hist = run_sim(cfg, 1000, hook=update_hook, delay=0.05)
        print(f"\nСимуляция завершена! Всего шагов: {len(hist)}")
        print("Нажмите Enter для закрытия окна...")
        input()
    except KeyboardInterrupt:
        print("\nСимуляция прервана пользователем.")
    finally:
        visualizer.close()
        print("Окно закрыто.")

