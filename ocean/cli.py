"""CLI интерфейс для симуляции."""
import typer
import numpy as np

from .config import SimConfig
from .sim import run_sim
from .plotting import plot_counts, RealtimeVisualizer
from .env import OceanEnv

app = typer.Typer()


@app.command()
def sim(steps: int = 2000, gui: bool = False, delay: float = 0.01, interactive: bool = True, no_interactive: bool = False):
    """Запуск чистой симуляции с графиками популяций.
    
    Args:
        steps: количество шагов симуляции
        gui: использовать графический интерфейс в реальном времени
        delay: задержка между шагами в секундах (только для GUI режима)
        interactive: показать диалог выбора параметров перед запуском (по умолчанию True для GUI)
        no_interactive: отключить интерактивный диалог (приоритет над interactive)
    """
    cfg = SimConfig()
    
    if gui:
        # Режим с графическим интерфейсом
        import matplotlib.pyplot as plt
        
        # Если включен интерактивный режим и не отключен явно, показываем диалог настроек
        show_dialog = interactive and not no_interactive
        
        if show_dialog:
            try:
                from .config_dialog import show_config_dialog
                result = show_config_dialog(cfg)
                
                if result is None:
                    print("Запуск отменен пользователем.")
                    return
                
                cfg = result['config']
                steps = result['steps']
                delay = result['delay']
                print(f"\nЗапуск симуляции с параметрами:")
                print(f"  Размер поля: {cfg.width}x{cfg.height}")
                print(f"  Плотность жертв: {cfg.fish_init_density*100:.1f}%")
                print(f"  Плотность хищников: {cfg.shark_init_density*100:.1f}%")
                print(f"  Количество шагов: {steps}")
                print(f"  Задержка: {delay} сек")
            except ImportError:
                print("Предупреждение: Tkinter не доступен, используем параметры по умолчанию")
            except Exception as e:
                print(f"Ошибка при открытии диалога настроек: {e}")
                print("Используем параметры по умолчанию")
        
        visualizer = RealtimeVisualizer(cfg)
        visualizer.show()
        
        def update_hook(t, grid, counts_dict):
            visualizer.update(t, grid, counts_dict)
            plt.pause(0.001)  # Небольшая пауза для обработки событий
        
        try:
            _, hist = run_sim(cfg, steps, hook=update_hook, delay=delay)
            print(f"\nСимуляция завершена. Нажмите Enter для закрытия окна...")
            input()
        except KeyboardInterrupt:
            print("\nСимуляция прервана пользователем.")
        finally:
            visualizer.close()
    else:
        # Обычный режим без GUI
        _, hist = run_sim(cfg, steps)
        plot_counts(hist)


@app.command()
def record(steps: int = 10000, out: str = "replay.npz"):
    """Генерация датасета переходов без обучения."""
    env = OceanEnv(SimConfig())
    o, _ = env.reset()
    
    data = {
        "obs_map": [],
        "obs_vec": [],
        "act": [],
        "rew": [],
        "done": []
    }
    
    for _ in range(steps):
        a = np.random.randint(env.action_space.n)
        no, r, d, _, _ = env.step(a)
        
        # Сохранение
        data["obs_map"].append(o["obs_map"])
        data["obs_vec"].append(o["obs_vec"])
        data["act"].append(a)
        data["rew"].append(r)
        data["done"].append(d)
        
        o = no if not d else env.reset()[0]
    
    np.savez_compressed(out, **{k: np.array(v) for k, v in data.items()})
    print(f"Сохранено {out}")


@app.command()
def rl_train(
    steps: int = 50000,
    lr: float = 1e-3,
    batch_size: int = 64,
    buffer_size: int = 100000,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay: int = 20000,
    save_path: str = "dqn.pt"
):
    """Обучение DQN агента."""
    import sys
    from pathlib import Path
    
    # Добавляем корень проекта в путь для импорта train_dqn
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from train_dqn import train_dqn
    
    train_dqn(
        steps=steps,
        lr=lr,
        batch_size=batch_size,
        buffer_size=buffer_size,
        gamma=gamma,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay=eps_decay,
        save_path=save_path
    )


@app.command()
def rl_eval(model_path: str = "dqn.pt", episodes: int = 10):
    """Воспроизведение обученной политики."""
    import sys
    from pathlib import Path
    
    # Добавляем корень проекта в путь для импорта eval_policy
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from eval_policy import eval_policy
    
    eval_policy(model_path=model_path, episodes=episodes)


def main():
    """Точка входа CLI."""
    app()


if __name__ == "__main__":
    main()

