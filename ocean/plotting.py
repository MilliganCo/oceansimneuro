"""Визуализация графиков популяций."""
import matplotlib.pyplot as plt
import numpy as np

from .grid import EMPTY, PREY, PRED, FOOD


def plot_counts(hist):
    """Построение графика динамики популяций.
    
    Args:
        hist: список словарей с ключами empty, prey, pred, food
    """
    t = list(range(len(hist)))
    prey = [h["prey"] for h in hist]
    pred = [h["pred"] for h in hist]
    food = [h.get("food", 0) for h in hist]
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, prey, label="Жертвы (prey)", color="blue")
    plt.plot(t, pred, label="Хищники (pred)", color="red")
    plt.plot(t, food, label="Трава (food)", color="green")
    plt.legend()
    plt.xlabel("Шаг")
    plt.ylabel("Количество")
    plt.title("Динамика популяций")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def render_grid(state, ax=None, cmap=None):
    """Визуализация сетки клеточного автомата.
    
    Args:
        state: массив состояния (H, W) с значениями EMPTY, PREY, PRED
        ax: ось matplotlib для отображения (если None, создается новая)
        cmap: цветовая карта (по умолчанию: синий для жертв, красный для хищников)
    
    Returns:
        ax: ось matplotlib
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Создание цветной карты: пусто=черный, жертвы=синий, хищники=красный, трава=зеленый
    if cmap is None:
        colored = np.zeros((*state.shape, 3), dtype=np.float32)
        colored[state == PREY] = [0.2, 0.4, 1.0]  # Синий для жертв
        colored[state == PRED] = [1.0, 0.2, 0.2]  # Красный для хищников
        colored[state == FOOD] = [0.2, 0.8, 0.2]  # Зеленый для травы
        # Пустые клетки остаются черными [0, 0, 0]
    else:
        colored = cmap(state)
    
    ax.clear()
    ax.imshow(colored, interpolation='nearest', origin='upper')
    ax.set_title("Симуляция Wa-Tor")
    ax.set_xlabel("Ширина")
    ax.set_ylabel("Высота")
    ax.set_aspect('equal')
    
    return ax


class RealtimeVisualizer:
    """Визуализатор для отображения симуляции в реальном времени."""
    
    def __init__(self, cfg, update_interval=50):
        """Инициализация визуализатора.
        
        Args:
            cfg: SimConfig конфигурация
            update_interval: интервал обновления в миллисекундах (не используется, оставлен для совместимости)
        """
        self.cfg = cfg
        self.update_interval = update_interval
        
        # Включение интерактивного режима matplotlib
        plt.ion()
        
        # Создание фигуры с двумя подграфиками
        self.fig = plt.figure(figsize=(16, 8))
        self.fig.suptitle("OceanSim - Симуляция в реальном времени", fontsize=14)
        
        # Левая панель: визуализация поля
        self.ax_grid = self.fig.add_subplot(121)
        self.ax_grid.set_title("Поле симуляции")
        
        # Правая панель: графики популяций
        self.ax_plot = self.fig.add_subplot(122)
        self.ax_plot.set_title("Динамика популяций")
        self.ax_plot.set_xlabel("Шаг")
        self.ax_plot.set_ylabel("Количество")
        self.ax_plot.grid(True, alpha=0.3)
        self.ax_plot.legend(['Жертвы', 'Хищники'], loc='upper right')
        
        # История для графиков
        self.history = {
            'prey': [],
            'pred': [],
            'empty': [],
            'food': [],
            'steps': []
        }
        
        # Текстовые метки для статистики
        self.stats_text = self.fig.text(0.5, 0.02, '', ha='center', fontsize=10)
        
        self.initialized = False
    
    def update(self, step_num, grid, counts_dict):
        """Обновление визуализации.
        
        Args:
            step_num: номер шага
            grid: объект Grid
            counts_dict: словарь с ключами empty, prey, pred
        """
        # Обновление истории
        self.history['steps'].append(step_num)
        self.history['prey'].append(counts_dict['prey'])
        self.history['pred'].append(counts_dict['pred'])
        self.history['empty'].append(counts_dict['empty'])
        self.history['food'].append(counts_dict.get('food', 0))
        
        # Ограничение истории для производительности (последние 1000 точек)
        max_history = 1000
        if len(self.history['steps']) > max_history:
            for key in self.history:
                self.history[key] = self.history[key][-max_history:]
        
        # Обновление визуализации поля
        render_grid(grid.state, ax=self.ax_grid)
        
        # Обновление графиков
        self.ax_plot.clear()
        self.ax_plot.plot(
            self.history['steps'],
            self.history['prey'],
            label="Жертвы (prey)",
            color="blue",
            linewidth=1.5
        )
        self.ax_plot.plot(
            self.history['steps'],
            self.history['pred'],
            label="Хищники (pred)",
            color="red",
            linewidth=1.5
        )
        self.ax_plot.plot(
            self.history['steps'],
            self.history['food'],
            label="Трава (food)",
            color="green",
            linewidth=1.5
        )
        self.ax_plot.set_title("Динамика популяций")
        self.ax_plot.set_xlabel("Шаг")
        self.ax_plot.set_ylabel("Количество")
        self.ax_plot.grid(True, alpha=0.3)
        self.ax_plot.legend(loc='upper right')
        
        # Автоматическое масштабирование осей
        if len(self.history['steps']) > 0:
            max_count = max(
                max(self.history['prey']),
                max(self.history['pred']),
                max(self.history['food'])
            )
            if max_count > 0:
                self.ax_plot.set_ylim(0, max_count * 1.1)
            self.ax_plot.set_xlim(
                max(0, self.history['steps'][0] if self.history['steps'] else 0),
                self.history['steps'][-1] + 1 if self.history['steps'] else 1
            )
        
        # Обновление статистики
        total = counts_dict['prey'] + counts_dict['pred'] + counts_dict['empty'] + counts_dict.get('food', 0)
        if total > 0:
            stats_str = (
                f"Шаг: {step_num} | "
                f"Жертвы: {counts_dict['prey']} ({counts_dict['prey']/total*100:.1f}%) | "
                f"Хищники: {counts_dict['pred']} ({counts_dict['pred']/total*100:.1f}%) | "
                f"Трава: {counts_dict.get('food', 0)} ({counts_dict.get('food', 0)/total*100:.1f}%) | "
                f"Пусто: {counts_dict['empty']} ({counts_dict['empty']/total*100:.1f}%)"
            )
        else:
            stats_str = f"Шаг: {step_num} | Все клетки пусты"
        self.stats_text.set_text(stats_str)
        
        # Принудительное обновление
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        self.initialized = True
    
    def show(self):
        """Показать окно визуализации."""
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)  # Небольшая пауза для инициализации
    
    def close(self):
        """Закрыть окно визуализации."""
        plt.close(self.fig)
        plt.ioff()  # Отключение интерактивного режима
