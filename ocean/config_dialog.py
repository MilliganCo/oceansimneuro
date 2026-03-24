"""GUI диалог для настройки параметров симуляции."""
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional
from .config import SimConfig


class ConfigDialog:
    """Диалоговое окно для настройки параметров симуляции."""
    
    def __init__(self, default_config: Optional[SimConfig] = None):
        """Инициализация диалога.
        
        Args:
            default_config: конфигурация по умолчанию
        """
        if default_config is None:
            default_config = SimConfig()
        
        self.config = None
        self.result = None
        
        # Создание окна
        self.root = tk.Tk()
        self.root.title("Настройки симуляции OceanSim")
        self.root.resizable(False, False)
        
        # Центрирование окна
        self.root.update_idletasks()
        width = 400
        height = 600  # Увеличено для новых полей
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
        # Переменные для значений
        self.steps_var = tk.IntVar(value=2000)
        self.width_var = tk.IntVar(value=default_config.width)
        self.height_var = tk.IntVar(value=default_config.height)
        self.fish_density_var = tk.DoubleVar(value=default_config.fish_init_density)
        self.shark_density_var = tk.DoubleVar(value=default_config.shark_init_density)
        self.food_density_var = tk.DoubleVar(value=default_config.food_init_density)
        self.food_growth_var = tk.DoubleVar(value=default_config.food_growth_rate)
        self.delay_var = tk.DoubleVar(value=0.01)
        self.seed_var = tk.IntVar(value=default_config.seed)
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Создание виджетов интерфейса."""
        # Заголовок
        title_label = ttk.Label(
            self.root,
            text="Параметры симуляции",
            font=("Arial", 14, "bold")
        )
        title_label.pack(pady=10)
        
        # Фрейм для полей ввода
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Количество шагов
        ttk.Label(main_frame, text="Количество шагов:").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        steps_entry = ttk.Entry(main_frame, textvariable=self.steps_var, width=20)
        steps_entry.grid(row=0, column=1, pady=5, padx=10)
        
        # Размер поля
        ttk.Label(main_frame, text="Ширина поля:").grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        width_entry = ttk.Entry(main_frame, textvariable=self.width_var, width=20)
        width_entry.grid(row=1, column=1, pady=5, padx=10)
        
        ttk.Label(main_frame, text="Высота поля:").grid(
            row=2, column=0, sticky=tk.W, pady=5
        )
        height_entry = ttk.Entry(main_frame, textvariable=self.height_var, width=20)
        height_entry.grid(row=2, column=1, pady=5, padx=10)
        
        # Плотность жертв
        ttk.Label(main_frame, text="Плотность жертв (0.0-1.0):").grid(
            row=3, column=0, sticky=tk.W, pady=5
        )
        fish_entry = ttk.Entry(main_frame, textvariable=self.fish_density_var, width=20)
        fish_entry.grid(row=3, column=1, pady=5, padx=10)
        fish_info = ttk.Label(
            main_frame,
            text=f"({self.fish_density_var.get()*100:.1f}% поля)",
            font=("Arial", 8)
        )
        fish_info.grid(row=3, column=2, sticky=tk.W, padx=5)
        
        # Плотность хищников
        ttk.Label(main_frame, text="Плотность хищников (0.0-1.0):").grid(
            row=4, column=0, sticky=tk.W, pady=5
        )
        shark_entry = ttk.Entry(main_frame, textvariable=self.shark_density_var, width=20)
        shark_entry.grid(row=4, column=1, pady=5, padx=10)
        shark_info = ttk.Label(
            main_frame,
            text=f"({self.shark_density_var.get()*100:.1f}% поля)",
            font=("Arial", 8)
        )
        shark_info.grid(row=4, column=2, sticky=tk.W, padx=5)
        
        # Плотность травы
        ttk.Label(main_frame, text="Плотность травы (0.0-1.0):").grid(
            row=5, column=0, sticky=tk.W, pady=5
        )
        food_entry = ttk.Entry(main_frame, textvariable=self.food_density_var, width=20)
        food_entry.grid(row=5, column=1, pady=5, padx=10)
        food_info = ttk.Label(
            main_frame,
            text=f"({self.food_density_var.get()*100:.1f}% поля)",
            font=("Arial", 8)
        )
        food_info.grid(row=5, column=2, sticky=tk.W, padx=5)
        
        # Скорость роста травы
        ttk.Label(main_frame, text="Скорость роста травы (0.0-1.0):").grid(
            row=6, column=0, sticky=tk.W, pady=5
        )
        growth_entry = ttk.Entry(main_frame, textvariable=self.food_growth_var, width=20)
        growth_entry.grid(row=6, column=1, pady=5, padx=10)
        
        # Задержка между шагами
        ttk.Label(main_frame, text="Задержка (сек):").grid(
            row=7, column=0, sticky=tk.W, pady=5
        )
        delay_entry = ttk.Entry(main_frame, textvariable=self.delay_var, width=20)
        delay_entry.grid(row=7, column=1, pady=5, padx=10)
        
        # Seed
        ttk.Label(main_frame, text="Seed (случайность):").grid(
            row=8, column=0, sticky=tk.W, pady=5
        )
        seed_entry = ttk.Entry(main_frame, textvariable=self.seed_var, width=20)
        seed_entry.grid(row=8, column=1, pady=5, padx=10)
        
        # Обновление процентов при изменении плотности
        def update_percentages(*args):
            fish_info.config(text=f"({self.fish_density_var.get()*100:.1f}% поля)")
            shark_info.config(text=f"({self.shark_density_var.get()*100:.1f}% поля)")
            food_info.config(text=f"({self.food_density_var.get()*100:.1f}% поля)")
        
        self.fish_density_var.trace_add("write", update_percentages)
        self.shark_density_var.trace_add("write", update_percentages)
        self.food_density_var.trace_add("write", update_percentages)
        
        # Кнопки
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=9, column=0, columnspan=3, pady=20)
        
        ttk.Button(
            button_frame,
            text="Запустить",
            command=self._on_ok,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Отмена",
            command=self._on_cancel,
            width=15
        ).pack(side=tk.LEFT, padx=5)
    
    def _on_ok(self):
        """Обработка нажатия кнопки OK."""
        try:
            # Валидация значений
            steps = self.steps_var.get()
            width = self.width_var.get()
            height = self.height_var.get()
            fish_density = self.fish_density_var.get()
            shark_density = self.shark_density_var.get()
            food_density = self.food_density_var.get()
            food_growth = self.food_growth_var.get()
            delay = self.delay_var.get()
            seed = self.seed_var.get()
            
            if steps <= 0:
                raise ValueError("Количество шагов должно быть больше 0")
            if width <= 0 or height <= 0:
                raise ValueError("Размеры поля должны быть больше 0")
            if not (0.0 <= fish_density <= 1.0):
                raise ValueError("Плотность жертв должна быть от 0.0 до 1.0")
            if not (0.0 <= shark_density <= 1.0):
                raise ValueError("Плотность хищников должна быть от 0.0 до 1.0")
            if not (0.0 <= food_density <= 1.0):
                raise ValueError("Плотность травы должна быть от 0.0 до 1.0")
            if not (0.0 <= food_growth <= 1.0):
                raise ValueError("Скорость роста травы должна быть от 0.0 до 1.0")
            if fish_density + shark_density > 1.0:
                raise ValueError("Сумма плотностей жертв и хищников не должна превышать 1.0")
            if delay < 0:
                raise ValueError("Задержка не может быть отрицательной")
            
            # Создание конфигурации
            self.config = SimConfig(
                width=width,
                height=height,
                fish_init_density=fish_density,
                shark_init_density=shark_density,
                food_init_density=food_density,
                food_growth_rate=food_growth,
                seed=seed
            )
            
            self.result = {
                'config': self.config,
                'steps': steps,
                'delay': delay
            }
            
            self.root.destroy()
        except ValueError as e:
            messagebox.showerror("Ошибка", str(e))
        except Exception as e:
            messagebox.showerror("Ошибка", f"Неверный формат данных: {e}")
    
    def _on_cancel(self):
        """Обработка нажатия кнопки Отмена."""
        self.result = None
        self.root.destroy()
    
    def show(self) -> Optional[dict]:
        """Показать диалог и вернуть результат.
        
        Returns:
            dict с ключами 'config', 'steps', 'delay' или None если отменено
        """
        self.root.mainloop()
        return self.result


def show_config_dialog(default_config: Optional[SimConfig] = None) -> Optional[dict]:
    """Показать диалог настройки параметров.
    
    Args:
        default_config: конфигурация по умолчанию
    
    Returns:
        dict с ключами 'config', 'steps', 'delay' или None если отменено
    """
    dialog = ConfigDialog(default_config)
    return dialog.show()

