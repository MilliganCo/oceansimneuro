# OceanSim

Клеточный автомат хищник-жертва на решетке с встроенным рендером графиков популяций и Gym-совместимой средой для обучения агента управлять одним хищником.

## Описание

OceanSim реализует классическую модель Wa-Tor - клеточный автомат, моделирующий экосистему хищников и жертв. Проект включает:

- **Симуляцию** клеточного автомата с двойным буфером
- **Визуализацию** динамики популяций
- **RL-среду** для обучения агента управлять хищником
- **Экспорт** эпизодов и траекторий в датасет

## Установка

```bash
# Создание виртуального окружения
python -m venv .venv

# Активация (Windows)
.venv\Scripts\activate

# Активация (Linux/Mac)
source .venv/bin/activate

# Установка зависимостей
pip install -U pip
pip install -e .
```

## Быстрый старт

### Чистая симуляция

```bash
ocean sim --steps 2000
```

### Симуляция с графическим интерфейсом в реальном времени

```bash
# С диалогом выбора параметров (по умолчанию)
ocean sim --gui

# Без диалога (с параметрами по умолчанию)
ocean sim --gui --no-interactive --steps 2000 --delay 0.01
```

GUI режим показывает:
- **Диалог настроек** (по умолчанию): окно для выбора параметров перед запуском:
  - Количество шагов
  - Размер поля (ширина и высота)
  - Плотность жертв (0.0-1.0)
  - Плотность хищников (0.0-1.0)
  - Задержка между шагами
  - Seed для воспроизводимости
- **Левая панель**: визуализация поля симуляции (синий = жертвы, красный = хищники)
- **Правая панель**: графики популяций, обновляющиеся в реальном времени
- **Статистика**: текущие значения популяций и их проценты

Параметры:
- `--gui`: включить графический интерфейс
- `--interactive`: показать диалог выбора параметров (по умолчанию True для GUI)
- `--no-interactive`: отключить диалог и использовать параметры по умолчанию
- `--delay`: задержка между шагами в секундах (по умолчанию 0.01, используется если диалог отключен)

Или через Python:

```python
from ocean.config import SimConfig
from ocean.sim import run_sim
from ocean.plotting import plot_counts, RealtimeVisualizer
import matplotlib.pyplot as plt

# Базовый пример без GUI
cfg = SimConfig(width=64, height=64, seed=7)
_, hist = run_sim(cfg, 3000)
plot_counts(hist)

# Пример с GUI в реальном времени
cfg = SimConfig(width=64, height=64, seed=7)
visualizer = RealtimeVisualizer(cfg)
visualizer.show()

def update_hook(t, grid, counts_dict):
    visualizer.update(t, grid, counts_dict)
    plt.pause(0.001)

_, hist = run_sim(cfg, 3000, hook=update_hook, delay=0.01)
visualizer.close()

# Пример с диалогом выбора параметров
from ocean.config_dialog import show_config_dialog

result = show_config_dialog()  # Показывает диалог с настройками
if result:
    cfg = result['config']
    steps = result['steps']
    delay = result['delay']
    # Запуск симуляции с выбранными параметрами...
```

### Генерация датасета

```bash
ocean record --steps 50000 --out replay.npz
```

### Обучение DQN агента

```bash
ocean rl-train --steps 50000 --save-path dqn.pt
```

Или напрямую:

```bash
python train_dqn.py
```

### Оценка обученной политики

```bash
ocean rl-eval --model-path dqn.pt --episodes 10
```

Или напрямую:

```bash
python eval_policy.py
```

## Правила мира

Поле `H×W` с тороидальной топологией. Состояния клетки:

- `0` - пусто
- `1` - жертва (fish). Поля: возраст
- `2` - хищник (shark). Поля: возраст, энергия

### Тик симуляции

1. **Жертвы** ходят первыми:
   - Выбирают случайного пустого соседа из 4
   - Возраст +1
   - Если `age ≥ fish_breed_age`: размножение
   - Детеныш остается в старой клетке, возраст сбрасывается

2. **Хищники**:
   - Если есть соседняя жертва: шаг в нее и еда, `energy += eat_energy`
   - Иначе шаг в случайного пустого соседа
   - Возраст +1, `energy -= move_cost`
   - Если `age ≥ shark_breed_age and energy ≥ breed_energy`: размножение
   - Если `energy ≤ 0`: смерть

Коллизии решаются двойным буфером. Порядок обхода клеток каждый тик перемешивается.

## RL-среда

Единый агент управляет одним хищником, остальные сущности действуют по правилам автомата.

### Observation

- Окно `patch_size × patch_size` вокруг агента с оберткой по тору
- Каналы: `prey_mask`, `pred_mask_without_self`, `empty_mask`
- Скалярные фичи: `energy_norm`, `age_norm`, локальные счетчики
- Формат: `dict(obs_map: float[3, P, P], obs_vec: float[4])`

### Action

- `0..4`: stay, up, down, left, right

### Reward

- +1 за поедание жертвы
- −1 за смерть
- +0.01 за выживание хода
- +0.2 за размножение
- Кэп на шаг: `clip(-1, +1.5)`

### Episode end

- Смерть агента
- `step_limit` тиков

## Конфигурация

Параметры настраиваются через `SimConfig`:

```python
from ocean.config import SimConfig

cfg = SimConfig(
    width=64,
    height=64,
    fish_init_density=0.2,
    shark_init_density=0.05,
    fish_breed_age=3,
    shark_breed_age=6,
    eat_energy=5,
    move_cost=1,
    breed_energy=8,
    child_energy=4,
    max_age=128,
    max_steps=10000,
    patch_size=11,
    seed=42
)
```

## Структура проекта

```
ocean-sim/
  pyproject.toml
  README.md
  Makefile
  ocean/
    __init__.py
    config.py
    grid.py
    rules.py
    sim.py
    stats.py
    plotting.py
    env.py
    dataset.py
    cli.py
  train_dqn.py
  eval_policy.py
  examples/
    quickstart.py
  tests/
    test_rules.py
    test_env.py
```

## Зависимости

- Python 3.10+
- numpy
- gymnasium
- matplotlib
- pydantic
- typer
- torch 2.x
- tqdm
- rich (опционально)

## Тестирование

```bash
pytest -q
```

## Лицензия

MIT

