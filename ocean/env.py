"""Gymnasium-совместимая среда для RL."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .grid import Grid, EMPTY, PREY, PRED, FOOD
from .rules import step as world_step


ACTIONS = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]  # stay, up, down, right, left


class OceanEnv(gym.Env):
    """Среда для обучения агента управлять одним хищником."""
    
    metadata = {"render_modes": []}
    
    def __init__(self, cfg):
        """Инициализация среды.
        
        Args:
            cfg: SimConfig конфигурация
        """
        self.cfg = cfg
        P = cfg.patch_size
        
        self.observation_space = spaces.Dict({
            "obs_map": spaces.Box(0, 1, shape=(3, P, P), dtype=np.float32),
            "obs_vec": spaces.Box(-1, 1, shape=(4,), dtype=np.float32),
        })
        
        self.action_space = spaces.Discrete(len(ACTIONS))
        
        self._grid = None
        self._agent_rc = None
        self._step = 0
        self._rng = np.random.default_rng(cfg.seed)
    
    def reset(self, seed=None, options=None):
        """Сброс среды."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        self._grid = Grid.init(
            self.cfg.height,
            self.cfg.width,
            self.cfg.fish_init_density,
            self.cfg.shark_init_density,
            self.cfg.food_init_density,
            int(self._rng.integers(1e9))
        )
        
        # Размещение агента на хищнике или создание нового
        pos = np.argwhere(self._grid.state == PRED)
        
        if len(pos) == 0:
            # Нет хищников - создаем одного
            r = int(self._rng.integers(self.cfg.height))
            c = int(self._rng.integers(self.cfg.width))
            self._grid.state[r, c] = PRED
            self._grid.energy[r, c] = 5
            self._grid.age[r, c] = 0
            pos = np.array([[r, c]])
        
        i = int(self._rng.integers(len(pos)))
        self._agent_rc = tuple(pos[i])
        self._step = 0
        
        return self._obs(), {}
    
    def step(self, action):
        """Шаг среды."""
        # Сохраняем позицию и состояние агента до шага
        r, c = self._agent_rc
        H, W = self._grid.state.shape
        st = self._grid.state
        en = self._grid.energy
        
        # Сохраняем состояние агента
        agent_energy_before = en[r, c]
        agent_age_before = self._grid.age[r, c]
        
        # Временно помечаем клетку агента как занятую (чтобы world_step не обрабатывал его)
        # Используем специальное значение или просто исключаем из обработки
        # Вместо этого, мы применим затраты энергии вручную после world_step
        
        # Движение агента
        dr, dc = ACTIONS[int(action)]
        nr = (r + dr) % H
        nc = (c + dc) % W
        
        rew = 0.0
        died = False
        ate_prey = False
        moved = False
        
        # Движение агента (может идти на EMPTY, PREY или FOOD)
        if st[nr, nc] in (EMPTY, PREY, FOOD):
            if st[nr, nc] == PREY:
                rew += 1.0
                agent_energy_before += self.cfg.eat_energy
                ate_prey = True
            
            # Перемещение агента
            st[nr, nc] = PRED
            st[r, c] = EMPTY
            self._grid.age[nr, nc] = agent_age_before + 1
            self._grid.age[r, c] = 0
            en[nr, nc] = agent_energy_before
            en[r, c] = 0
            self._agent_rc = (nr, nc)
            moved = True
        
        # Применяем затраты энергии за движение
        agent_energy_after_move = agent_energy_before
        if moved:
            agent_energy_after_move -= self.cfg.move_cost
            en[self._agent_rc[0], self._agent_rc[1]] = agent_energy_after_move
        else:
            # Остается на месте, но все равно тратит энергию
            agent_energy_after_move -= self.cfg.move_cost
            en[r, c] = agent_energy_after_move
            self._grid.age[r, c] = min(agent_age_before + 1, self.cfg.max_age)
        
        # Временно сохраняем состояние агента и убираем его из сетки
        agent_pos = self._agent_rc
        agent_state_saved = {
            'pos': agent_pos,
            'energy': agent_energy_after_move,
            'age': self._grid.age[agent_pos[0], agent_pos[1]]
        }
        
        # Временно убираем агента из сетки перед world_step
        st[agent_pos[0], agent_pos[1]] = EMPTY
        en[agent_pos[0], agent_pos[1]] = 0
        self._grid.age[agent_pos[0], agent_pos[1]] = 0
        
        # Применяем world_step (агент не обрабатывается)
        world_step(self._grid, self.cfg)
        self._step += 1
        
        # Восстанавливаем агента после world_step
        ar, ac = agent_state_saved['pos']
        self._grid.state[ar, ac] = PRED
        self._grid.energy[ar, ac] = agent_state_saved['energy']
        self._grid.age[ar, ac] = agent_state_saved['age']
        self._agent_rc = (ar, ac)
        
        # Проверка выживания агента
        ar, ac = self._agent_rc
        if self._grid.state[ar, ac] != PRED or self._grid.energy[ar, ac] <= 0:
            died = True
            rew -= 1.0
        
        # Награда за выживание
        if not died:
            rew += 0.01
        
        # Ограничение награды
        rew = np.clip(rew, -1.0, 1.5)
        
        done = died or self._step >= self.cfg.max_steps
        obs = self._obs()
        
        return obs, float(rew), done, False, {}
    
    def _obs(self):
        """Получение наблюдения агента."""
        r, c = self._agent_rc
        P = self.cfg.patch_size
        rad = P // 2
        H, W = self._grid.state.shape
        
        # Извлечение патча с оберткой по тору
        rr = (np.arange(r - rad, r + rad + 1) % H)[:, None]
        cc = (np.arange(c - rad, c + rad + 1) % W)[None, :]
        patch = self._grid.state[rr, cc]
        
        # Маски
        prey = (patch == PREY).astype(np.float32)
        pred = (patch == PRED).astype(np.float32)
        empty = (patch == EMPTY).astype(np.float32)
        
        # Удаление самого агента из канала хищников
        pred[rad, rad] = 0.0
        
        # Нормализация скалярных признаков
        energy = float(self._grid.energy[r, c]) / max(1.0, float(self.cfg.breed_energy))
        age = float(self._grid.age[r, c]) / max(1.0, float(self.cfg.shark_breed_age))
        
        # Локальные счетчики
        v = np.array([
            energy,
            age,
            prey.sum() / (P * P),
            pred.sum() / (P * P)
        ], dtype=np.float32)
        
        return {
            "obs_map": np.stack([prey, pred, empty], axis=0),
            "obs_vec": v
        }

