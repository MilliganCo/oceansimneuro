"""Буфер воспроизведения для RL."""
import numpy as np


class ReplayBuffer:
    """Буфер воспроизведения для хранения переходов (s, a, r, s', done)."""
    
    def __init__(self, cap, obs_map_shape, obs_vec_shape):
        """Инициализация буфера.
        
        Args:
            cap: максимальная емкость буфера
            obs_map_shape: форма карты наблюдений
            obs_vec_shape: форма вектора наблюдений
        """
        self.cap = cap
        self.ptr = 0
        self.full = False
        
        self.obs_map = np.zeros((cap,) + obs_map_shape, dtype=np.float32)
        self.obs_vec = np.zeros((cap,) + obs_vec_shape, dtype=np.float32)
        self.act = np.zeros((cap,), dtype=np.int64)
        self.rew = np.zeros((cap,), dtype=np.float32)
        self.done = np.zeros((cap,), dtype=np.float32)
        self.next_obs_map = np.zeros((cap,) + obs_map_shape, dtype=np.float32)
        self.next_obs_vec = np.zeros((cap,) + obs_vec_shape, dtype=np.float32)
    
    def add(self, o, a, r, d, no):
        """Добавление перехода в буфер.
        
        Args:
            o: текущее наблюдение (dict с obs_map и obs_vec)
            a: действие
            r: награда
            d: флаг завершения эпизода
            no: следующее наблюдение
        """
        i = self.ptr
        
        self.obs_map[i] = o["obs_map"]
        self.obs_vec[i] = o["obs_vec"]
        self.act[i] = a
        self.rew[i] = r
        self.done[i] = float(d)
        self.next_obs_map[i] = no["obs_map"]
        self.next_obs_vec[i] = no["obs_vec"]
        
        self.ptr = (self.ptr + 1) % self.cap
        if self.ptr == 0:
            self.full = True
    
    def size(self):
        """Текущий размер буфера."""
        return self.cap if self.full else self.ptr

