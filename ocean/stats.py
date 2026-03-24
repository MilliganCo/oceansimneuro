"""Подсчет статистики популяций."""
import numpy as np
from .grid import EMPTY, PREY, PRED, FOOD


def counts(state: np.ndarray):
    """Подсчет количества пустых клеток, жертв, хищников и травы."""
    return {
        "empty": int((state == EMPTY).sum()),
        "prey": int((state == PREY).sum()),
        "pred": int((state == PRED).sum()),
        "food": int((state == FOOD).sum()),
    }

