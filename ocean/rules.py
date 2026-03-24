"""Правила тика клеточного автомата."""
import numpy as np
from .grid import Grid, EMPTY, PREY, PRED, FOOD


NEI = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)], dtype=np.int8)


def torus_idx(x, n):
    """Векторизованный модуль для тора."""
    return (x + n) % n


def shuffle_indices(H, W, rng):
    """Перемешивание индексов для случайного порядка обхода."""
    idx = np.arange(H * W)
    rng.shuffle(idx)
    return np.unravel_index(idx, (H, W))


def step(grid: Grid, cfg):
    """Один тик симуляции с двойным буфером."""
    H, W = grid.state.shape
    st, age, en, rng = grid.state, grid.age, grid.energy, grid.rng
    
    # Двойной буфер
    st_next = st.copy()
    age_next = age.copy()
    en_next = en.copy()
    
    # 1) Жертвы ходят первыми
    rows, cols = shuffle_indices(H, W, rng)
    
    for r, c in zip(rows, cols):
        if st[r, c] != PREY:
            continue
        
        # Соседи
        nei_r = np.array([r, r, r + 1, r - 1])
        nei_c = np.array([c + 1, c - 1, c, c])
        rr = torus_idx(nei_r, H)
        cc = torus_idx(nei_c, W)
        
        # Приоритет: сначала ищем траву, потом пустые клетки
        food_mask = (st_next[rr, cc] == FOOD)
        empties = (st_next[rr, cc] == EMPTY)
        moved = False
        
        if food_mask.any():
            # Есть трава - идем к ней и едим
            k = rng.choice(np.where(food_mask)[0])
            nr, nc = rr[k], cc[k]
            
            # Поедание травы
            st_next[nr, nc] = PREY
            age_next[nr, nc] = age[r, c] + 1
            en_next[nr, nc] = en[r, c] + cfg.fish_eat_energy - cfg.fish_move_cost
            
            st_next[r, c] = EMPTY
            age_next[r, c] = 0
            en_next[r, c] = 0
            moved = True
        elif empties.any():
            # Нет травы - идем в пустую клетку
            k = rng.choice(np.where(empties)[0])
            nr, nc = rr[k], cc[k]
            
            st_next[nr, nc] = PREY
            age_next[nr, nc] = age[r, c] + 1
            en_next[nr, nc] = en[r, c] - cfg.fish_move_cost
            
            st_next[r, c] = EMPTY
            age_next[r, c] = 0
            en_next[r, c] = 0
            moved = True
        
        if not moved:
            # Остается на месте
            age_next[r, c] = min(age[r, c] + 1, cfg.max_age)
            en_next[r, c] = en[r, c] - cfg.fish_move_cost
    
    # 2) Смерть жертв от голода и размножение
    prey_pos = np.where(st_next == PREY)
    for r, c in zip(*prey_pos):
        # Смерть от голода
        if en_next[r, c] <= 0:
            st_next[r, c] = EMPTY
            age_next[r, c] = 0
            en_next[r, c] = 0
            continue
        
        # Размножение жертв
        if age_next[r, c] >= cfg.fish_breed_age and en_next[r, c] >= cfg.fish_eat_energy * 2:
            # Попытка разместить детеныша в случайной пустой соседней клетке
            nei_r = np.array([r, r, r + 1, r - 1])
            nei_c = np.array([c + 1, c - 1, c, c])
            rr = torus_idx(nei_r, H)
            cc = torus_idx(nei_c, W)
            
            empties = (st_next[rr, cc] == EMPTY)
            
            if empties.any():
                k = rng.choice(np.where(empties)[0])
                cr, cc2 = rr[k], cc[k]
                
                st_next[cr, cc2] = PREY
                age_next[cr, cc2] = 0
                en_next[cr, cc2] = cfg.fish_start_energy
                en_next[r, c] -= cfg.fish_eat_energy  # Трата энергии на размножение
                age_next[r, c] = 0  # Сброс возраста
    
    # 3) Хищники
    rows, cols = shuffle_indices(H, W, rng)
    
    for r, c in zip(rows, cols):
        if st[r, c] != PRED:
            continue
        
        # Соседи
        nei_r = np.array([r, r, r + 1, r - 1])
        nei_c = np.array([c + 1, c - 1, c, c])
        rr = torus_idx(nei_r, H)
        cc = torus_idx(nei_c, W)
        
        prey_mask = (st_next[rr, cc] == PREY)
        moved = False
        
        if prey_mask.any():
            # Есть жертва - идем к ней
            k = rng.choice(np.where(prey_mask)[0])
            nr, nc = rr[k], cc[k]
            
            # Поедание
            st_next[nr, nc] = PRED
            age_next[nr, nc] = age[r, c] + 1
            en_next[nr, nc] = en[r, c] + cfg.eat_energy - cfg.move_cost
            
            st_next[r, c] = EMPTY
            age_next[r, c] = 0
            en_next[r, c] = 0
            moved = True
        else:
            # Нет жертвы - идем в пустую клетку или на траву
            empties = (st_next[rr, cc] == EMPTY) | (st_next[rr, cc] == FOOD)
            
            if empties.any():
                k = rng.choice(np.where(empties)[0])
                nr, nc = rr[k], cc[k]
                
                st_next[nr, nc] = PRED
                age_next[nr, nc] = age[r, c] + 1
                en_next[nr, nc] = en[r, c] - cfg.move_cost
                
                st_next[r, c] = EMPTY
                age_next[r, c] = 0
                en_next[r, c] = 0
                moved = True
        
        if not moved:
            # Остается на месте
            age_next[r, c] = age[r, c] + 1
            en_next[r, c] = en[r, c] - cfg.move_cost
    
    # 4) Размножение и смерть хищников
    pred_pos = np.where(st_next == PRED)
    
    for r, c in zip(*pred_pos):
        # Смерть от голода
        if en_next[r, c] <= 0:
            st_next[r, c] = EMPTY
            age_next[r, c] = 0
            en_next[r, c] = 0
            continue
        
        # Размножение
        if age_next[r, c] >= cfg.shark_breed_age and en_next[r, c] >= cfg.breed_energy:
            # Попытка разместить детеныша в случайной пустой соседней клетке
            nei_r = np.array([r, r, r + 1, r - 1])
            nei_c = np.array([c + 1, c - 1, c, c])
            rr = torus_idx(nei_r, H)
            cc = torus_idx(nei_c, W)
            
            empties = (st_next[rr, cc] == EMPTY) | (st_next[rr, cc] == FOOD)
            
            if empties.any():
                k = rng.choice(np.where(empties)[0])
                cr, cc2 = rr[k], cc[k]
                
                st_next[cr, cc2] = PRED
                age_next[cr, cc2] = 0
                en_next[cr, cc2] = cfg.child_energy
                en_next[r, c] -= cfg.child_energy
            
            # Сброс возраста для избежания мгновенного цикла
            age_next[r, c] = 0
    
    # 5) Рост травы на пустых клетках
    empty_pos = np.where(st_next == EMPTY)
    if len(empty_pos[0]) > 0:
        growth_mask = rng.random(len(empty_pos[0])) < cfg.food_growth_rate
        if growth_mask.any():
            grow_indices = np.where(growth_mask)[0]
            for idx in grow_indices:
                r, c = empty_pos[0][idx], empty_pos[1][idx]
                st_next[r, c] = FOOD
                en_next[r, c] = 1
    
    # Применение изменений
    grid.state, grid.age, grid.energy = st_next, age_next, en_next
