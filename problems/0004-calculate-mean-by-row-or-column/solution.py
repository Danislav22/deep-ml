import numpy as np


def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if not mode:
        return AttributeError('mode может быть column/row')
    
    return np.mean(matrix, axis=-1) if mode == 'row' else np.mean(matrix, axis=-2)