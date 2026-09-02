"""Extraccion de features tabulares a partir de las senales crudas de REHAB."""

import numpy as np

CANALES = [
    "IMU1_Pitch", "IMU1_Yaw", "IMU1_Roll",
    "IMU2_Pitch", "IMU2_Yaw", "IMU2_Roll",
    "Glove_Thumb", "Glove_Index", "Glove_Middle", "Glove_Ring", "Glove_Pinky",
    "Glove_Pitch",
]

ESTADISTICOS = ["media", "std", "min", "max", "rms"]


def extraer_features_por_muestra(X):
    """(N, 880, 12) -> (N, 60): media/std/min/max/rms por canal."""
    n_muestras, _, n_canales = X.shape
    n_stats = len(ESTADISTICOS)
    features = np.zeros((n_muestras, n_canales * n_stats))

    for c in range(n_canales):
        canal = X[:, :, c]
        media = canal.mean(axis=1)
        std = canal.std(axis=1)
        minimo = canal.min(axis=1)
        maximo = canal.max(axis=1)
        rms = np.sqrt((canal ** 2).mean(axis=1))
        features[:, c * n_stats:(c + 1) * n_stats] = np.stack(
            [media, std, minimo, maximo, rms], axis=1
        )

    return features


def nombres_columnas():
    return [f"{canal}_{stat}" for canal in CANALES for stat in ESTADISTICOS]
