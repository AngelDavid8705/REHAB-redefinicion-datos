"""Carga y consolidacion del dataset REHAB - Rehab_exercise."""

from pathlib import Path
import numpy as np

N_MOVEMENTS = 16
SIGNAL_LENGTH = 880
MISSING_FILES = {(14, 1)}  # 014_1.npy corrupto en Science Data Bank

MOVEMENT_NAMES = {
    0: "bobath handshake",
    1: "bobath flexion/extension",
    2: "bobath forward flexion/extension",
    3: "bobath anterior/posterior rotation",
    4: "elbow flexion and wrist compression",
    5: "wrist flexion and extension",
    6: "finger-to-finger training",
    7: "ball gripping",
    8: "shoulder joint internal and external rotation",
    9: "breast expansion",
    10: "flexion-pressure rotation forward and backward",
    11: "elbow joint flexion and touch",
    12: "shoulder touch training",
    13: "ankle extension & knee internal/external rotation",
    14: "knee flexion and extension",
    15: "hip flexion and extension",
}


def _load_pair(base_path: Path, movement_id: int):
    mov_str = f"{movement_id:03d}"
    arr1 = arr2 = None

    f1 = base_path / f"{mov_str}_1.npy"
    if f1.exists() and (movement_id, 1) not in MISSING_FILES:
        arr1 = np.load(f1)

    f2 = base_path / f"{mov_str}_2.npy"
    if f2.exists() and (movement_id, 2) not in MISSING_FILES:
        arr2 = np.load(f2)

    return arr1, arr2


def build_dataset(data_dir: str, incluir_mov14_parcial: bool = False, verbose: bool = True):
    base_path = Path(data_dir)
    X_parts, y_parts = [], []
    meta = {}

    for mov in range(N_MOVEMENTS):
        arr1, arr2 = _load_pair(base_path, mov)
        meta[mov] = {
            "nombre": MOVEMENT_NAMES[mov],
            "sensor_imu_disponible": arr1 is not None,
            "sensor_guante_disponible": arr2 is not None,
        }

        if arr1 is not None and arr2 is not None:
            if arr1.shape[0] != arr2.shape[0]:
                raise ValueError(
                    f"Movimiento {mov}: numero de muestras distinto entre "
                    f"sensor 1 ({arr1.shape[0]}) y sensor 2 ({arr2.shape[0]})"
                )
            X_mov = np.concatenate([arr1, arr2], axis=-1)
            meta[mov]["muestras"] = X_mov.shape[0]
        elif arr2 is not None and incluir_mov14_parcial:
            d = arr2.shape[0]
            relleno = np.full((d, SIGNAL_LENGTH, 6), np.nan)
            X_mov = np.concatenate([relleno, arr2], axis=-1)
            meta[mov]["muestras"] = d
            meta[mov]["advertencia"] = "canales IMU rellenados con NaN (014_1.npy corrupto)"
        else:
            meta[mov]["muestras"] = 0
            if arr1 is None and (mov, 1) in MISSING_FILES:
                meta[mov]["advertencia"] = "excluido: 014_1.npy corrupto"
            continue

        X_parts.append(X_mov)
        y_parts.append(np.full(X_mov.shape[0], mov, dtype=int))

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    if verbose:
        print(f"Dataset consolidado: X={X.shape}, y={y.shape}")
        print(f"Clases presentes: {sorted(set(y.tolist()))}")
        for mov, info in meta.items():
            flag = "" if info["muestras"] > 0 else "  <-- SIN DATOS"
            print(f"  mov {mov:02d} ({info['nombre']}): {info['muestras']} muestras{flag}")

    return X, y, meta


if __name__ == "__main__":
    import sys
    build_dataset(sys.argv[1] if len(sys.argv) > 1 else "data")
