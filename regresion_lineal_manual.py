import argparse
import csv
import os

import numpy as np

N_MOVEMENTS = 16
MISSING_FILES = {(14, 1)}  # 014_1.npy viene corrupto desde la fuente

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

CANALES = [
    "IMU1_Pitch", "IMU1_Yaw", "IMU1_Roll",
    "IMU2_Pitch", "IMU2_Yaw", "IMU2_Roll",
    "Glove_Thumb", "Glove_Index", "Glove_Middle", "Glove_Ring", "Glove_Pinky",
    "Glove_Pitch",
]

ESTADISTICOS = ["media", "std", "min", "max", "rms"]


def _cargar_par(base_path, movement_id):
    mov_str = f"{movement_id:03d}"
    arr1 = arr2 = None

    f1 = os.path.join(base_path, f"{mov_str}_1.npy")
    if os.path.exists(f1) and (movement_id, 1) not in MISSING_FILES:
        arr1 = np.load(f1)

    f2 = os.path.join(base_path, f"{mov_str}_2.npy")
    if os.path.exists(f2) and (movement_id, 2) not in MISSING_FILES:
        arr2 = np.load(f2)

    return arr1, arr2


def construir_dataset_crudo(data_dir, verbose=True):
    X_parts, y_parts = [], []

    for mov in range(N_MOVEMENTS):
        arr1, arr2 = _cargar_par(data_dir, mov)
        if arr1 is None or arr2 is None:
            if verbose:
                print(f"  mov {mov:02d} ({MOVEMENT_NAMES[mov]}): sin datos, se omite")
            continue
        if arr1.shape[0] != arr2.shape[0]:
            raise ValueError(f"movimiento {mov}: numero de muestras distinto entre sensores")
        X_mov = np.concatenate([arr1, arr2], axis=-1)
        X_parts.append(X_mov)
        y_parts.append(np.full(X_mov.shape[0], mov, dtype=int))
        if verbose:
            print(f"  mov {mov:02d} ({MOVEMENT_NAMES[mov]}): {X_mov.shape[0]} muestras")

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    return X, y


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


def guardar_dataset_csv(features, y, out_path):
    columnas = nombres_columnas() + ["movimiento_id", "movimiento_nombre"]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columnas)
        for fila, clase in zip(features, y):
            writer.writerow(list(fila) + [int(clase), MOVEMENT_NAMES[int(clase)]])
    print(f"dataset guardado en {out_path} ({features.shape[0]} filas, {features.shape[1]} columnas)")


def split_train_test_manual(X, y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx_train, idx_test = [], []

    for clase in np.unique(y):
        idx_clase = np.where(y == clase)[0]
        rng.shuffle(idx_clase)
        n_test = max(1, int(round(len(idx_clase) * test_size)))
        idx_test.extend(idx_clase[:n_test])
        idx_train.extend(idx_clase[n_test:])

    idx_train = np.array(idx_train)
    idx_test = np.array(idx_test)
    rng.shuffle(idx_train)
    rng.shuffle(idx_test)

    return X[idx_train], y[idx_train], X[idx_test], y[idx_test]


class RegresionLinealManual:
    def __init__(self, n_features, n_clases, tasa_aprendizaje=0.05, epochs=300, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.01, size=(n_features, n_clases))
        self.b = np.zeros(n_clases)
        self.tasa_aprendizaje = tasa_aprendizaje
        self.epochs = epochs
        self.historial_perdida = []
        self.media_ = None
        self.std_ = None

    @staticmethod
    def _one_hot(y, n_clases):
        m = np.zeros((len(y), n_clases))
        m[np.arange(len(y)), y] = 1.0
        return m

    def _estandarizar(self, X):
        return (X - self.media_) / self.std_

    def fit(self, X_train, y_train, verbose=True):
        self.media_ = X_train.mean(axis=0)
        self.std_ = X_train.std(axis=0)
        self.std_[self.std_ == 0] = 1.0

        X = self._estandarizar(X_train)
        Y = self._one_hot(y_train, self.W.shape[1])
        n = X.shape[0]

        for epoch in range(self.epochs):
            pred = X @ self.W + self.b
            error = pred - Y

            grad_W = (X.T @ error) / n
            grad_b = error.mean(axis=0)

            self.W -= self.tasa_aprendizaje * grad_W
            self.b -= self.tasa_aprendizaje * grad_b

            perdida = np.mean(error ** 2)
            self.historial_perdida.append(perdida)

            if verbose and (epoch % 50 == 0 or epoch == self.epochs - 1):
                print(f"  epoch {epoch:4d}  mse={perdida:.4f}")

        return self

    def predecir_scores(self, X):
        return self._estandarizar(X) @ self.W + self.b

    def predecir(self, X):
        return np.argmax(self.predecir_scores(X), axis=1)


def exactitud(y_real, y_pred):
    return float(np.mean(y_real == y_pred))


def matriz_confusion(y_real, y_pred, n_clases):
    m = np.zeros((n_clases, n_clases), dtype=int)
    for real, pred in zip(y_real, y_pred):
        m[real, pred] += 1
    return m


def exactitud_por_clase(matriz):
    return matriz.diagonal() / matriz.sum(axis=1).clip(min=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--csv-out", default="rehab_features_dataset.csv")
    args = parser.parse_args()

    print("cargando dataset...")
    X_crudo, y = construir_dataset_crudo(args.data_dir)
    print(f"X_crudo: {X_crudo.shape}   y: {y.shape}")

    print("\nconstruyendo tabla de features...")
    features = extraer_features_por_muestra(X_crudo)
    print(f"features: {features.shape}")
    guardar_dataset_csv(features, y, args.csv_out)

    print("\nsplit train/test...")
    X_train, y_train, X_test, y_test = split_train_test_manual(
        features, y, test_size=args.test_size
    )
    print(f"train: {X_train.shape[0]} muestras   test: {X_test.shape[0]} muestras")

    print("\nentrenando...")
    modelo = RegresionLinealManual(
        n_features=features.shape[1],
        n_clases=N_MOVEMENTS,
        tasa_aprendizaje=args.lr,
        epochs=args.epochs,
    )
    modelo.fit(X_train, y_train)

    print("\nresultados:")
    pred_train = modelo.predecir(X_train)
    pred_test = modelo.predecir(X_test)
    print(f"exactitud train: {exactitud(y_train, pred_train):.3f}")
    print(f"exactitud test:  {exactitud(y_test, pred_test):.3f}")

    matriz = matriz_confusion(y_test, pred_test, N_MOVEMENTS)
    acc_clase = exactitud_por_clase(matriz)
    print("\nexactitud por clase (test):")
    for c in range(N_MOVEMENTS):
        print(f"  clase {c:02d} ({MOVEMENT_NAMES[c]}): {acc_clase[c]:.2f}")

    print("\npredicciones de ejemplo:")
    rng = np.random.default_rng(123)
    idx_demo = rng.choice(len(X_test), size=min(10, len(X_test)), replace=False)
    for i in idx_demo:
        real = y_test[i]
        pred = pred_test[i]
        marca = "OK" if real == pred else "X "
        print(f"  [{marca}] real={MOVEMENT_NAMES[real]:45s} prediccion={MOVEMENT_NAMES[pred]}")


if __name__ == "__main__":
    main()
