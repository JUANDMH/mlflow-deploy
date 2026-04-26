import os
import mlflow
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Configurar MLflow local
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("penguins-classification")

# Leer el run_id generado durante el entrenamiento
if not os.path.exists("run_id.txt"):
    raise FileNotFoundError("No se encontró run_id.txt. Primero ejecuta make train.")

with open("run_id.txt", "r") as f:
    run_id = f.read().strip()

# Cargar modelo registrado en MLflow
model_uri = f"runs:/{run_id}/model"
model = mlflow.sklearn.load_model(model_uri)

# Cargar nuevamente el dataset externo
DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"

df = pd.read_csv(DATA_URL)

columns = [
    "species",
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]

df = df[columns].dropna()

X = df[
    [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]
]

y = df["species"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

_, X_test, _, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded,
)

# Predicción
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Guardar reporte local
with open("validation_report.txt", "w") as f:
    f.write(f"Validation accuracy: {accuracy:.4f}\n\n")
    f.write("Classification report:\n")
    f.write(report)

# Registrar métrica de validación en el mismo experimento
with mlflow.start_run(run_id=run_id):
    mlflow.log_metric("validation_accuracy", accuracy)
    mlflow.log_artifact("validation_report.txt")

# Umbral mínimo de validación
if accuracy < 0.80:
    raise ValueError(f"Modelo no aceptado. Accuracy: {accuracy:.4f}")

print("Validación finalizada correctamente.")
print(f"Validation accuracy: {accuracy:.4f}")
