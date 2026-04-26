import os
import json
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Crear carpetas necesarias
os.makedirs("mlruns", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

# Configurar MLflow local
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("penguins-classification")

# Dataset externo en formato CSV
DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"

# Cargar datos
df = pd.read_csv(DATA_URL)

# Seleccionar variables útiles
columns = [
    "species",
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]

df = df[columns].dropna()

# Separar variables predictoras y variable objetivo
X = df[
    [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]
]

y = df["species"]

# Codificar variable objetivo
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Guardar clases del modelo
classes = {int(i): label for i, label in enumerate(encoder.classes_)}

with open("artifacts/classes.json", "w") as f:
    json.dump(classes, f, indent=4)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded,
)

# Definir modelo
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
)

# Entrenar modelo
model.fit(X_train, y_train)

# Evaluar en datos de prueba
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

# Crear firma e input example para MLflow
input_example = X_train.head(5)
signature = infer_signature(X_train, model.predict(X_train))

# Registrar experimento en MLflow
with mlflow.start_run() as run:
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("dataset", "penguins.csv external dataset")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    mlflow.log_artifact("artifacts/classes.json")

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
    )

    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)

print("Entrenamiento finalizado correctamente.")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Run ID: {run.info.run_id}")
