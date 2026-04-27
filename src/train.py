import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# Crear carpeta para MLflow
os.makedirs("mlruns", exist_ok=True)

# Configurar MLflow local
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("ci-cd-mlflow-local")

# Cargar datos
X, y = load_diabetes(return_X_y=True)

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Crear modelo
model = LinearRegression()

# Entrenar modelo
model.fit(X_train, y_train)

# Guardar modelo como archivo .pkl para que validate.py lo pueda usar
joblib.dump(model, "model.pkl")
print("Modelo guardado como model.pkl")

# Realizar predicciones
predictions = model.predict(X_test)

# Calcular métrica
mse = mean_squared_error(y_test, predictions)

# Registrar experimento en MLflow
with mlflow.start_run():
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("dataset", "diabetes")
    mlflow.log_metric("mse", mse)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

print("Entrenamiento finalizado correctamente.")
print(f"MSE: {mse}")
