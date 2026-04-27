import joblib

from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# Cargar datos
X, y = load_diabetes(return_X_y=True)

# Dividir datos en entrenamiento y prueba
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Cargar modelo entrenado
model = joblib.load("model.pkl")

# Realizar predicciones
predictions = model.predict(X_test)

# Calcular métrica de desempeño
mse = mean_squared_error(y_test, predictions)

print("Validación finalizada correctamente.")
print(f"MSE: {mse}")

# Validar umbral mínimo de desempeño
if mse < 3000:
    print("Modelo aceptado: cumple con el umbral de desempeño.")
else:
    raise Exception("Modelo no cumple el umbral mínimo de desempeño.")
