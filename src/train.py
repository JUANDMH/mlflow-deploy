import joblib
model.fit(X_train, y_train)
joblib.dump(model, "model.pkl")
print("✅ Modelo guardado como model.pkl")
