# Taller 4 - MLflow Github Actions

## Descripción del proyecto

Este proyecto implementa un pipeline de Machine Learning con integración continua usando GitHub Actions y MLflow. El flujo automatiza la instalación de dependencias, el formateo del código, el entrenamiento del modelo, el registro del experimento en MLflow y la validación del modelo entrenado.

El objetivo principal es demostrar un flujo reproducible de MLOps en el que cada ejecución del pipeline registra parámetros, métricas, artefactos y el modelo entrenado.

## Dataset utilizado

El proyecto utiliza el dataset externo `penguins.csv`, disponible públicamente en el repositorio de datos de Seaborn:

https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv

Este dataset contiene información morfológica de pingüinos, incluyendo variables como longitud del pico, profundidad del pico, longitud de la aleta y masa corporal. La variable objetivo es la especie del pingüino.

Se eligió este dataset porque es externo a `sklearn.datasets`, tiene una estructura clara, permite resolver un problema de clasificación supervisada y se integra fácilmente en un pipeline automatizado.

## Problema de Machine Learning

El modelo busca predecir la especie de un pingüino a partir de las siguientes variables:

- bill_length_mm
- bill_depth_mm
- flipper_length_mm
- body_mass_g

La variable objetivo es:

- species

## Modelo utilizado

Se utiliza un modelo `RandomForestClassifier` de Scikit-learn. Este modelo fue seleccionado porque es adecuado para problemas de clasificación, maneja relaciones no lineales y ofrece buen desempeño sin requerir una configuración demasiado compleja.

## Estructura del proyecto

```text
mlflow-deploy/
├── train.py
├── validate.py
├── requirements.txt
├── Makefile
├── mlruns/
├── artifacts/
├── run_id.txt
├── validation_report.txt
└── .github/
    └── workflows/
        └── mlflow-ci.yml
