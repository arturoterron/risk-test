# -*- coding: utf-8 -*-
"""scrl-test2.ipynb


"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Generar datos de ejemplo para el modelo (igual que antes)
np.random.seed(42)
n_samples = 1000

# Variables independientes (factores de riesgo)
edad = np.random.normal(40, 10, n_samples)
ingresos_anuales = np.random.normal(30000, 15000, n_samples)
antiguedad_laboral = np.random.normal(8, 5, n_samples)
deuda_actual = np.random.normal(15000, 10000, n_samples)
num_prestamos_anteriores = np.random.randint(0, 10, n_samples)
num_impagos = np.random.randint(0, 5, n_samples)
ratio_deuda_ingresos = deuda_actual / ingresos_anuales

# Creación del puntaje crediticio teórico (variable dependiente)
# Más alto = mejor solvencia
score_base = 600
contribucion_edad = (edad - 20) * 1.5
contribucion_ingresos = ingresos_anuales * 0.003
contribucion_antiguedad = antiguedad_laboral * 5
penalizacion_deuda = ratio_deuda_ingresos * 100
penalizacion_impagos = num_impagos * 30

# Añadir algunas relaciones no lineales para mostrar la ventaja de XGBoost
# XGBoost debería captar estas relaciones que la regresión lineal no puede
efecto_combinado = (edad > 30) * (ingresos_anuales > 40000) * 50  # Bonus para adultos con buenos ingresos
efecto_historial = np.where(num_impagos == 0, antiguedad_laboral * 2, 0)  # Bonus por historial limpio y antigüedad

# Fórmula del puntaje de crédito teórico con no linealidades
puntaje_credito = (
    score_base +
    contribucion_edad +
    contribucion_ingresos +
    contribucion_antiguedad -
    penalizacion_deuda -
    penalizacion_impagos +
    efecto_combinado +
    efecto_historial
)
# Añadir algo de ruido
puntaje_credito = puntaje_credito + np.random.normal(0, 25, n_samples)
# Asegurar que esté entre 300 y 850 (rango típico de puntaje crediticio)
puntaje_credito = np.clip(puntaje_credito, 300, 850)

# Crear DataFrame
datos = pd.DataFrame({
    'edad': edad,
    'ingresos_anuales': ingresos_anuales,
    'antiguedad_laboral': antiguedad_laboral,
    'deuda_actual': deuda_actual,
    'num_prestamos_anteriores': num_prestamos_anteriores,
    'num_impagos': num_impagos,
    'ratio_deuda_ingresos': ratio_deuda_ingresos,
    'puntaje_credito': puntaje_credito
})

# Dividir en conjunto de entrenamiento y prueba
X = datos.drop('puntaje_credito', axis=1)
y = datos['puntaje_credito']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizar características (opcional para XGBoost, pero ayuda a la comparabilidad)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. ENTRENAR EL MODELO DE REGRESIÓN LINEAL (para comparación)
from sklearn.linear_model import LinearRegression
modelo_lineal = LinearRegression()
modelo_lineal.fit(X_train_scaled, y_train)

# Evaluar el modelo lineal
y_pred_lineal = modelo_lineal.predict(X_test_scaled)
mse_lineal = mean_squared_error(y_test, y_pred_lineal)
r2_lineal = r2_score(y_test, y_pred_lineal)
print("=== MODELO DE REGRESIÓN LINEAL ===")
print(f"Error cuadrático medio: {mse_lineal:.2f}")
print(f"R² Score: {r2_lineal:.4f}")

# 2. ENTRENAR MODELO XGBOOST
# Crear y entrenar el modelo XGBoost
modelo_xgb = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
modelo_xgb.fit(X_train, y_train)

# Realizar predicciones con XGBoost
y_pred_xgb = modelo_xgb.predict(X_test)

# Evaluar el modelo XGBoost
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
print("\n=== MODELO XGBOOST ===")
print(f"Error cuadrático medio: {mse_xgb:.2f}")
print(f"R² Score: {r2_xgb:.4f}")
print(f"Mejora en R² sobre modelo lineal: {((r2_xgb - r2_lineal) / r2_lineal * 100):.2f}%")
print(f"Reducción del error: {((mse_lineal - mse_xgb) / mse_lineal * 100):.2f}%")

# Mostrar la importancia de las características en XGBoost
importancias = modelo_xgb.feature_importances_
indices_ordenados = np.argsort(importancias)[::-1]
nombres_variables = X.columns[indices_ordenados]

plt.figure(figsize=(10, 6))
plt.title("Importancia de Variables (XGBoost)")
plt.bar(range(X.shape[1]), importancias[indices_ordenados], align="center")
plt.xticks(range(X.shape[1]), nombres_variables, rotation=45)
plt.tight_layout()
plt.show()

# Comparar predicciones de ambos modelos
comparacion = pd.DataFrame({
    'Real': y_test,
    'Predicción Lineal': y_pred_lineal,
    'Predicción XGBoost': y_pred_xgb,
    'Error Lineal': np.abs(y_test - y_pred_lineal),
    'Error XGBoost': np.abs(y_test - y_pred_xgb)
})

print("\n=== COMPARACIÓN DE ERRORES ===")
print(f"Error medio lineal: {comparacion['Error Lineal'].mean():.2f}")
print(f"Error medio XGBoost: {comparacion['Error XGBoost'].mean():.2f}")

# Ejemplo de aplicación para un nuevo solicitante
nuevo_solicitante = pd.DataFrame({
    'edad': [35],
    'ingresos_anuales': [45000],
    'antiguedad_laboral': [5],
    'deuda_actual': [10000],
    'num_prestamos_anteriores': [2],
    'num_impagos': [0],
    'ratio_deuda_ingresos': [10000/45000]
})

# Clasificar riesgo
def clasificar_riesgo(puntaje):
    if puntaje >= 750:
        return "Muy Bajo"
    elif puntaje >= 650:
        return "Bajo"
    elif puntaje >= 550:
        return "Moderado"
    elif puntaje >= 450:
        return "Alto"
    else:
        return "Muy Alto"

# Predecir con ambos modelos para el nuevo solicitante
nuevo_solicitante_scaled = scaler.transform(nuevo_solicitante)
score_lineal = modelo_lineal.predict(nuevo_solicitante_scaled)[0]
score_xgb = modelo_xgb.predict(nuevo_solicitante)[0]

print("\n=== ANÁLISIS DEL NUEVO SOLICITANTE ===")
print("Modelo Lineal:")
print(f"  Puntaje predicho: {score_lineal:.2f}")
print(f"  Categoría de riesgo: {clasificar_riesgo(score_lineal)}")
print(f"  Decisión: {'Aprobar' if score_lineal >= 550 else 'Rechazar'} el crédito")

print("\nModelo XGBoost:")
print(f"  Puntaje predicho: {score_xgb:.2f}")
print(f"  Categoría de riesgo: {clasificar_riesgo(score_xgb)}")
print(f"  Decisión: {'Aprobar' if score_xgb >= 550 else 'Rechazar'} el crédito")

# Visualizar las predicciones vs valores reales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lineal, alpha=0.5, color='blue', label='Regresión Lineal')
plt.scatter(y_test, y_pred_xgb, alpha=0.5, color='red', label='XGBoost')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel('Puntaje Real')
plt.ylabel('Puntaje Predicho')
plt.title('Comparación de Predicciones: Regresión Lineal vs XGBoost')
plt.legend()
plt.tight_layout()
plt.show()
