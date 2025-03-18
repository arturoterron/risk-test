# -*- coding: utf-8 -*-
"""
test de analisis de supervivencia crediticia

"""

# ===========================================
# ANÁLISIS DE SUPERVIVENCIA EN RIESGO CREDITICIO
# ===========================================

# 1️ CARGAR LIBRERÍAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter

def generar_datos(n=200, seed=42):
    np.random.seed(seed)
    
    data = pd.DataFrame({
        "tiempo": np.random.exponential(scale=12, size=n),  # Tiempo hasta impago
        "impago": np.random.choice([1, 0], size=n, p=[0.4, 0.6]),  # 1=Impago, 0=Censurado
        "ingreso_mensual": np.random.randint(800, 5000, size=n),
        "monto_prestamo": np.random.randint(1000, 20000, size=n),
        "edad": np.random.randint(18, 65, size=n),
        "historico_impagos": np.random.randint(0, 5, size=n)
    })

    # Ajustar los tiempos censurados correctamente
    censurados = data[data["impago"] == 0]  # Filtrar registros censurados
    num_censurados = len(censurados)
    
    # Añadir un valor aleatorio entre 5 y 15 meses a los registros censurados
    if num_censurados > 0:
        data.loc[data["impago"] == 0, "tiempo"] += np.random.randint(5, 15, size=num_censurados)

    return data

# Análisis de supervivencia con Kaplan-Meier
def curva_kaplan_meier(data, ax):
    kmf = KaplanMeierFitter()
    kmf.fit(data["tiempo"], event_observed=data["impago"])
    
    kmf.plot_survival_function(ax=ax)
    ax.set_title("Curva de Supervivencia (Kaplan-Meier)")
    ax.set_xlabel("Tiempo (meses)")
    ax.set_ylabel("Probabilidad de Supervivencia (No Impago)")
    ax.grid()

# Modelo de riesgos proporcionales de Cox
def modelo_cox(data, ax):
    cph = CoxPHFitter()
    cph.fit(data, duration_col="tiempo", event_col="impago")
    
    print("\n🔹 Resumen del Modelo de Cox:\n")
    cph.print_summary()
    
    # Visualizar coeficientes
    cph.plot(ax=ax)
    ax.set_title("Efecto de las Variables en el Riesgo de Impago")
    ax.grid()

# Ejecutando el análisis
if __name__ == "__main__":
    print("\n🔹 Generando datos de microcréditos...\n")
    data = generar_datos()

    print("\n🔹 Mostrando los primeros registros del dataset:\n")
    print(data.head())

    # Crear figura con dos subgráficas (1 fila, 2 columnas)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    print("\n🔹 Generando curva de Kaplan-Meier...\n")
    curva_kaplan_meier(data, axes[0])

    print("\n🔹 Ajustando el modelo de Cox...\n")
    modelo_cox(data, axes[1])

    print("\n✅ Análisis de supervivencia completado con éxito.")

    # Mostrar ambas gráficas
    plt.tight_layout()  # Ajustar espacio entre subgráficas
    plt.show()