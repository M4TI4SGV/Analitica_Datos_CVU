# 02_eda_facturas.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Opcional: Establecer el estilo de gráficos
sns.set(style="whitegrid")

# Ruta del dataset maestro (ajusta si es necesario)
ruta_dataset = "datos_enriquecidos/dataset_maestro_facturas.csv"

# Cargar el dataset maestro
df = pd.read_csv(ruta_dataset, low_memory=False)

# ----------------------------------------------------------
# 1. Información general sobre el DataFrame
# ----------------------------------------------------------
print("Información del DataFrame:")
print(df.info())

print("\nDescripción de variables numéricas:")
print(df.select_dtypes(include=[np.number]).describe().T)

print("\nPorcentaje de nulos por columna:")
print((df.isna().mean() * 100).round(2).sort_values(ascending=False))

# ----------------------------------------------------------
# 2. Histogramas de las variables numéricas claves
# ----------------------------------------------------------
cols_numericas_clave = [
    "vlr_total_neto_factura",
    "vlr_total_item_factura",
    "vlr_total_neto_item_factura",
    "suma_vlr_presupuesto_ppto",
    "prom_vlr_presupuesto_ppto",
    "n_proveedores",
]

df[cols_numericas_clave].hist(bins=50, figsize=(14, 10))
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 3. Análisis de variables categóricas: Barras para género, estado civil, etc.
# ----------------------------------------------------------
cat_cols_basicas = ["genero", "estado_civil", "rango_edades", "region_colombia"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for col, ax in zip(cat_cols_basicas, axes):
    df[col].value_counts(dropna=False).plot(kind="bar", ax=ax)
    ax.set_title(f"Distribución de {col}")
    ax.set_xlabel("")
    ax.set_ylabel("Frecuencia")

plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 4. Nacional vs Internacional (basado en destino_pais)
# ----------------------------------------------------------
plt.figure(figsize=(5, 4))
df["es_internacional"].value_counts().sort_index().plot(kind="bar", rot=0)
plt.xticks([0, 1], ["Nacional (0)", "Internacional (1)"])
plt.title("Distribución de viajes nacionales vs internacionales")
plt.ylabel("Número de facturas")
plt.show()

# Ver porcentaje de viajes internacionales
print("\nProporción de viajes internacionales:")
print(df["es_internacional"].mean())

# ----------------------------------------------------------
# 5. Top destinos más frecuentes
# ----------------------------------------------------------
top_destinos = df["destino_ciudad"].value_counts().head(10)
print("\nTop 10 destinos más frecuentes:")
print(top_destinos)

# ----------------------------------------------------------
# 6. Top 10 proveedores más frecuentes
# ----------------------------------------------------------
top_proveedores = df["proveedor_principal"].value_counts().head(10)
print("\nTop 10 proveedores más frecuentes:")
print(top_proveedores)

# ----------------------------------------------------------
# 7. Análisis de valores por continente de destino
# ----------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="destino_continente", y="vlr_total_neto_factura")
plt.title("Distribución de valores por continente de destino")
plt.show()

# ----------------------------------------------------------
# 8. Análisis de valores por rango de edad
# ----------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="rango_edades", y="vlr_total_neto_factura")
plt.title("Distribución de valores por rango de edad")
plt.xticks(rotation=45)
plt.show()

# ----------------------------------------------------------
# 9. Correlaciones entre las variables numéricas
# ----------------------------------------------------------
correlation_matrix = df[cols_numericas_clave].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Matriz de correlación")
plt.show()
