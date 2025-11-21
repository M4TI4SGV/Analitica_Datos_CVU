# 03_kmeans_segmentacion.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Cargar el dataset maestro
ruta_dataset = "datos_enriquecidos/dataset_maestro_facturas.csv"
df = pd.read_csv(ruta_dataset, low_memory=False)

# ----------------------------------------------------------
# 1. Preparación de los datos
# ----------------------------------------------------------

# Seleccionar las columnas de interés: género, rango de edad y valor de la factura
df_clean = df[['genero', 'rango_edades', 'vlr_total_neto_factura', 'vlr_total_item_factura']]

# Imputar los valores nulos con la media para las variables numéricas
df_clean['vlr_total_neto_factura'].fillna(df_clean['vlr_total_neto_factura'].mean(), inplace=True)
df_clean['vlr_total_item_factura'].fillna(df_clean['vlr_total_item_factura'].mean(), inplace=True)

# Convertir las variables categóricas a variables numéricas
df_clean = pd.get_dummies(df_clean, drop_first=True)

# Normalizar los datos
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_clean)

# ----------------------------------------------------------
# 2. Aplicar K-Means para segmentar a los clientes
# ----------------------------------------------------------

# Determinar el número de clústeres óptimo (usando el método del codo)
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

# Graficar el método del codo
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Método del Codo para K-Means')
plt.xlabel('Número de Clústeres')
plt.ylabel('Inercia')
plt.show()

# Aplicar K-Means con el número óptimo de clústeres (por ejemplo, 3)
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(df_scaled)

# ----------------------------------------------------------
# 3. Visualización de los clústeres
# ----------------------------------------------------------

pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Graficar los clústeres
plt.figure(figsize=(8, 6))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df['cluster'], cmap='viridis')
plt.title('Segmentación de Clientes con K-Means')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.show()

# ----------------------------------------------------------
# 4. Análisis de los clústeres según género y rango de edad
# ----------------------------------------------------------

# Ver qué género y rango de edad pertenecen a cada clúster
print("\nDistribución de género y rango de edad por clúster:")
print(df.groupby(['cluster', 'genero']).size())
print(df.groupby(['cluster', 'rango_edades']).size())
