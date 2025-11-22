import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Cargar los datos
df = pd.read_csv('datos_enriquecidos/dataset_maestro_facturas.csv')  # Ajusta la ruta si es necesario

# Eliminar las columnas no numéricas que no aportan al clustering
df_numeric = df.select_dtypes(include=[np.number])

# Imputar o eliminar los valores faltantes (NaN) en las columnas numéricas
df_numeric = df_numeric.fillna(df_numeric.mean())  # O usa .dropna() si prefieres eliminar las filas con NaN

# Normalizar los datos
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)

# Aplicar K-Means
kmeans = KMeans(n_clusters=4, random_state=42)  # Ajusta el número de clusters
df['cluster'] = kmeans.fit_predict(df_scaled)

# Resumen por cluster (media de las columnas numéricas)
cluster_summary = df.groupby('cluster')[df_numeric.columns].mean()

# Mostrar el resumen
print("Resumen de Clusters")
print(cluster_summary)

# Graficar el método del codo para elegir el número de clusters
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o', color='blue')
plt.title("Método del Codo para K-Means")
plt.xlabel("Número de Clústeres")
plt.ylabel("Inercia")
plt.show()

# Graficar la segmentación de clientes en 2D (PCA para reducir la dimensionalidad)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df['cluster'], cmap='viridis')
plt.title("Segmentación de Clientes con K-Means")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.colorbar()
plt.show()
