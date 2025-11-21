# 05_regresion_logistica.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# Cargar el dataset maestro
ruta_dataset = "datos_enriquecidos/dataset_maestro_facturas.csv"
df = pd.read_csv(ruta_dataset, low_memory=False)

# ----------------------------------------------------------
# 1. Crear la variable objetivo para "alta_factura"
# ----------------------------------------------------------

# Crear una variable binaria "alta_factura" si la factura es mayor que 10,000
df['alta_factura'] = (df['vlr_total_neto_factura'] > 10000).astype(int)

# ----------------------------------------------------------
# 2. Seleccionar las variables predictoras
# ----------------------------------------------------------

# Seleccionar las variables predictoras: "es_internacional" y "otros atributos"
X = df[['es_internacional', 'cant_polizas', 'n_proveedores', 'genero', 'rango_edades']]

# Convertir las variables categóricas a variables numéricas
X = pd.get_dummies(X, drop_first=True)

# ----------------------------------------------------------
# 3. Imputar valores nulos (si los hay) en las variables predictoras
# ----------------------------------------------------------

# Usar un imputador para llenar los valores nulos con la media de cada columna
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# ----------------------------------------------------------
# 4. Crear la variable objetivo
# ----------------------------------------------------------

y = df['alta_factura']

# ----------------------------------------------------------
# 5. Dividir los datos en entrenamiento y prueba
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# ----------------------------------------------------------
# 6. Aplicar SMOTE para balancear las clases
# ----------------------------------------------------------

smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# ----------------------------------------------------------
# 7. Aplicar regresión logística
# ----------------------------------------------------------

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_smote, y_train_smote)

# Realizar predicciones
y_pred = log_reg.predict(X_test)

# ----------------------------------------------------------
# 8. Evaluar el modelo
# ----------------------------------------------------------

print("\nEvaluación del modelo de regresión logística con SMOTE:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
