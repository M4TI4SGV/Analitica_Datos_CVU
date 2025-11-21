# 06_modelos_alternativos.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
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
# 7. Entrenar modelos de Random Forest y Gradient Boosting
# ----------------------------------------------------------

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_smote, y_train_smote)

# Gradient Boosting
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train_smote, y_train_smote)

# ----------------------------------------------------------
# 8. Evaluar los modelos
# ----------------------------------------------------------

# Predicciones de ambos modelos
y_pred_rf = rf_model.predict(X_test)
y_pred_gb = gb_model.predict(X_test)

# Resultados para Random Forest
print("\nEvaluación de Random Forest:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))

# Resultados para Gradient Boosting
print("\nEvaluación de Gradient Boosting:")
print("Accuracy:", accuracy_score(y_test, y_pred_gb))
print(classification_report(y_test, y_pred_gb))
print(confusion_matrix(y_test, y_pred_gb))
