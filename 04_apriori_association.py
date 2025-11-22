import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Cargar el dataset maestro
ruta_dataset = "datos_enriquecidos/dataset_maestro_facturas.csv"
df = pd.read_csv(ruta_dataset)

# Tomar una muestra del 50% de los datos para probar
df_sample = df.sample(frac=0.50, random_state=42)

# Eliminar las filas con valores nulos
df_transacciones = df_sample.dropna()

# Seleccionar las columnas relevantes para la asociación
df_transacciones = df_transacciones[['proveedor_principal', 'destino_ciudad', 'genero', 'rango_edades']]

# Realizamos One-Hot Encoding en lugar de Label Encoding
df_transacciones = pd.get_dummies(df_transacciones)

# Ejecutar Apriori para encontrar los conjuntos frecuentes con un soporte mínimo de 0.01
frequent_itemsets = apriori(df_transacciones, min_support=0.01, use_colnames=True)

# Generar las reglas de asociación
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.5)

# Mostrar todas las reglas de asociación encontradas
print("Reglas de asociación encontradas:")
print(rules)

# Filtrar las reglas de alta confianza (mayores a 0.7)
rules_high_confidence = rules[rules['confidence'] > 0.7]
print("\nReglas de alta confianza:")
print(rules_high_confidence)

# Filtrar las reglas con alto lift (mayores a 3)
rules_high_lift = rules[rules['lift'] > 3]
print("\nReglas con alto lift:")
print(rules_high_lift)

# Probar con diferentes valores de min_support y min_threshold
# Cambiar el min_support a 0.02 y min_threshold a 1.5 para obtener reglas más comunes y fuertes
frequent_itemsets_2 = apriori(df_transacciones, min_support=0.02, use_colnames=True)
rules_2 = association_rules(frequent_itemsets_2, metric="lift", min_threshold=1.5)

# Mostrar las nuevas reglas
print("\nNuevas reglas con min_support=0.02 y min_threshold=1.5:")
print(rules_2)

# Filtrar las nuevas reglas de alta confianza y alto lift
rules_high_confidence_2 = rules_2[rules_2['confidence'] > 0.7]
rules_high_lift_2 = rules_2[rules_2['lift'] > 3]

print("\nNuevas reglas de alta confianza:")
print(rules_high_confidence_2)

print("\nNuevas reglas con alto lift:")
print(rules_high_lift_2)
