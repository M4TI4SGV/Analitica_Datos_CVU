# 04_apriori_proveedores_destinos.py

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Cargar el dataset maestro
ruta_dataset = "datos_enriquecidos/dataset_maestro_facturas.csv"
df = pd.read_csv(ruta_dataset, low_memory=False)

# ----------------------------------------------------------
# 1. Preparación de los datos para Apriori (proveedor y destino)
# ----------------------------------------------------------

# Crear una lista de transacciones (por cada factura, los destinos y proveedores como items)
transactions = []

for _, row in df.iterrows():
    transaction = []
    
    # Agregar destinos y proveedores como items en la transacción
    if pd.notna(row['destino_ciudad']):
        transaction.append(row['destino_ciudad'])
    
    if pd.notna(row['proveedor_principal']):
        transaction.append(row['proveedor_principal'])
    
    transactions.append(transaction)

# ----------------------------------------------------------
# 2. Transformar los datos para Apriori (Transaction Encoder)
# ----------------------------------------------------------

# Usamos TransactionEncoder para convertir las transacciones en un formato adecuado para Apriori
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_apriori = pd.DataFrame(te_ary, columns=te.columns_)

# ----------------------------------------------------------
# 3. Aplicar Apriori para encontrar los itemsets frecuentes
# ----------------------------------------------------------

# Obtener itemsets frecuentes con un soporte mínimo de 0.003 (ajustamos el soporte)
frequent_itemsets = apriori(df_apriori, min_support=0.003, use_colnames=True)

# ----------------------------------------------------------
# 4. Generar reglas de asociación
# ----------------------------------------------------------

# Generar reglas de asociación con una confianza mínima de 0.4
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4)

# ----------------------------------------------------------
# 5. Visualizar las reglas
# ----------------------------------------------------------

# Mostrar las primeras 10 reglas con soporte, confianza y lift
print("\nPrimeras 10 reglas de asociación encontradas:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# Guardar las reglas en un archivo CSV (opcional)
rules.to_csv("reglas_asociacion_proveedores_destinos.csv", index=False)
