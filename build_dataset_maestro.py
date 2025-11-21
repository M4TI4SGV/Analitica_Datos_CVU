# 01_build_dataset_maestro_facturas.py

import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------
# 1. Rutas de archivos (ajusta si tu estructura es distinta)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # carpeta donde está este script
RUTA_DATOS_ENRIQ = os.path.join(BASE_DIR, "datos_enriquecidos")

RUTA_CLIENTES = os.path.join(RUTA_DATOS_ENRIQ, "clientes_enriquecido.csv")
RUTA_FACTURAS = os.path.join(RUTA_DATOS_ENRIQ, "facturas_enriquecido.csv")
RUTA_PROV = os.path.join(RUTA_DATOS_ENRIQ, "proveedores_por_factura_enriquecido.csv")

print("Cargando archivos enriquecidos...")

df_clientes = pd.read_csv(RUTA_CLIENTES, low_memory=False)
df_facturas = pd.read_csv(RUTA_FACTURAS, low_memory=False)
df_prov = pd.read_csv(RUTA_PROV, low_memory=False)

print(f"clientes_enriquecido: {len(df_clientes):,} filas")
print(f"facturas_enriquecido: {len(df_facturas):,} filas")
print(f"proveedores_por_factura_enriquecido: {len(df_prov):,} filas")

# ---------------------------------------------------------
# 2. Asegurar tipos básicos
# ---------------------------------------------------------
# no_factura debería ser string para evitar problemas con ceros a la izquierda, etc.
for df in [df_facturas, df_prov]:
    if "no_factura" in df.columns:
        df["no_factura"] = df["no_factura"].astype(str).str.strip()

# vlr_presupuesto_ppto a numérico
if "vlr_presupuesto_ppto" in df_prov.columns:
    df_prov["vlr_presupuesto_ppto"] = pd.to_numeric(
        df_prov["vlr_presupuesto_ppto"], errors="coerce"
    )

# ---------------------------------------------------------
# 3. Agregación de proveedores por factura
#    (puede haber varias filas por no_factura)
# ---------------------------------------------------------
def proveedor_principal(grupo):
    """
    Devuelve el nombre_proveedor principal para una factura:
    - si hay valores de vlr_presupuesto_ppto, el de mayor valor;
    - si no, el proveedor más frecuente;
    - si no hay nombre_proveedor, NaN.
    """
    sub = grupo.dropna(subset=["nombre_proveedor"])
    if sub.empty:
        return np.nan

    if sub["vlr_presupuesto_ppto"].notna().any():
        idx = sub["vlr_presupuesto_ppto"].fillna(0).idxmax()
        return sub.loc[idx, "nombre_proveedor"]

    # si no hay valores numéricos, usar el modo
    return sub["nombre_proveedor"].mode().iloc[0]


def agregar_proveedores(grupo):
    n_prov = grupo["nombre_proveedor"].dropna().nunique()
    suma = grupo["vlr_presupuesto_ppto"].sum(min_count=1)
    prom = grupo["vlr_presupuesto_ppto"].mean()
    prov_princ = proveedor_principal(grupo)

    return pd.Series(
        {
            "n_proveedores": n_prov,
            "suma_vlr_presupuesto_ppto": suma,
            "prom_vlr_presupuesto_ppto": prom,
            "proveedor_principal": prov_princ,
        }
    )


print("Agregando información de proveedores por factura...")
df_prov_agg = (
    df_prov.groupby("no_factura")
    .apply(agregar_proveedores)
    .reset_index()
)

print(f"Tabla agregada de proveedores: {len(df_prov_agg):,} facturas")

# ---------------------------------------------------------
# 4. Unir facturas con atributos del 'cliente' de esa factura
#    (en tu pipeline, id_cliente es 1:1 con no_factura)
# ---------------------------------------------------------
if "id_cliente" not in df_facturas.columns:
    raise ValueError("facturas_enriquecido no tiene columna 'id_cliente'.")

if "id_cliente" not in df_clientes.columns:
    raise ValueError("clientes_enriquecido no tiene columna 'id_cliente'.")

print("Uniendo facturas con información de clientes (por id_cliente)...")

df_fact_cli = df_facturas.merge(
    df_clientes,
    on="id_cliente",
    how="left",
    suffixes=("_fac", "_cli")  # por si en algún momento hay nombres repetidos
)

print(f"Facturas + clientes: {len(df_fact_cli):,} filas")

# ---------------------------------------------------------
# 5. Unir la agregación de proveedores a nivel factura
# ---------------------------------------------------------
print("Uniendo información de proveedores (por no_factura)...")

df_maestro = df_fact_cli.merge(
    df_prov_agg,
    on="no_factura",
    how="left"
)

# ---------------------------------------------------------
# 6. Crear algunas variables derivadas útiles (nacional vs internacional)
# ---------------------------------------------------------
print("Creando variables derivadas...")

# es_internacional = 1 si el destino no es Colombia y no es NaN
df_maestro["es_internacional"] = np.where(
    (df_maestro["destino_pais"].notna()) & (df_maestro["destino_pais"] != "Colombia"),
    1,
    0,
)

# año_factura (si no existe ya)
if "fecha_factura" in df_maestro.columns and "anio_factura" not in df_maestro.columns:
    df_maestro["fecha_factura"] = pd.to_datetime(
        df_maestro["fecha_factura"], errors="coerce"
    )
    df_maestro["anio_factura"] = df_maestro["fecha_factura"].dt.year

print(f"Dataset maestro a nivel factura: {len(df_maestro):,} filas")

# ---------------------------------------------------------
# 7. Guardar resultado
# ---------------------------------------------------------
RUTA_SALIDA = os.path.join(RUTA_DATOS_ENRIQ, "dataset_maestro_facturas.csv")
df_maestro.to_csv(RUTA_SALIDA, index=False, encoding="utf-8-sig")

print(f"\nArchivo guardado en: {RUTA_SALIDA}")

print("\nColumnas del dataset maestro:")
print(df_maestro.columns.tolist())

print("\nVista rápida de 5 filas:")
print(df_maestro.head())
