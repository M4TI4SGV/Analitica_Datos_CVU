import pandas as pd
import glob
import os
import numpy as np

# -----------------------------------------------------------------------------
# 1. DEFINICIÓN DE COLUMNAS
# -----------------------------------------------------------------------------

# Columnas originales que queremos leer de los CSV
COLUMNAS_DESEADAS = [
    'Estado Civil',
    'Pais Residencia',
    'Cant Polizas',
    'Rango Edades',
    'Genero',
    'Zonas Ciudades Cli',
    'Fecha Factura',
    'Ciudad Destino', # <-- AÑADIDO
    'Nombre Proveedor',
    'Valor Presupuesto Servicios Ppto',
    'No. Factura',
    'Valor Total Neto Factura',
    'Valor Total Item Factura',
    'Valor Total Neto Item Factura'
]

# Mapeo para renombrar columnas a un formato limpio
MAPEO_NOMBRES = {
    'Estado Civil': 'estado_civil',
    'Pais Residencia': 'pais_residencia',
    'Cant Polizas': 'cant_polizas',
    'Rango Edades': 'rango_edades',
    'Genero': 'genero',
    'Zonas Ciudades Cli': 'zonas_ciudades_cli',
    'Fecha Factura': 'fecha_factura',
    'Ciudad Destino': 'ciudad_destino', # <-- AÑADIDO
    'Nombre Proveedor': 'nombre_proveedor',
    'Valor Presupuesto Servicios Ppto': 'vlr_presupuesto_ppto',
    'No. Factura': 'no_factura',
    'Valor Total Neto Factura': 'vlr_total_neto_factura',
    'Valor Total Item Factura': 'vlr_total_item_factura',
    'Valor Total Neto Item Factura': 'vlr_total_neto_item_factura'
}

# Columnas de atributos que definen a un cliente
COLUMNAS_ATRIBUTOS_CLIENTE = ['estado_civil', 'pais_residencia', 'cant_polizas', 'rango_edades', 'genero', 'zonas_ciudades_cli']
# Columnas para la tabla final de Factura
# NOTA: 'id_cliente' se añadirá durante el procesamiento
COLUMNAS_FACTURA_FINAL = [
    'no_factura', 
    'id_cliente', 
    'fecha_factura', 
    'ciudad_destino', # <-- AÑADIDO
    'vlr_total_neto_factura', 
    'vlr_total_item_factura', 
    'vlr_total_neto_item_factura'
]
# Columnas para la tabla de Proveedores
COLUMNAS_PROVEEDOR_FACTURA = ['no_factura', 'nombre_proveedor', 'vlr_presupuesto_ppto']

# -----------------------------------------------------------------------------
# 2. FUNCIONES DE PROCESAMIENTO
# -----------------------------------------------------------------------------

def cargar_y_consolidar(carpeta_entrada):
    """
    Carga y une todos los CSV de la carpeta de entrada, leyendo solo las columnas deseadas.
    """
    print(f"Iniciando el procesamiento de la carpeta: {carpeta_entrada}")
    patron_archivos = os.path.join(carpeta_entrada, "*.csv")
    lista_archivos_csv = glob.glob(patron_archivos)
    
    if not lista_archivos_csv:
        print(f"Error: No se encontraron archivos CSV en la carpeta '{carpeta_entrada}'.")
        return None

    print(f"Se encontraron {len(lista_archivos_csv)} archivos CSV.")
    
    lista_dfs = []
    for archivo in lista_archivos_csv:
        try:
            print(f"Cargando archivo: {archivo}...")
            # Lee solo las columnas que nos interesan
            # Usamos 'on_bad_lines='skip'' por si alguna fila tiene más comas de las esperadas
            df = pd.read_csv(archivo, usecols=lambda c: c in COLUMNAS_DESEADAS, on_bad_lines='skip')
            
            # Asegurarse de que todas las columnas deseadas existan, rellenando con NaN si faltan
            for col in COLUMNAS_DESEADAS:
                if col not in df.columns:
                    df[col] = np.nan
            
            # Reordenar las columnas para que coincidan con COLUMNAS_DESEADAS
            df = df[COLUMNAS_DESEADAS]
            lista_dfs.append(df)
            
        except ValueError as ve:
            print(f"Advertencia: Posible error de columnas en {archivo}. {ve}")
            # Intentar leer de nuevo sin la restricción de 'usecols' para ver qué está pasando
            try:
                df_test = pd.read_csv(archivo, nrows=1)
                print(f"Columnas encontradas en el archivo: {df_test.columns.tolist()}")
            except Exception as e_test:
                print(f"No se pudo ni siquiera leer la cabecera: {e_test}")
                
        except Exception as e:
            print(f"Error al leer el archivo {archivo}: {e}")
            
    if not lista_dfs:
        print("No se pudo cargar ningún archivo. Abortando.")
        return None

    df_consolidado = pd.concat(lista_dfs, ignore_index=True)
    print(f"Total de filas consolidadas (brutas): {df_consolidado.shape[0]}")
    return df_consolidado

def limpiar_datos(df):
    """
    Aplica la limpieza de tipos y nulos al DataFrame consolidado.
    """
    print("Iniciando limpieza de datos...")
    
    # 1. Renombrar columnas
    df = df.rename(columns=MAPEO_NOMBRES)
    
    # 2. Manejar Nulos (convertir "" a pd.NA/NaN)
    print("Convirtiendo campos de texto vacíos a Nulo (NaN)...")
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)

    # 3. Convertir tipos de datos
    print("Convirtiendo tipos de datos (fechas y números)...")
    df['fecha_factura'] = pd.to_datetime(df['fecha_factura'], errors='coerce')
    
    numeric_cols = ['vlr_total_neto_factura', 'vlr_total_item_factura', 'vlr_total_neto_item_factura', 'vlr_presupuesto_ppto']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
    print("Limpieza de tipos completada.")
    return df

def crear_tablas_normalizadas(df_limpio, carpeta_salida):
    """
    Crea las 3 tablas normalizadas (Clientes, Facturas, Proveedores_Factura) y las guarda como CSV.
    OPCIÓN 1: Un ID de Cliente ÚNICO por cada Factura ÚNICA.
    """
    print("\nIniciando normalización de tablas (OPCIÓN 1)...")
    
    # --- 1. Obtener Facturas Únicas ---
    # Tomamos la primera aparición de cada 'no_factura' para definir al cliente
    print("Identificando facturas únicas para crear clientes...")
    df_facturas_unicas = df_limpio.drop_duplicates(subset=['no_factura'], keep='first').reset_index(drop=True)
    
    # Crear el ID de Cliente Sintético (cliente_1, cliente_2...)
    # Habrá un cliente por cada factura única
    df_facturas_unicas['id_cliente'] = [f'cliente_{i+1}' for i in range(len(df_facturas_unicas))]
    
    # --- 2. Crear Tabla CLIENTE ---
    # Contiene una fila por cada cliente único (que es uno por factura)
    print("Creando tabla 'clientes'...")
    # Asegurarse de que todas las columnas de atributos existan
    columnas_cliente_presentes = ['id_cliente'] + [col for col in COLUMNAS_ATRIBUTOS_CLIENTE if col in df_facturas_unicas.columns]
    df_clientes = df_facturas_unicas[columnas_cliente_presentes]
    
    ruta_clientes = os.path.join(carpeta_salida, 'clientes.csv')
    df_clientes.to_csv(ruta_clientes, index=False, encoding='utf-8-sig')
    print(f"Tabla 'clientes.csv' guardada con {len(df_clientes)} clientes únicos (uno por factura).")

    # --- 3. Crear Tabla FACTURA ---
    # Contiene una fila única por factura, con el 'id_cliente' correspondiente
    print("Creando tabla 'facturas'...")
    # Asegurarse de que todas las columnas de factura existan
    columnas_factura_presentes = [col for col in COLUMNAS_FACTURA_FINAL if col in df_facturas_unicas.columns]
    df_facturas = df_facturas_unicas[columnas_factura_presentes]
    
    ruta_facturas = os.path.join(carpeta_salida, 'facturas.csv')
    df_facturas.to_csv(ruta_facturas, index=False, encoding='utf-8-sig')
    print(f"Tabla 'facturas.csv' guardada con {len(df_facturas)} facturas únicas.")

    # --- 4. Crear Tabla PROVEEDOR_FACTURA ---
    # Esta tabla usa el df_limpio COMPLETO para encontrar todas las relaciones proveedor-factura
    print("Creando tabla 'proveedores_por_factura'...")
     # Asegurarse de que todas las columnas de proveedor existan
    columnas_proveedor_presentes = [col for col in COLUMNAS_PROVEEDOR_FACTURA if col in df_limpio.columns]
    df_proveedores_factura = df_limpio[columnas_proveedor_presentes].drop_duplicates().dropna(subset=['no_factura', 'nombre_proveedor'])
    
    ruta_proveedores = os.path.join(carpeta_salida, 'proveedores_por_factura.csv')
    df_proveedores_factura.to_csv(ruta_proveedores, index=False, encoding='utf-8-sig')
    print(f"Tabla 'proveedores_por_factura.csv' guardada con {len(df_proveedores_factura)} registros de proveedores.")
    
# -----------------------------------------------------------------------------
# 3. EJECUCIÓN PRINCIPAL
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Define la carpeta donde están tus CSVs brutos
    CARPETA_DATOS_ENTRADA = 'datos_csv'
    
    # Define la carpeta donde se guardarán los 3 CSVs limpios
    CARPETA_DATOS_SALIDA = 'datos_limpios'
    
    # Crear la carpeta de salida si no existe
    os.makedirs(CARPETA_DATOS_SALIDA, exist_ok=True)
    
    # --- PASO 1: Cargar y Consolidar ---
    df_bruto = cargar_y_consolidar(CARPETA_DATOS_ENTRADA)
    
    if df_bruto is not None:
        # --- PASO 2: Limpiar Datos ---
        df_limpio = limpiar_datos(df_bruto)
        
        # --- PASO 3: Crear Tablas Normalizadas ---
        crear_tablas_normalizadas(df_limpio, CARPETA_DATOS_SALIDA)
        
        print("\n" + "="*30)
        print("¡PROCESO DE NORMALIZACIÓN (OPCIÓN 1) COMPLETADO!")
        print(f"Tus 3 archivos CSV limpios están en la carpeta: '{CARPETA_DATOS_SALIDA}'")
        print("="*30)
    else:
        print("El proceso no pudo continuar porque no se cargaron datos.")

