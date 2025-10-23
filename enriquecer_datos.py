import pandas as pd
import os
import numpy as np
import unicodedata
import re
import time # Para medir tiempo

# --- IMPORTANTE: Instalación de nuevas librerías ---
# Este script necesita librerías GEO. Antes de ejecutar,
# abre tu terminal y corre:
# pip install pycountry geonamescache tqdm
# (tqdm es opcional, para barra de progreso)
# ----------------------------------------------------
try:
    import pycountry
    from geonamescache import GeonamesCache
    print("Librerías 'pycountry' y 'geonamescache' importadas correctamente.")
except ImportError:
    print("Error: Faltan librerías. Por favor, instala 'pycountry' y 'geonamescache'.")
    print("Corre en tu terminal: pip install pycountry geonamescache")
    exit()

# -----------------------------------------------------------------------------
# 1. DEFINICIONES DE CLASIFICACIÓN
# -----------------------------------------------------------------------------

# --- A. Clasificación de Regiones de Colombia ---
MAPEO_REGIONES_COLOMBIA = {
    # Región Caribe
    'COSTA': 'CARIBE', 'CARTAGENA': 'CARIBE', 'SANTA MARTA': 'CARIBE',
    'BARRANQUILLA': 'CARIBE',

    # Región Andina
    'BOGOTA Y ALREDEDORES': 'ANDINA', 'BOGOTA': 'ANDINA', 'ANTIOQUIA': 'ANDINA',
    'MEDELLIN': 'ANDINA', 'EJE CAFETERO': 'ANDINA', 'SANTANDERES': 'ANDINA',
    'BOYACA': 'ANDINA', 'CUNDINAMARCA': 'ANDINA', 'TOLIMA': 'ANDINA',
    'HUILA': 'ANDINA', 'CALI': 'ANDINA', 'SANTANDER': 'ANDINA', #Añadido

    # Región Pacífico
    'SUROCCIDENTE': 'PACIFICO', 'CHOCO': 'PACIFICO',

    # Región Orinoquía
    'LLANOS ORIENTALES': 'ORINOQUIA', 'LLANOS': 'ORINOQUIA',

    # Región Amazonía
    'AMAZONAS': 'AMAZONIA', 'AMAZONIA': 'AMAZONIA', 'LETICIA': 'AMAZONIA',

    # Región Insular
    'SAN ANDRES': 'INSULAR', 'SAN ANDRES ISLAS': 'INSULAR',

    # Valores nulos o no informados
    # NAN y '' se manejarán directamente como Nulo en el código.
    'NO INFORMA': None,
    'DESCONOCIDO': None,
}

# --- B. Clasificación Geográfica de Destinos (Mundial) ---
print("Inicializando bases de datos geográficas (esto puede tardar unos segundos)...")
start_time_geo_init = time.time()
gc = GeonamesCache()
CITIES = gc.get_cities()
COUNTRIES = gc.get_countries()
CONTINENTS = gc.get_continents()

def normalize_geo_name(name):
    """Convierte a mayúsculas, quita tildes y caracteres especiales para búsqueda."""
    if pd.isna(name) or not isinstance(name, str) or name.strip() == '':
        return None # Devolver None para nulos o vacíos
    try:
        # NFD separa tildes, encode/decode las quita
        normalized = unicodedata.normalize('NFD', name.upper())
        # Eliminar caracteres no alfanuméricos excepto espacios
        normalized = re.sub(r'[^\w\s]', '', normalized)
        # Reemplazar múltiples espacios con uno solo y quitar espacios al inicio/final
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        # Manejar casos especiales que quedan después de normalizar
        if normalized == 'BOGOTA D C': normalized = 'BOGOTA'
        # Si después de limpiar queda vacío, retornar None
        return normalized if normalized else None
    except Exception:
        return None # Retornar None si hay error en normalización

# Crear diccionarios de búsqueda rápida (normalizados)
CITIES_LOOKUP = {}
for city_id, city_data in CITIES.items():
    norm_name = normalize_geo_name(city_data.get('name', ''))
    if norm_name: CITIES_LOOKUP[norm_name] = city_data
    alt_names = city_data.get('alternatenames', [])
    if isinstance(alt_names, list):
        for alt in alt_names:
            norm_alt = normalize_geo_name(alt)
            # Solo añadir si no existe ya Y no es un número Y tiene más de 2 caracteres
            if norm_alt and norm_alt not in CITIES_LOOKUP and not norm_alt.isdigit() and len(norm_alt) > 2:
                CITIES_LOOKUP[norm_alt] = city_data

COUNTRIES_LOOKUP = {}
COUNTRY_ISO_LOOKUP = {}
for country_code, country_data in COUNTRIES.items():
    norm_name = normalize_geo_name(country_data.get('name', ''))
    iso_code = country_data.get('iso')
    if norm_name: COUNTRIES_LOOKUP[norm_name] = country_data
    if iso_code: COUNTRY_ISO_LOOKUP[iso_code] = country_data.get('name')
    try:
        pyc_country = pycountry.countries.get(alpha_2=iso_code)
        if pyc_country and hasattr(pyc_country, 'alpha_3'):
            norm_a3 = normalize_geo_name(pyc_country.alpha_3)
            if norm_a3 and norm_a3 not in COUNTRIES_LOOKUP:
                COUNTRIES_LOOKUP[norm_a3] = country_data
    except Exception: pass

COUNTRIES_LOOKUP['ESTADOS UNIDOS'] = COUNTRIES_LOOKUP.get('UNITED STATES')
COUNTRIES_LOOKUP['USA'] = COUNTRIES_LOOKUP.get('UNITED STATES')
COUNTRIES_LOOKUP['EEUU'] = COUNTRIES_LOOKUP.get('UNITED STATES')
COUNTRIES_LOOKUP['EEU'] = COUNTRIES_LOOKUP.get('UNITED STATES')
COUNTRIES_LOOKUP['EESTADOS UNIDOS'] = COUNTRIES_LOOKUP.get('UNITED STATES')
COUNTRIES_LOOKUP['REINO UNIDO'] = COUNTRIES_LOOKUP.get('UNITED KINGDOM')
COUNTRIES_LOOKUP['UK'] = COUNTRIES_LOOKUP.get('UNITED KINGDOM')
COUNTRIES_LOOKUP['PAISES BAJOS'] = COUNTRIES_LOOKUP.get('NETHERLANDS')
COUNTRIES_LOOKUP['NUEVA ZELANDA'] = COUNTRIES_LOOKUP.get('NEW ZEALAND')
COUNTRIES_LOOKUP['NEW ZELANDA'] = COUNTRIES_LOOKUP.get('NEW ZEALAND')

print(f"Bases de datos GEO inicializadas en {time.time() - start_time_geo_init:.2f} segundos.")

# Formato: ('TIPO', 'Ciudad/Nombre Estandarizado', 'País Oficial', 'Continente Oficial')
MAPEO_DESTINOS_MANUAL = {
    # Conceptos/Tours/Abstractos -> Tipo específico, Nombre en Ciudad, País/Cont Nulos
    'PLAN ORLANDO': ('TOUR', 'PLAN ORLANDO', 'United States', 'North America'), # Modificado
    'PLAN A LADRILLEROS': ('TOUR', 'PLAN A LADRILLEROS', 'Colombia', 'South America'), # Modificado
    'MEDELLIN NUQUI': ('TOUR', 'MEDELLIN NUQUI', 'Colombia', 'South America'), # Modificado
    'MEDELLINEJE CAFETER': ('TOUR', 'MEDELLIN Y EJE CAFETERO', 'Colombia', 'South America'), # Modificado
    'CRUCERO POR ELC ARIB': ('CRUCERO', 'CRUCERO POR EL CARIBE', None, 'CARIBE'), # Añadido
    'CRUCEOR POR EL CARIB': ('CRUCERO', 'CRUCERO POR EL CARIBE', None, 'CARIBE'), # Añadido
    'CRUCER POR EL CARIBE': ('CRUCERO', 'CRUCERO POR EL CARIBE', None, 'CARIBE'), # Añadido
    'CRUCERO CARIBE': ('CRUCERO', 'CRUCERO POR EL CARIBE', None, 'CARIBE'),
    'CRUCERO POR EL CARIB': ('CRUCERO', 'CRUCERO POR EL CARIBE', None, 'CARIBE'),
    'CRCUERO POR EL CARIB': ('CRUCERO', 'CRUCERO POR EL CARIBE', None, 'CARIBE'),
    'CRUCERO POR E CARIBE': ('CRUCERO', 'CRUCERO POR EL CARIBE', None, 'CARIBE'),
    'CRUCERO POR EL CARI': ('CRUCERO', 'CRUCERO POR EL CARIBE', None, 'CARIBE'),
    'CRUERO POR EL CARIBE': ('CRUCERO', 'CRUCERO POR EL CARIBE', None, 'CARIBE'),
    'CRUCEO POR EL CARIBE': ('CRUCERO', 'CRUCERO POR EL CARIBE', None, 'CARIBE'),
    'CRUCERO POE EL CARIB': ('CRUCERO', 'CRUCERO POR EL CARIBE', None, 'CARIBE'),
    'RUCERO POR EL CARIBE': ('CRUCERO', 'CRUCERO POR EL CARIBE', None, 'CARIBE'),
    'CRUCERO POR ELC ARIB': ('CRUCERO', 'CRUCERO POR EL CARIBE', None, 'CARIBE'), # NUEVO
    'CRUCEOR POR EL CARIB': ('CRUCERO', 'CRUCERO POR EL CARIBE', None, 'CARIBE'), # NUEVO
    'CRUCER POR EL CARIBE': ('CRUCERO', 'CRUCERO POR EL CARIBE', None, 'CARIBE'), # NUEVO
    'CRUCERO': ('CRUCERO', 'CRUCERO', None, None),
    'CURCERO': ('CRUCERO', 'CRUCERO', None, None),
    'CRCUERO': ('CRUCERO', 'CRUCERO', None, None), # NUEVO
    'CIRCUITO EUROPA': ('TOUR', 'CIRCUITO EUROPA', None, 'EUROPA'),
    'PLAN EUROPA': ('TOUR', 'PLAN EUROPA', None, 'EUROPA'),
    'PLAN EJE CAFETERO': ('TOUR', 'PLAN EJE CAFETERO', 'Colombia', 'South America'),
    'INTERNACIONAL': ('CONCEPTO', 'INTERNACIONAL', None, None),
    'INTERNAICONAL': ('CONCEPTO', 'INTERNACIONAL', None, None),
    'INETRNACIONAL': ('CONCEPTO', 'INTERNACIONAL', None, None),
    'INTERNACION': ('CONCEPTO', 'INTERNACIONAL', None, None),
    'ITERNACIONAL': ('CONCEPTO', 'INTERNACIONAL', None, None),
    'INTEERNACIONAL': ('CONCEPTO', 'INTERNACIONAL', None, None),
    'INTERN': (None, None, None, None), # NUEVO - BORRAR
    'NAICONAL': ('CONCEPTO', 'NACIONAL', 'Colombia', 'South America'),
    'NACONAL': ('CONCEPTO', 'NACIONAL', 'Colombia', 'South America'),
    'CIRCUITOS SURAMERICA': ('TOUR', 'CIRCUITOS SURAMERICA', None, 'South America'),
    'CIRCUITO DEL SUR': ('TOUR', 'CIRCUITO DEL SUR', None, 'South America'),
    'SUR DE COLOMBIA': ('TOUR', 'SUR DE COLOMBIA', 'Colombia', 'South America'),
    'SUR COLOMBIA': ('TOUR', 'SUR DE COLOMBIA', 'Colombia', 'South America'),
    'SUR DE COLMBIA': ('TOUR', 'SUR DE COLOMBIA', 'Colombia', 'South America'),
    'TERMALES': ('SITIO TURISTICO', 'TERMALES', 'Colombia', 'South America'),
    'TERMALES SV': ('SITIO TURISTICO', 'TERMALES SAN VICENTE', 'Colombia', 'South America'),
    'TERMALES OTONO': ('SITIO TURISTICO', 'TERMALES EL OTOÑO', 'Colombia', 'South America'),
    'PANACA': ('SITIO TURISTICO', 'PANACA', 'Colombia', 'South America'),
    'PARQUE DEL CAFE': ('SITIO TURISTICO', 'PARQUE DEL CAFE', 'Colombia', 'South America'),
    'PARCAFE': ('SITIO TURISTICO', 'PARQUE DEL CAFE', 'Colombia', 'South America'),
    'PARQUE CAFFE': ('SITIO TURISTICO', 'PARQUE DEL CAFE', 'Colombia', 'South America'), # NUEVO
    'CONSOTA': ('SITIO TURISTICO', 'PARQUE CONSOTA', 'Colombia', 'South America'),
    'UKUMARI': ('SITIO TURISTICO', 'UKUMARI', 'Colombia', 'South America'),
    'UKUMARIA': ('SITIO TURISTICO', 'UKUMARI', 'Colombia', 'South America'),
    'CANO CRISTALES': ('SITIO TURISTICO', 'CAÑO CRISTALES', 'Colombia', 'South America'),
    'CANO CRISTAL': ('SITIO TURISTICO', 'CAÑO CRISTALES', 'Colombia', 'South America'),
    'ISLA DEL ROSARIO': ('ISLA', 'ISLAS DEL ROSARIO', 'Colombia', 'South America'),
    'ISLAS DEL ROSARIO': ('ISLA', 'ISLAS DEL ROSARIO', 'Colombia', 'South America'),
    'ISLA COCOLISO': ('ISLA', 'ISLA COCOLISO (ROSARIO)', 'Colombia', 'South America'),
    'COCOLISO': ('ISLA', 'ISLA COCOLISO (ROSARIO)', 'Colombia', 'South America'),
    'ISLA DEL ENCANTO': ('ISLA', 'ISLA DEL ENCANTO (ROSARIO)', 'Colombia', 'South America'),
    'ISLA MUCURA': ('ISLA', 'ISLA MUCURA (SAN BERNARDO)', 'Colombia', 'South America'),
    'MUCURA': ('ISLA', 'ISLA MUCURA (SAN BERNARDO)', 'Colombia', 'South America'),
    'ISLA PALMA': ('ISLA', 'ISLA PALMA (SAN BERNARDO)', 'Colombia', 'South America'),
    'DISNEY': ('SITIO TURISTICO', 'DISNEY WORLD', 'United States', 'North America'),
    'SEAWORLD': ('SITIO TURISTICO', 'SEAWORLD ORLANDO', 'United States', 'North America'),
    'HACIENDA NAPOLES': ('SITIO TURISTICO', 'HACIENDA NAPOLES', 'Colombia', 'South America'),
    'ANTILLAS': ('REGION GEO', 'ANTILLAS', None, 'CARIBE'),
    'CRUCERO POR ANTILLAS': ('CRUCERO', 'CRUCERO POR ANTILLAS', None, 'CARIBE'),
    'CRUCERO ANTILLAS': ('CRUCERO', 'CRUCERO POR ANTILLAS', None, 'CARIBE'),
    'CRUCERO ANTILLAS Y C': ('CRUCERO', 'CRUCERO ANTILLAS Y CARIBE SUR', None, 'CARIBE'),
    'ISLAS GRIEGAS': ('REGION GEO', 'ISLAS GRIEGAS', 'Greece', 'Europe'),
    'CRUCERO POR EL MEDIT': ('CRUCERO', 'CRUCERO POR EL MEDITERRANEO', None, 'EUROPA'),
    'COSTA ATLANTICA': ('REGION COL', 'COSTA ATLANTICA', 'Colombia', 'South America'),
    'TAMBOS DEL CARIBE': ('HOTEL/LUGAR', 'TAMBOS DEL CARIBE', 'Colombia', 'South America'),
    'LOS TAMBOS': ('HOTEL/LUGAR', 'LOS TAMBOS', 'Colombia', 'South America'),
    'BOSQUE SEL SAMAN': ('HOTEL/LUGAR', 'BOSQUES DEL SAMAN', 'Colombia', 'South America'),
    'BOSQUE DEL SAMAN': ('HOTEL/LUGAR', 'BOSQUES DEL SAMAN', 'Colombia', 'South America'),
    'BOSQUES DEL SAMAN': ('HOTEL/LUGAR', 'BOSQUES DEL SAMAN', 'Colombia', 'South America'),
    'CATAMARAN': ('CONCEPTO', 'CATAMARAN', None, None),
    'PALMARENA': ('HOTEL/LUGAR', 'PALMARENA', 'Colombia', 'South America'),
    'CALIMA': ('REGION COL', 'LAGO CALIMA', 'Colombia', 'South America'),
    'LAGO CALIMA': ('REGION COL', 'LAGO CALIMA', 'Colombia', 'South America'),
    'LAGOS CALIMA': ('REGION COL', 'LAGO CALIMA', 'Colombia', 'South America'),
    'RIU BAMBU': ('HOTEL/LUGAR', 'HOTEL RIU BAMBU', 'Dominican Republic', 'North America'),
    'RIU PAAMA PLAYA': ('HOTEL/LUGAR', 'HOTEL RIU', 'Dominican Republic', 'North America'),
    'RIVERA MAYA': ('REGION GEO', 'RIVIERA MAYA', 'Mexico', 'North America'),
    'RECEPTIVOS': ('CONCEPTO', 'RECEPTIVOS', None, None),
    'NAVIERA': ('CONCEPTO', 'NAVIERA', None, None),
    'PLAYA BLANCA': ('LUGAR', 'PLAYA BLANCA', 'Colombia', 'South America'),
    'JFK': ('AEROPUERTO', 'JFK Airport', 'United States', 'North America'),
    'CDG': ('AEROPUERTO', 'CDG Airport', 'France', 'Europe'),
    'RANCHO TOTA': ('HOTEL/LUGAR', 'RANCHO TOTA', 'Colombia', 'South America'),
    'SANSIRAKA': ('HOTEL/LUGAR', 'HOTEL SANSIRAKA', 'Colombia', 'South America'),
    'MAGUIPI': ('LUGAR', 'PLAYA MAGÜIPI', 'Colombia', 'South America'),
    'FINCA LA ESMERALDA': ('HOTEL/LUGAR', 'FINCA LA ESMERALDA', 'Colombia', 'South America'),
    'CRUCE ANDINO': ('TOUR', 'CRUCE ANDINO', None, 'South America'),
    'HOSTERIA REAL': ('HOTEL/LUGAR', 'HOSTERIA REAL', None, None), #Ambiguo
    'WYNDHAM': ('HOTEL/LUGAR', 'HOTEL WYNDHAM', None, None), #Ambiguo
    'CARTAGENA PLAZA': ('HOTEL/LUGAR', 'HOTEL CARTAGENA PLAZA', 'Colombia', 'South America'),
    'DECAMERON CINCO HERR': ('HOTEL/LUGAR', 'DECAMERON PANACA', 'Colombia', 'South America'),
    'HOTEL MEDELLIN': ('HOTEL/LUGAR', 'HOTEL MEDELLIN', 'Colombia', 'South America'),
    'DESTINO HOTEL ALIMEN': ('OTRO', 'DESTINO HOTEL ALIMEN', None, None),
    'SAN SILVESTRE': ('OTRO', 'SAN SILVESTRE', None, None),
    'HERNANDEZ HURTADO': ('OTRO', 'HERNANDEZ HURTADO', None, None),
    'TRIANGULO DEL ESTE': ('TOUR', 'TRIANGULO DEL ESTE USA', 'United States', 'North America'),
    'LAS CAMELIAS': ('HOTEL/LUGAR', 'HOTEL LAS CAMELIAS', 'Colombia', 'South America'),
    'EL RANCHO': ('OTRO', 'EL RANCHO', None, None),
    'LANTOURS': ('OTRO', 'LANTOURS', None, None),
    'ARHUCO': ('HOTEL/LUGAR', 'ARHUCO', 'Colombia', 'South America'),
    'AGUSTIN': ('OTRO', 'AGUSTIN', None, None),
    'LAGOTOURS': ('OTRO', 'LAGOTOURS', None, None),
    'SERVINCLUIDOS': ('CONCEPTO', 'SERVICIOS INCLUIDOS', None, None),
    'MONARCH': (None, None, None, None), # NUEVO - BORRAR
    'ENERO': (None, None, None, None), # NUEVO - BORRAR
    'DIRADAL': (None, None, None, None), # NUEVO - BORRAR
    'PAILITA': (None, None, None, None), # NUEVO - BORRAR
    'TURQUIA Y DUABI': (None, None, None, None), # NUEVO - BORRAR
    'PASADIA ZAPATOCA': (None, None, None, None), # NUEVO - BORRAR
    'LOMAS': (None, None, None, None), # NUEVO - BORRAR
    'TA EN CRUCERO POR EL': (None, None, None, None), # NUEVO - BORRAR
    'CENTRO AMERICA': (None, None, None, None), # NUEVO - BORRAR

    # Continentes y Regiones Geo Amplias
    'EUROPA': ('CONTINENTE', None, None, 'EUROPA'),
    'ERUOPA': ('CONTINENTE', None, None, 'EUROPA'), # Añadido
    'ASIA': ('CONTINENTE', None, None, 'ASIA'),
    'SUR AMERICA': ('CONTINENTE', None, None, 'South America'),
    'AMERICA': ('CONTINENTE', None, None, 'AMERICA'),
    'EUORPA': ('CONTINENTE', None, None, 'EUROPA'),
    'EUROA': ('CONTINENTE', None, None, 'EUROPA'),
    'EUROPEA': ('CONTINENTE', None, None, 'EUROPA'),
    'ERUOPA': ('CONTINENTE', None, None, 'EUROPA'), # NUEVO
    'SURAMERICA': ('CONTINENTE', None, None, 'South America'),
    'SUDAMERICA': ('CONTINENTE', None, None, 'South America'),
    'ANDINA': ('REGION COL', 'ANDINA', 'Colombia', 'South America'),
    'TIERRA SANTA': ('REGION GEO', 'TIERRA SANTA', None, 'Asia'), # NUEVO

    # Países (Nombres oficiales de geonames y correcciones)
    'TURQIA': ('PAIS', None, 'Turkey', 'Asia'), # Añadido
    'CUZACAO': ('PAIS', None, 'Curacao', 'South America'), # Añadido
    'PANAMMA': ('PAIS', None, 'Panama', 'North America'), # Añadido
    'PIERTO RICO': ('PAIS', None, 'Puerto Rico', 'North America'), # Añadido
    'SUECIA': ('PAIS', None, 'Sweden', 'Europe'), # Añadido
    'PLAN PERU': ('PAIS', None, 'Peru', 'South America'), # Añadido
    'PERU': ('PAIS', None, 'Peru', 'South America'),
    'PLAN PERU': ('PAIS', None, 'Peru', 'South America'), # NUEVO
    'TURQUIA': ('PAIS', None, 'Turkey', 'Asia'),
    'TURKEEY': ('PAIS', None, 'Turkey', 'Asia'),
    'TURQIA': ('PAIS', None, 'Turkey', 'Asia'), # NUEVO
    'ESTADOS UNIDOS': ('PAIS', None, 'United States', 'North America'),
    'USA': ('PAIS', None, 'United States', 'North America'),
    'EEUU': ('PAIS', None, 'United States', 'North America'),
    'EEU': ('PAIS', None, 'United States', 'North America'),
    'EESTADOS UNIDOS': ('PAIS', None, 'United States', 'North America'),
    'MEXICO': ('PAIS', None, 'Mexico', 'North America'),
    'MAEXICO': ('PAIS', None, 'Mexico', 'North America'),
    'RUSIA': ('PAIS', None, 'Russia', 'Europe'),
    'COLOMBIA': ('PAIS', None, 'Colombia', 'South America'),
    'PANAMA': ('PAIS', None, 'Panama', 'North America'),
    'PANAMMA': ('PAIS', None, 'Panama', 'North America'), # NUEVO
    'REPUBLICA DOMINICANA': ('PAIS', None, 'Dominican Republic', 'North America'),
    'DOMINICANA': ('PAIS', None, 'Dominican Republic', 'North America'),
    'REPUBLICA DOMIINICAN': ('PAIS', None, 'Dominican Republic', 'North America'),
    'REPUBLICADOMINICANA': ('PAIS', None, 'Dominican Republic', 'North America'),
    'REPUBLICA DOMICANA': ('PAIS', None, 'Dominican Republic', 'North America'), # NUEVO
    'ESPANA': ('PAIS', None, 'Spain', 'Europe'),
    'BRASIL': ('PAIS', None, 'Brazil', 'South America'),
    'ARGENTINA': ('PAIS', None, 'Argentina', 'South America'),
    'CUBA': ('PAIS', None, 'Cuba', 'North America'),
    'CURAZAO': ('PAIS', None, 'Curacao', 'South America'),
    'CURACOA': ('PAIS', None, 'Curacao', 'South America'),
    'CUARCAO': ('PAIS', None, 'Curacao', 'South America'),
    'CUZACAO': ('PAIS', None, 'Curacao', 'South America'), # NUEVO
    'ARUBA': ('PAIS', None, 'Aruba', 'South America'),
    'AURBA': ('PAIS', None, 'Aruba', 'South America'),
    'ALEMANIA': ('PAIS', None, 'Germany', 'Europe'),
    'MARRUECOS': ('PAIS', None, 'Morocco', 'Africa'),
    'EGIPTO': ('PAIS', None, 'Egypt', 'Africa'),
    'EGYPTO': ('PAIS', None, 'Egypt', 'Africa'),
    'ISLANDIA': ('PAIS', None, 'Iceland', 'Europe'),
    'EL SALVADOR': ('PAIS', None, 'El Salvador', 'North America'),
    'PAISES BAJOS': ('PAIS', None, 'Netherlands', 'Europe'),
    'NUEVA ZELANDA': ('PAIS', None, 'New Zealand', 'Oceania'),
    'NEW ZELANDA': ('PAIS', None, 'New Zealand', 'Oceania'),
    'LAS BAHAMAS': ('PAIS', None, 'Bahamas', 'North America'),
    'HABAMAS': ('PAIS', None, 'Bahamas', 'North America'),
    'BAHANA': ('PAIS', None, 'Bahamas', 'North America'),
    'NORUEGA': ('PAIS', None, 'Norway', 'Europe'),
    'SUIZA': ('PAIS', None, 'Switzerland', 'Europe'),
    'FRANCIA': ('PAIS', None, 'France', 'Europe'),
    'POLONIA': ('PAIS', None, 'Poland', 'Europe'),
    'JORDANIA': ('PAIS', None, 'Jordan', 'Asia'),
    'SUDAFRICA': ('PAIS', None, 'South Africa', 'Africa'),
    'BELGICA': ('PAIS', None, 'Belgium', 'Europe'),
    'EMIRATOS ARABES': ('PAIS', None, 'United Arab Emirates', 'Asia'),
    'JAPON': ('PAIS', None, 'Japan', 'Asia'),
    'IRLANDA': ('PAIS', None, 'Ireland', 'Europe'),
    'CROACIA': ('PAIS', None, 'Croatia', 'Europe'),
    'GRECIA': ('PAIS', None, 'Greece', 'Europe'),
    'GRECEE': ('PAIS', None, 'Greece', 'Europe'),
    'SUECIA': ('PAIS', None, 'Sweden', 'Europe'), # NUEVO
    'PIERTO RICO': ('PAIS', None, 'Puerto Rico', 'North America'), # NUEVO

    # Ciudades y Lugares Específicos Internacionales
    'ISLAS MARGARITA': ('ISLA', 'ISLA MARGARITA', 'Venezuela', 'South America'), # Añadido
    'PUNTA DEL ESTE': ('CIUDAD', 'Punta del Este', 'Uruguay', 'South America'), # Añadido
    'CUIBA': ('CIUDAD', 'Cuiabá', 'Brazil', 'South America'), # Añadido
    'USHIAIA': ('CIUDAD', 'Ushuaia', 'Argentina', 'South America'), # Añadido
    'GRAND TURK': ('ISLA', 'Grand Turk', 'Turks and Caicos Islands', 'North America'), # Añadido
    'CAUCUN': ('CIUDAD', 'Cancun', 'Mexico', 'North America'), # Añadido
    'PAYA DEL CARMEN': ('CIUDAD', 'Playa del Carmen', 'Mexico', 'North America'), # Añadido
    'SAU PAULO': ('CIUDAD', 'São Paulo', 'Brazil', 'South America'), # Añadido
    'ROATAN': ('ISLA', 'Roatán', 'Honduras', 'North America'), # Añadido
    'AMAMI O SHIMA': ('ISLA', 'Amami Ōshima', 'Japan', 'Asia'), # Añadido
    'CANUCUN': ('CIUDAD', 'Cancun', 'Mexico', 'North America'), # Añadido
    'CANDUN': ('CIUDAD', 'Cancun', 'Mexico', 'North America'), # Añadido
    'TULSAN': ('CIUDAD', 'Tulsa', 'United States', 'North America'), # Añadido
    'SANTIAGO CHILE': ('CIUDAD', 'Santiago', 'Chile', 'South America'), # Añadido
    'MIAM': ('CIUDAD', 'Miami', 'United States', 'North America'), # Añadido
    'CHICEN ITZA': ('SITIO TURISTICO', 'Chichen Itza', 'Mexico', 'North America'), # Añadido
    'LYON FRANCE': ('CIUDAD', 'Lyon', 'France', 'Europe'), # Añadido
    'SAN PETERBURGO': ('CIUDAD', 'Saint Petersburg', 'Russia', 'Europe'), # Añadido
    'PUNTA CANA': ('CIUDAD', 'Punta Cana', 'Dominican Republic', 'North America'),
    'PIUNTA CANA': ('CIUDAD', 'Punta Cana', 'Dominican Republic', 'North America'),
    'PLAN PUNTA CANA': ('CIUDAD', 'Punta Cana', 'Dominican Republic', 'North America'), # NUEVO
    'CANCUN': ('CIUDAD', 'Cancun', 'Mexico', 'North America'),
    'CANUCN': ('CIUDAD', 'Cancun', 'Mexico', 'North America'),
    'CANCN': ('CIUDAD', 'Cancun', 'Mexico', 'North America'),
    'CACNUN': ('CIUDAD', 'Cancun', 'Mexico', 'North America'),
    'CANCUBN': ('CIUDAD', 'Cancun', 'Mexico', 'North America'),
    'MIAMI': ('CIUDAD', 'Miami', 'United States', 'North America'),
    'MAIMI': ('CIUDAD', 'Miami', 'United States', 'North America'),
    'MIAM': ('CIUDAD', 'Miami', 'United States', 'North America'), # NUEVO
    'PANAMA CITY': ('CIUDAD', 'Panama City', 'Panama', 'North America'),
    'CIUDAD DE PANAMA': ('CIUDAD', 'Panama City', 'Panama', 'North America'),
    'PANANA CITY': ('CIUDAD', 'Panama City', 'Panama', 'North America'),
    'MSPANAMA CITY': ('OTRO', 'MSPANAMA CITY', None, None), # Caso raro, mantener OTRO
    'MEXICO CITY': ('CIUDAD', 'Mexico City', 'Mexico', 'North America'),
    'CIUDAD DE MEXICO': ('CIUDAD', 'Mexico City', 'Mexico', 'North America'),
    'MOSCU': ('CIUDAD', 'Moscow', 'Russia', 'Europe'),
    'MADRID': ('CIUDAD', 'Madrid', 'Spain', 'Europe'),
    'MADIRD': ('CIUDAD', 'Madrid', 'Spain', 'Europe'),
    'LIMA': ('CIUDAD', 'Lima', 'Peru', 'South America'),
    'BUENOS AIRES': ('CIUDAD', 'Buenos Aires', 'Argentina', 'South America'),
    'ORLANDO': ('CIUDAD', 'Orlando', 'United States', 'North America'),
    'ORLANDON': ('CIUDAD', 'Orlando', 'United States', 'North America'),
    'NUEVA YORK': ('CIUDAD', 'New York', 'United States', 'North America'),
    'NEW YORK': ('CIUDAD', 'New York', 'United States', 'North America'),
    'VARADERO': ('CIUDAD', 'Varadero', 'Cuba', 'North America'),
    'LA HABANA': ('CIUDAD', 'Havana', 'Cuba', 'North America'),
    'HAVANA': ('CIUDAD', 'Havana', 'Cuba', 'North America'),
    'CUZCO': ('CIUDAD', 'Cusco', 'Peru', 'South America'),
    'DUBAI': ('CIUDAD', 'Dubai', 'United Arab Emirates', 'Asia'),
    'IGUAZU': ('ATRACCION', 'Cataratas del Iguazú', None, 'South America'),
    'IGUEZU': ('ATRACCION', 'Cataratas del Iguazú', None, 'South America'),
    'LONDRES': ('CIUDAD', 'London', 'United Kingdom', 'Europe'),
    'ESTAMBUL': ('CIUDAD', 'Istanbul', 'Turkey', 'Asia'),
    'ESTAMBULL': ('CIUDAD', 'Istanbul', 'Turkey', 'Asia'),
    'ESTAMBUEL': ('CIUDAD', 'Istanbul', 'Turkey', 'Asia'),
    'FRANKFURT': ('CIUDAD', 'Frankfurt am Main', 'Germany', 'Europe'),
    'RANKFURT': ('CIUDAD', 'Frankfurt am Main', 'Germany', 'Europe'),
    'FRANFURT': ('CIUDAD', 'Frankfurt am Main', 'Germany', 'Europe'),
    'BARILOCHE': ('CIUDAD', 'San Carlos de Bariloche', 'Argentina', 'South America'),
    'SAN CARLOS BARILOCHE': ('CIUDAD', 'San Carlos de Bariloche', 'Argentina', 'South America'),
    'SEDONA': ('CIUDAD', 'Sedona', 'United States', 'North America'),
    'TANANARIVE': ('CIUDAD', 'Antananarivo', 'Madagascar', 'Africa'),
    'FLORIANAPOLIS': ('CIUDAD', 'Florianópolis', 'Brazil', 'South America'),
    'ATACAMES': ('CIUDAD', 'Atacames', 'Ecuador', 'South America'),
    'ACAPULCO': ('CIUDAD', 'Acapulco', 'Mexico', 'North America'),
    'CHICHEN ITZA': ('SITIO TURISTICO', 'Chichen Itza', 'Mexico', 'North America'),
    'CHICEN ITZA': ('SITIO TURISTICO', 'Chichen Itza', 'Mexico', 'North America'), # NUEVO
    'GINEBRA': ('CIUDAD', 'Geneva', 'Switzerland', 'Europe'), # Podria ser ambiguo con Ginebra, Valle
    'PALMA MALLORCA': ('CIUDAD', 'Palma', 'Spain', 'Europe'),
    'MALLORCA': ('ISLA', 'MALLORCA', 'Spain', 'Europe'),
    'LAS PALMAS DE GRAN C': ('CIUDAD', 'Las Palmas de Gran Canaria', 'Spain', 'Europe'),
    'GRAN CANARIA': ('ISLA', 'GRAN CANARIA', 'Spain', 'Europe'),
    'ROMA': ('CIUDAD', 'Rome', 'Italy', 'Europe'),
    'PERTO VALLARTA': ('CIUDAD', 'Puerto Vallarta', 'Mexico', 'North America'),
    'INTERNATIONAL FALLS': ('CIUDAD', 'International Falls', 'United States', 'North America'),
    'ARTHURS TOWN': ("CIUDAD", "Arthur's Town", "Bahamas", "North America"),
    'FORT LAUDERDALE': ('CIUDAD', 'Fort Lauderdale', 'United States', 'North America'),
    'FORTLAUDERDALE': ('CIUDAD', 'Fort Lauderdale', 'United States', 'North America'),
    'FOURT LAUDERDALE': ('CIUDAD', 'Fort Lauderdale', 'United States', 'North America'),
    'LAUDERDALE': ('CIUDAD', 'Fort Lauderdale', 'United States', 'North America'),
    'FORT LAU': ('CIUDAD', 'Fort Lauderdale', 'United States', 'North America'),
    'FORLODER': ('CIUDAD', 'Fort Lauderdale', 'United States', 'North America'),
    'FORT LAUDERDAL': ('CIUDAD', 'Fort Lauderdale', 'United States', 'North America'), # NUEVO
    'ATHENAS': ('CIUDAD', 'Athens', 'Greece', 'Europe'),
    'TUCAMAN': ('CIUDAD', 'San Miguel de Tucumán', 'Argentina', 'South America'),
    'LOS ANEGLES': ('CIUDAD', 'Los Angeles', 'United States', 'North America'),
    'SAO PABLO': ('CIUDAD', 'São Paulo', 'Brazil', 'South America'),
    'RIO JANEIRO': ('CIUDAD', 'Rio de Janeiro', 'Brazil', 'South America'),
    'ESCANABA': ('CIUDAD', 'Escanaba', 'United States', 'North America'),
    'EAST STROUDSBURG': ('CIUDAD', 'East Stroudsburg', 'United States', 'North America'),
    'ZHANGJIANG': ('CIUDAD', 'Zhanjiang', 'China', 'Asia'),
    'ALBURY': ('CIUDAD', 'Albury', 'Australia', 'Oceania'),
    'MT NEWMAN': ('CIUDAD', 'Newman', 'Australia', 'Oceania'),
    'SANTA COLOMA DE GRAM': ('CIUDAD', 'Santa Coloma de Gramenet', 'Spain', 'Europe'),
    'GRAND CANYON': ('SITIO TURISTICO', 'GRAND CANYON', 'United States', 'North America'),
    'SANTIAGO DE LOS CABA': ('CIUDAD', 'Santiago de los Caballeros', 'Dominican Republic', 'North America'),
    'NUKU ALOFA': ("CIUDAD", "Nuku'alofa", "Tonga", "Oceania"),
    'MYKONOS': ('ISLA', 'MYKONOS', 'Greece', 'Europe'),
    'MACHUICHU': ('SITIO TURISTICO', 'Machu Picchu', 'Peru', 'South America'),
    'HOUGHTON': ('CIUDAD', 'Houghton', 'United States', 'North America'), # Asumir USA
    'ISLA MARGARITA': ('ISLA', 'ISLA MARGARITA', 'Venezuela', 'South America'),
    'ISLA MARGARTITA': ('ISLA', 'ISLA MARGARITA', 'Venezuela', 'South America'),
    'ISLAS MARGARITAS': ('ISLA', 'ISLA MARGARITA', 'Venezuela', 'South America'),
    'ISLA MARGA': ('ISLA', 'ISLA MARGARITA', 'Venezuela', 'South America'),
    'CERRO MAGGIORE': ('CIUDAD', 'Cerro Maggiore', 'Italy', 'Europe'),
    'FORT NELSON': ('CIUDAD', 'Fort Nelson', 'Canada', 'North America'),
    'TULSAN': ('CIUDAD', 'Tulsa', 'United States', 'North America'), # NUEVO
    'SANTIAGO CHILE': ('CIUDAD', 'Santiago', 'Chile', 'South America'), # NUEVO
    'LYON FRANCE': ('CIUDAD', 'Lyon', 'France', 'Europe'), # NUEVO
    'SAN PETERBURGO': ('CIUDAD', 'Saint Petersburg', 'Russia', 'Europe'), # NUEVO
    'OXFORT': ('CIUDAD', 'Oxford', 'United Kingdom', 'Europe'), # NUEVO

    # Destinos Nacionales Colombia (Ciudades, Lugares, Regiones)
   'EJE CAFETER': ('REGION COL', 'Eje Cafetero', 'Colombia', 'South America'), # Añadido
    'ALCALA': ('CIUDAD', 'Alcalá', 'Colombia', 'South America'), # Añadido
    'TAGANGA': ('LUGAR', 'Taganga', 'Colombia', 'South America'), # Añadido (Corregimiento)
    'TAYRONA': ('SITIO TURISTICO', 'PARQUE TAYRONA', 'Colombia', 'South America'), # Añadido
    'LE TEBAIDA': ('CIUDAD', 'La Tebaida', 'Colombia', 'South America'), # Añadido
    'MENGAR': ('CIUDAD', 'Melgar', 'Colombia', 'South America'), # Añadido
    'LAGO DE TOTA': ('LUGAR', 'LAGO DE TOTA', 'Colombia', 'South America'), # Añadido
    'CUCUCTA': ('CIUDAD', 'Cúcuta', 'Colombia', 'South America'), # Añadido
    'BOYACBA': ('REGION COL', 'BOYACA', 'Colombia', 'South America'), # Añadido
    'EJE CAFTERO': ('REGION COL', 'Eje Cafetero', 'Colombia', 'South America'), # Añadido
    'SAN ANTENO': ('CIUDAD', 'San Antero', 'Colombia', 'South America'), # Añadido
    'MELGRA': ('CIUDAD', 'Melgar', 'Colombia', 'South America'), # Añadido
    'EJE C AFETERO': ('REGION COL', 'Eje Cafetero', 'Colombia', 'South America'), # Añadido
    'SAN ANDDES': ('CIUDAD', 'San Andrés', 'Colombia', 'South America'), # Añadido
    'SANTA MARTA E': ('CIUDAD', 'Santa Marta', 'Colombia', 'South America'), # Añadido
    'SAN ANDRSE': ('CIUDAD', 'San Andrés', 'Colombia', 'South America'), # Añadido
    'EJE CAFETOR': ('REGION COL', 'Eje Cafetero', 'Colombia', 'South America'), # Añadido
    'EJE CAFATERO': ('REGION COL', 'Eje Cafetero', 'Colombia', 'South America'), # Añadido
    'EJE CAFETEORR': ('REGION COL', 'Eje Cafetero', 'Colombia', 'South America'), # Añadido
    'SAN ANDRRES': ('CIUDAD', 'San Andrés', 'Colombia', 'South America'), # Añadido
    'ANTA MARTA': ('CIUDAD', 'Santa Marta', 'Colombia', 'South America'), # Añadido
    'SA NANDRES': ('CIUDAD', 'San Andrés', 'Colombia', 'South America'), # Añadido
    'SAN ANRES': ('CIUDAD', 'San Andrés', 'Colombia', 'South America'), # Añadido
    'EJE CAFETEO': ('REGION COL', 'Eje Cafetero', 'Colombia', 'South America'), # Añadido
    'GUJIRA': ('REGION COL', 'La Guajira', 'Colombia', 'South America'), # Añadido
    'VILLA DELEIVA': ('CIUDAD', 'Villa de Leyva', 'Colombia', 'South America'), # Añadido
    'SAN ANNDRES': ('CIUDAD', 'San Andrés', 'Colombia', 'South America'), # Añadido
    'PAIAPA': ('CIUDAD', 'Paipa', 'Colombia', 'South America'), # Añadido
    'EJE CEFETRO': ('REGION COL', 'Eje Cafetero', 'Colombia', 'South America'), # Añadido
    'RIOCHA': ('CIUDAD', 'Riohacha', 'Colombia', 'South America'), # Añadido
    'BYACA': ('REGION COL', 'BOYACA', 'Colombia', 'South America'), # Añadido
    'EJE CAFETRI': ('REGION COL', 'Eje Cafetero', 'Colombia', 'South America'), # Añadido
    'PARQUE CAFFE': ('SITIO TURISTICO', 'PARQUE DEL CAFE', 'Colombia', 'South America'), # Añadido
    'PEIREIA': ('CIUDAD', 'Pereira', 'Colombia', 'South America'), # Añadido
    'PLAN SAN ANDRES': ('CIUDAD', 'San Andrés', 'Colombia', 'South America'), # Añadido
    'SAN TAMARTA': ('CIUDAD', 'Santa Marta', 'Colombia', 'South America'), # Añadido
    'SAN ANDRESS': ('CIUDAD', 'San Andrés', 'Colombia', 'South America'), # Añadido
    'EJE CAFETEERO': ('REGION COL', 'Eje Cafetero', 'Colombia', 'South America'), # Añadido
    'GIAJIRA': ('REGION COL', 'La Guajira', 'Colombia', 'South America'), # Añadido
    'SANT AMART': ('CIUDAD', 'Santa Marta', 'Colombia', 'South America'), # Añadido
    'SAN CIPRIANO': ('LUGAR', 'SAN CIPRIANO', 'Colombia', 'South America'), # Añadido
    'VILLA DE LEYBA': ('CIUDAD', 'Villa de Leyva', 'Colombia', 'South America'), # Añadido
    'SAN ANDRES': ('CIUDAD', 'San Andrés', 'Colombia', 'South America'),
    'SANANDRES': ('CIUDAD', 'San Andrés', 'Colombia', 'South America'),
    'SAN ANDES': ('CIUDAD', 'San Andrés', 'Colombia', 'South America'),
    'SANA ANDRES': ('CIUDAD', 'San Andrés', 'Colombia', 'South America'),
    'SANA NDRES': ('CIUDAD', 'San Andrés', 'Colombia', 'South America'),
    'SAN NADRES': ('CIUDAD', 'San Andrés', 'Colombia', 'South America'),
    'SAN ANRDES': ('CIUDAD', 'San Andrés', 'Colombia', 'South America'),
    'SA NA NDRES': ('CIUDAD', 'San Andrés', 'Colombia', 'South America'),
    'SAN ANDERS': ('CIUDAD', 'San Andrés', 'Colombia', 'South America'),
    'SA NADRES': ('CIUDAD', 'San Andrés', 'Colombia', 'South America'),
    'SN ANDRES': ('CIUDAD', 'San Andrés', 'Colombia', 'South America'),
    'SAN ADRES': ('CIUDAD', 'San Andrés', 'Colombia', 'South America'),
    'SAN ANDRESQ': ('CIUDAD', 'San Andrés', 'Colombia', 'South America'),
    'SAN ANDRS': ('CIUDAD', 'San Andrés', 'Colombia', 'South America'),
    'SAN ANDRES0': ('CIUDAD', 'San Andrés', 'Colombia', 'South America'),
    'SAN ADNRES': ('CIUDAD', 'San Andrés', 'Colombia', 'South America'),
    'SAN ANDRES D': ('CIUDAD', 'San Andrés', 'Colombia', 'South America'),
    'SAN ANDNRES': ('CIUDAD', 'San Andrés', 'Colombia', 'South America'),
    'SAN ANDRESS': ('CIUDAD', 'San Andrés', 'Colombia', 'South America'), # NUEVO
    'PLAN SAN ANDRES': ('CIUDAD', 'San Andrés', 'Colombia', 'South America'), # NUEVO
    'CARTAGENA': ('CIUDAD', 'Cartagena', 'Colombia', 'South America'),
    'CARTAGEN': ('CIUDAD', 'Cartagena', 'Colombia', 'South America'),
    'CARTAEGENA': ('CIUDAD', 'Cartagena', 'Colombia', 'South America'),
    'CARTAEGNA': ('CIUDAD', 'Cartagena', 'Colombia', 'South America'),
    'CARTAGENE': ('CIUDAD', 'Cartagena', 'Colombia', 'South America'),
    'CARTAENA': ('CIUDAD', 'Cartagena', 'Colombia', 'South America'),
    'CARTAGENNA': ('CIUDAD', 'Cartagena', 'Colombia', 'South America'),
    'CARTAGEMNA': ('CIUDAD', 'Cartagena', 'Colombia', 'South America'),
    'CARTGANEA': ('CIUDAD', 'Cartagena', 'Colombia', 'South America'),
    'CARATEGNA': ('CIUDAD', 'Cartagena', 'Colombia', 'South America'),
    'CATRAGEBA': ('CIUDAD', 'Cartagena', 'Colombia', 'South America'),
    'CATRGANEA': ('CIUDAD', 'Cartagena', 'Colombia', 'South America'),
    'CARAGENA': ('CIUDAD', 'Cartagena', 'Colombia', 'South America'),
    'CATAGENA': ('CIUDAD', 'Cartagena', 'Colombia', 'South America'),
    'CARTGENA': ('CIUDAD', 'Cartagena', 'Colombia', 'South America'),
    'LCARTAGENA': ('CIUDAD', 'Cartagena', 'Colombia', 'South America'),
    '28CARTAGENA': ('CIUDAD', 'Cartagena', 'Colombia', 'South America'),
    'CARTABGENA': ('CIUDAD', 'Cartagena', 'Colombia', 'South America'), # NUEVO
    'SANTA MARTA': ('CIUDAD', 'Santa Marta', 'Colombia', 'South America'),
    'SAMTA MARTA': ('CIUDAD', 'Santa Marta', 'Colombia', 'South America'),
    'SANTAMARTA': ('CIUDAD', 'Santa Marta', 'Colombia', 'South America'),
    'SANTA MATRA': ('CIUDAD', 'Santa Marta', 'Colombia', 'South America'),
    'SANA MARTA': ('CIUDAD', 'Santa Marta', 'Colombia', 'South America'),
    'SANTA NARTA': ('CIUDAD', 'Santa Marta', 'Colombia', 'South America'),
    'SNTA MARTA': ('CIUDAD', 'Santa Marta', 'Colombia', 'South America'),
    'NTA MARTA': ('CIUDAD', 'Santa Marta', 'Colombia', 'South America'),
    'SA2NTA MARTA': ('CIUDAD', 'Santa Marta', 'Colombia', 'South America'),
    'SANTAM ARTA': ('CIUDAD', 'Santa Marta', 'Colombia', 'South America'),
    'SSANTA MARTA': ('CIUDAD', 'Santa Marta', 'Colombia', 'South America'),
    'SAN MARTA': ('CIUDAD', 'Santa Marta', 'Colombia', 'South America'),
    'SANT AMART': ('CIUDAD', 'Santa Marta', 'Colombia', 'South America'), # NUEVO
    'SANTA MARTA D': ('CIUDAD', 'Santa Marta', 'Colombia', 'South America'), # NUEVO
    'SAN TAMARTA': ('CIUDAD', 'Santa Marta', 'Colombia', 'South America'), # NUEVO
    'SATA MARTA': ('CIUDAD', 'Santa Marta', 'Colombia', 'South America'), # NUEVO
    'QUINDIO': ('REGION COL', 'Quindío', 'Colombia', 'South America'),
    'QUIDIO': ('REGION COL', 'Quindío', 'Colombia', 'South America'),
    'QUINQUIO': ('REGION COL', 'Quindío', 'Colombia', 'South America'),
    'QUNDIO': ('REGION COL', 'Quindío', 'Colombia', 'South America'),
    'EJE CAFETERO': ('REGION COL', 'Eje Cafetero', 'Colombia', 'South America'),
    'EJE CFETERO': ('REGION COL', 'Eje Cafetero', 'Colombia', 'South America'),
    'EJE CAFETRO': ('REGION COL', 'Eje Cafetero', 'Colombia', 'South America'),
    'EJE CAFETEROP': ('REGION COL', 'Eje Cafetero', 'Colombia', 'South America'),
    'EJE CAETERO': ('REGION COL', 'Eje Cafetero', 'Colombia', 'South America'),
    'EJE CFATERO': ('REGION COL', 'Eje Cafetero', 'Colombia', 'South America'),
    'JE CAFETERO': ('REGION COL', 'Eje Cafetero', 'Colombia', 'South America'),
    'EJE CAFTEERO': ('REGION COL', 'Eje Cafetero', 'Colombia', 'South America'),
    'EJE CAFETETO': ('REGION COL', 'Eje Cafetero', 'Colombia', 'South America'),
    'EJE CEFETERO': ('REGION COL', 'Eje Cafetero', 'Colombia', 'South America'),
    'EJE CEFETRO': ('REGION COL', 'Eje Cafetero', 'Colombia', 'South America'), # NUEVO
    'EJE CAGETERO': ('REGION COL', 'Eje Cafetero', 'Colombia', 'South America'),
    'EJE CAFETERI': ('REGION COL', 'Eje Cafetero', 'Colombia', 'South America'),
    'EJE CAFETERA': ('REGION COL', 'Eje Cafetero', 'Colombia', 'South America'),
    'EJE CAFEERO': ('REGION COL', 'Eje Cafetero', 'Colombia', 'South America'),
    'CAFETERO': ('REGION COL', 'Eje Cafetero', 'Colombia', 'South America'),
    'EJE CAFETEERO': ('REGION COL', 'Eje Cafetero', 'Colombia', 'South America'), # NUEVO
    'EJE CAFETRI': ('REGION COL', 'Eje Cafetero', 'Colombia', 'South America'), # NUEVO
    'MACARENA': ('REGION COL', 'La Macarena', 'Colombia', 'South America'),
    'LA MACARENA': ('REGION COL', 'La Macarena', 'Colombia', 'South America'),
    'LETICIA': ('CIUDAD', 'Leticia', 'Colombia', 'South America'),
    'LEITICA': ('CIUDAD', 'Leticia', 'Colombia', 'South America'),
    'PASTO': ('CIUDAD', 'Pasto', 'Colombia', 'South America'),
    'ARMENIA': ('CIUDAD', 'Armenia', 'Colombia', 'South America'),
    'MEDELLIN': ('CIUDAD', 'Medellín', 'Colombia', 'South America'),
    'MEDELLN': ('CIUDAD', 'Medellín', 'Colombia', 'South America'),
    'MEDEELIN': ('CIUDAD', 'Medellín', 'Colombia', 'South America'),
    'MDELLIN': ('CIUDAD', 'Medellín', 'Colombia', 'South America'),
    'MEDELIIN': ('CIUDAD', 'Medellín', 'Colombia', 'South America'),
    'MEDELLION': ('CIUDAD', 'Medellín', 'Colombia', 'South America'),
    'MEDELLINB': ('CIUDAD', 'Medellín', 'Colombia', 'South America'),
    'BOGOTA': ('CIUDAD', 'Bogotá', 'Colombia', 'South America'),
    'BOGOTG': ('CIUDAD', 'Bogotá', 'Colombia', 'South America'),
    'BGOTA': ('CIUDAD', 'Bogotá', 'Colombia', 'South America'),
    'BOGTA': ('CIUDAD', 'Bogotá', 'Colombia', 'South America'),
    'CALI': ('CIUDAD', 'Cali', 'Colombia', 'South America'),
    'BARRANQUILLA': ('CIUDAD', 'Barranquilla', 'Colombia', 'South America'),
    'BARRANQUILAL': ('CIUDAD', 'Barranquilla', 'Colombia', 'South America'),
    'BARANQUILLA': ('CIUDAD', 'Barranquilla', 'Colombia', 'South America'),
    'BARRANQULLA': ('CIUDAD', 'Barranquilla', 'Colombia', 'South America'),
    'BARRANUQILLA': ('CIUDAD', 'Barranquilla', 'Colombia', 'South America'),
    'BARRRANQUILLA': ('CIUDAD', 'Barranquilla', 'Colombia', 'South America'),
    'BUCARAMANGA': ('CIUDAD', 'Bucaramanga', 'Colombia', 'South America'),
    'BUCARMANGA': ('CIUDAD', 'Bucaramanga', 'Colombia', 'South America'),
    'BUCARAMANAGA': ('CIUDAD', 'Bucaramanga', 'Colombia', 'South America'),
    'PEREIRA': ('CIUDAD', 'Pereira', 'Colombia', 'South America'),
    'PERIRA': ('CIUDAD', 'Pereira', 'Colombia', 'South America'),
    'PEIREIA': ('CIUDAD', 'Pereira', 'Colombia', 'South America'), # NUEVO
    'MANIZALES': ('CIUDAD', 'Manizales', 'Colombia', 'South America'),
    'MANIZALEZ': ('CIUDAD', 'Manizales', 'Colombia', 'South America'),
    'COVENAS': ('CIUDAD', 'Coveñas', 'Colombia', 'South America'),
    'COVENA': ('CIUDAD', 'Coveñas', 'Colombia', 'South America'),
    'PAIPA': ('CIUDAD', 'Paipa', 'Colombia', 'South America'),
    'PIAPA': ('CIUDAD', 'Paipa', 'Colombia', 'South America'),
    'PAIAPA': ('CIUDAD', 'Paipa', 'Colombia', 'South America'), # NUEVO
    'VILLA DE LEYVA': ('CIUDAD', 'Villa de Leyva', 'Colombia', 'South America'),
    'VILLADELEIVA': ('CIUDAD', 'Villa de Leyva', 'Colombia', 'South America'),
    'VILLA DE LEYBA': ('CIUDAD', 'Villa de Leyva', 'Colombia', 'South America'), # NUEVO
    'VILLA DE LEIVA': ('CIUDAD', 'Villa de Leyva', 'Colombia', 'South America'), # NUEVO
    'GIRARDOT': ('CIUDAD', 'Girardot', 'Colombia', 'South America'),
    'GIRARDO': ('CIUDAD', 'Girardot', 'Colombia', 'South America'),
    'GUATAPE': ('CIUDAD', 'Guatapé', 'Colombia', 'South America'),
    'GUARTAPE': ('CIUDAD', 'Guatapé', 'Colombia', 'South America'),
    'BAHIA SOLANO': ('CIUDAD', 'Bahía Solano', 'Colombia', 'South America'),
    'NECOCLI': ('CIUDAD', 'Necoclí', 'Colombia', 'South America'),
    'PALOMINO': ('LUGAR', 'Palomino', 'Colombia', 'South America'),
    'BARICHARA': ('CIUDAD', 'Barichara', 'Colombia', 'South America'),
    'NUQUI': ('CIUDAD', 'Nuquí', 'Colombia', 'South America'),
    'TOLU': ('CIUDAD', 'Tolú', 'Colombia', 'South America'),
    'CAPURGANA': ('LUGAR', 'Capurganá', 'Colombia', 'South America'),
    'MOMPOX': ('CIUDAD', 'Mompox', 'Colombia', 'South America'),
    'RIOHACHA': ('CIUDAD', 'Riohacha', 'Colombia', 'South America'),
    'RIOACHA': ('CIUDAD', 'Riohacha', 'Colombia', 'South America'),
    'RIOCHA': ('CIUDAD', 'Riohacha', 'Colombia', 'South America'), # NUEVO
    'DUITAMA': ('CIUDAD', 'Duitama', 'Colombia', 'South America'),
    'TOCANCIPA': ('CIUDAD', 'Tocancipá', 'Colombia', 'South America'),
    'PUERTO IRINIDA': ('CIUDAD', 'Inírida', 'Colombia', 'South America'),
    'GUAJIRA': ('REGION COL', 'La Guajira', 'Colombia', 'South America'),
    'LA GUAJRA': ('REGION COL', 'La Guajira', 'Colombia', 'South America'),
    'GIAJIRA': ('REGION COL', 'La Guajira', 'Colombia', 'South America'), # NUEVO
    'MOMPICHE': ('CIUDAD', 'Mompiche', 'Ecuador', 'South America'),
    'MEMPICHE': ('CIUDAD', 'Mompiche', 'Ecuador', 'South America'),
    'MONPICHE': ('CIUDAD', 'Mompiche', 'Ecuador', 'South America'),
    'MAMPICHE': ('CIUDAD', 'Mompiche', 'Ecuador', 'South America'),
    'SALENTO': ('CIUDAD', 'Salento', 'Colombia', 'South America'),
    'BOYOCA': ('REGION COL', 'BOYACA', 'Colombia', 'South America'),
    'BYACA': ('REGION COL', 'BOYACA', 'Colombia', 'South America'), # NUEVO
    'SANTADER': ('REGION COL', 'SANTANDERES', 'Colombia', 'South America'),
    'LLANOS ORIENTALES': ('REGION COL', 'LLANOS ORIENTALES', 'Colombia', 'South America'),
    'LLANOS': ('REGION COL', 'LLANOS ORIENTALES', 'Colombia', 'South America'),
    'LLANO': ('REGION COL', 'LLANOS ORIENTALES', 'Colombia', 'South America'),
    'RICAURTE': ('CIUDAD', 'Ricaurte', 'Colombia', 'South America'),
    'GUADAS': ('CIUDAD', 'Guaduas', 'Colombia', 'South America'),
    'SAN GERONIMO': ('CIUDAD', 'San Jerónimo', 'Colombia', 'South America'),
    'SANTA ROSA DEL CABAL': ('CIUDAD', 'Santa Rosa de Cabal', 'Colombia', 'South America'),
    'CHOACHI': ('CIUDAD', 'Choachí', 'Colombia', 'South America'),
    'SANGIL': ('CIUDAD', 'San Gil', 'Colombia', 'South America'),
    'UBAQUE': ('CIUDAD', 'Ubaque', 'Colombia', 'South America'),
    'BARRANQUEBERMEJA': ('CIUDAD', 'Barrancabermeja', 'Colombia', 'South America'),
    'APARATADO': ('CIUDAD', 'Apartadó', 'Colombia', 'South America'),
    'TIBASOSA': ('CIUDAD', 'Tibasosa', 'Colombia', 'South America'),
    'AGUADULCE': ('CIUDAD', 'Aguadulce', 'Panama', 'North America'), # O Colombia? Asumir Panama por contexto
    'TOLU Y COVENAS': ('REGION COL', 'TOLU Y COVENAS', 'Colombia', 'South America'),
    'ARBOLETES': ('CIUDAD', 'Arboletes', 'Colombia', 'South America'),
    'COCORNA': ('CIUDAD', 'Cocorná', 'Colombia', 'South America'),
    'SAN CIPRIANO': ('LUGAR', 'SAN CIPRIANO', 'Colombia', 'South America'), # NUEVO
    'TAMACA': ('HOTEL/LUGAR', 'HOTEL TAMACA', 'Colombia', 'South America'), # NUEVO
    'TATACOA': ('LUGAR', 'DESIERTO DE LA TATACOA', 'Colombia', 'South America'), # NUEVO
    'TATAMA': ('SITIO TURISTICO', 'PARQUE NACIONAL TATAMA', 'Colombia', 'South America'), # NUEVO
    'SAN ANTERO': ('CIUDAD', 'San Antero', 'Colombia', 'South America'), # NUEVO
    'IGUAQUE': ('SITIO TURISTICO', 'SANTUARIO IGUAQUE', 'Colombia', 'South America'), # NUEVO
    'MELGAR': ('CIUDAD', 'Melgar', 'Colombia', 'South America'), # NUEVO
    'DORADO': ('AEROPUERTO', 'EL DORADO BOGOTA', 'Colombia', 'South America'), # NUEVO
    'CANO DULCE': ('LUGAR', 'VOLCAN CANO DULCE', 'Colombia', 'South America'), # NUEVO
    'EJE CAFETERO PARQUE': ('SITIO TURISTICO', 'PARQUE DEL CAFE', 'Colombia', 'South America'), # NUEVO
    'GORGONA': ('ISLA', 'ISLA GORGONA', 'Colombia', 'South America'), # NUEVO
    'DORADAL': ('LUGAR', 'Doradal', 'Colombia', 'South America'), # NUEVO - Corregimiento
    'NEVADO RUIZ': ('SITIO TURISTICO', 'NEVADO DEL RUIZ', 'Colombia', 'South America'), # NUEVO
    'CIENEGA': ('CIUDAD', 'Ciénaga', 'Colombia', 'South America'), # NUEVO

    # Otros no clasificados geográficamente o ambiguos
    'OTRO': ('OTRO', 'OTRO', None, None), # Si ya viene como 'OTRO'
    'VAYJU': ('OTRO', 'VAYJU', None, None),
    'POYANA': ('OTRO', 'POYANA', None, None),
    'SALNEOD': ('OTRO', 'SALNEOD', None, None),
    'AHMED': ('OTRO', 'AHMED', None, None),
    'SANTUARIOS M': ('OTRO', 'SANTUARIOS M', None, None),

    # --- Entradas a DEJAR NULAS ('borrar') ---
    'INTERN': (None, None, None, None),
    'MONARCH': (None, None, None, None),
    'ENERO': (None, None, None, None),
    'DIRADAL': (None, None, None, None),
    'PAILITA': (None, None, None, None), # Aunque es municipio, se indica borrar
    'TURQUIA Y DUABI': (None, None, None, None),
    'PASADIA ZAPATOCA': (None, None, None, None),
    'LOMAS': (None, None, None, None),
    'TA EN CRUCERO POR EL': (None, None, None, None),
    'CENTRO AMERICA': (None, None, None, None), # Aunque es región, se indica borrar
    'RSVA FRES03': (None, None, None, None),

    # ... (añadir aquí cualquier otro código numérico o que se decida ignorar)
    '11419490': (None, None, None, None), '2045': (None, None, None, None),
    '00016898': (None, None, None, None), '1110470217': (None, None, None, None),
    '15471118': (None, None, None, None), '00017505': (None, None, None, None),
    '10010350541': (None, None, None, None), '00017690': (None, None, None, None),
    '57052446': (None, None, None, None), '17517681': (None, None, None, None),
    '15452064': (None, None, None, None), '15864170': (None, None, None, None),
    '57055125': (None, None, None, None), '16710712': (None, None, None, None),
    '18269561': (None, None, None, None), '14407459': (None, None, None, None),
    'RSV13120800': (None, None, None, None), 'CLO4916': (None, None, None, None),
    'FVT7250': (None, None, None, None),

}

# -----------------------------------------------------------------------------
# 2. FUNCIONES DE PROCESAMIENTO (Sin cambios necesarios aquí)
# -----------------------------------------------------------------------------

# ... (función limpiar_texto_geo sin cambios) ...
def limpiar_texto_geo(texto):
    """Limpia texto para búsqueda geográfica, devuelve None si es nulo/vacío."""
    return normalize_geo_name(texto)

# ... (función clasificar_destino sin cambios, ya usa la lógica correcta) ...
def clasificar_destino(destino_limpio):
    """
    Clasifica un destino (ya limpio y no nulo) en Tipo, Ciudad, País y Continente.
    Devuelve (None, None, None, None) para nulos o inválidos.
    """
    # 1. Manejar Nulos o Vacíos (ya debería venir como None de la limpieza)
    if destino_limpio is None:
        return (None, None, None, None)

    # 2. Manejar Códigos/Números -> Devuelve Nulo para todo
    if destino_limpio.isdigit():
        return (None, None, None, None)
    if (re.match(r'^[A-Z]{2,4}\d+$', destino_limpio) or \
        re.match(r'^\d+[A-Z]{2,4}$', destino_limpio) or \
        re.match(r'^[A-Z0-9]{8,}$', destino_limpio) and ' ' not in destino_limpio):
        return (None, None, None, None)


    # 3. Buscar en Mapeo Manual (PRIORIDAD ALTA)
    if destino_limpio in MAPEO_DESTINOS_MANUAL:
        map_result = MAPEO_DESTINOS_MANUAL[destino_limpio]
        # Si el mapeo devuelve (None, None, None, None), retornar eso directamente (para borrar)
        if all(v is None for v in map_result):
            return (None, None, None, None)

        tipo, ciudad, pais, cont = map_result
        # Si el tipo es OTRO/CONCEPTO/etc. y ciudad es None, usar el nombre mapeado como ciudad
        if tipo not in ['CIUDAD', 'PAIS', 'CONTINENTE', 'ISLA', 'REGION GEO', 'REGION COL'] and ciudad is None:
             ciudad = destino_limpio # Usar el nombre limpio original
        return (tipo, ciudad, pais, cont)

    # 4. Buscar en Ciudades (geonames) - Búsqueda exacta
    if destino_limpio in CITIES_LOOKUP:
        try:
            city_data = CITIES_LOOKUP[destino_limpio]
            country_code = city_data.get('countrycode')
            if country_code and country_code in COUNTRIES:
                country_data = COUNTRIES[country_code]
                continent_code = country_data.get('continentcode')
                if continent_code and continent_code in CONTINENTS:
                    continent = CONTINENTS[continent_code]['name']
                    ciudad_oficial = city_data.get('name')
                    pais_oficial = country_data.get('name')
                    if ciudad_oficial and pais_oficial:
                        return ('CIUDAD', ciudad_oficial, pais_oficial, continent)
        except Exception as e:
            print(f"Warning: Error procesando ciudad '{destino_limpio}' en geonames: {e}")

    # 5. Buscar en Países (geonames) - Búsqueda exacta
    if destino_limpio in COUNTRIES_LOOKUP:
        try:
            country_data = COUNTRIES_LOOKUP[destino_limpio]
            continent_code = country_data.get('continentcode')
            if continent_code and continent_code in CONTINENTS:
                continent = CONTINENTS[continent_code]['name']
                pais_oficial = country_data.get('name')
                if pais_oficial:
                    return ('PAIS', None, pais_oficial, continent)
        except Exception as e:
            print(f"Warning: Error procesando país '{destino_limpio}' en geonames: {e}")

    # 6. Buscar en Países (pycountry) - Búsqueda fuzzy
    try:
        paises_encontrados = pycountry.countries.search_fuzzy(destino_limpio)
        if paises_encontrados:
            pais_pyc = paises_encontrados[0]
            if hasattr(pais_pyc, 'alpha_2'):
                country_data = COUNTRIES.get(pais_pyc.alpha_2)
                if country_data:
                    continent_code = country_data.get('continentcode')
                    if continent_code and continent_code in CONTINENTS:
                         continent = CONTINENTS[continent_code]['name']
                         pais_oficial = country_data.get('name', pais_pyc.name)
                         if pais_oficial:
                             return ('PAIS', None, pais_oficial, continent)
    except Exception: pass

    # 7. Si después de todo no se encontró, marcar como 'NO CLASIFICADO'
    print(f"Info: Destino '{destino_limpio}' no encontrado, marcado como NO CLASIFICADO.")
    return ('NO CLASIFICADO', destino_limpio, None, None) # Usar nombre limpio original

# ... (función enriquecer_datos sin cambios) ...
def enriquecer_datos(carpeta_entrada, carpeta_salida):
    """
    Función principal para leer los 3 CSV BÁSICOS, enriquecerlos,
    y guardarlos en la carpeta final.
    """
    print(f"Iniciando Script 2: Leyendo archivos básicos de: '{carpeta_entrada}'")
    start_script_time = time.time()

    # --- 1. Cargar archivos BÁSICOS ---
    try:
        df_clientes = pd.read_csv(os.path.join(carpeta_entrada, 'clientes.csv'), low_memory=False)
        df_facturas = pd.read_csv(os.path.join(carpeta_entrada, 'facturas.csv'), low_memory=False)
        df_proveedores = pd.read_csv(os.path.join(carpeta_entrada, 'proveedores_por_factura.csv'), low_memory=False)
        print(f"Archivos básicos cargados: {len(df_clientes)} clientes, {len(df_facturas)} facturas.")
    except FileNotFoundError:
        print(f"Error CRÍTICO: No se encontraron los 3 archivos CSV básicos en '{carpeta_entrada}'.")
        print("Asegúrate de haber ejecutado primero el script 'crear_tablas_basicas.py'.")
        return
    except Exception as e:
        print(f"Error CRÍTICO al leer los archivos CSV de entrada: {e}")
        return

    # --- 2. Enriquecer CLIENTES (Regiones de Colombia) ---
    print("Enriqueciendo CLIENTES con Regiones de Colombia...")
    # Aplicar limpieza primero, devuelve None para nulos/vacíos
    df_clientes['zona_busqueda'] = df_clientes['zonas_ciudades_cli'].apply(limpiar_texto_geo)
    # Mapear, los None se quedarán como NaN (Nulo en CSV)
    df_clientes['region_colombia'] = df_clientes['zona_busqueda'].map(MAPEO_REGIONES_COLOMBIA)
    # Si algún valor mapeado es 'DESCONOCIDA', convertirlo a None también (ya está manejado en el dict con None)
    # df_clientes['region_colombia'] = df_clientes['region_colombia'].replace('DESCONOCIDA', None) #Redundante
    df_clientes = df_clientes.drop(columns=['zona_busqueda'])
    print("Clientes enriquecidos.")

    # --- 3. Enriquecer FACTURAS (Geo-destinos) ---
    print("Enriqueciendo FACTURAS con clasificación GEO (esto puede tomar varios minutos)...")
    start_time_clasif = time.time()
    # Aplicar limpieza primero, devuelve None para nulos/vacíos
    df_facturas['destino_busqueda'] = df_facturas['ciudad_destino'].apply(limpiar_texto_geo)

    # Aplicar la clasificación GEO (manejará los None de 'destino_busqueda')
    print("Clasificando destinos...")
    try:
        from tqdm import tqdm
        tqdm.pandas(desc="Clasificando GEO")
        resultados_geo = df_facturas['destino_busqueda'].progress_apply(clasificar_destino)
    except ImportError:
        print("(Instala 'tqdm' con 'pip install tqdm' para ver una barra de progreso)")
        resultados_geo = df_facturas['destino_busqueda'].apply(clasificar_destino)

    # Convertir resultados (tuplas) a DataFrame
    df_geo = pd.DataFrame(resultados_geo.tolist(),
                          index=df_facturas.index,
                          columns=['destino_tipo', 'destino_ciudad', 'destino_pais', 'destino_continente'])

    # Unir las nuevas columnas GEO. Mantener la columna original 'ciudad_destino'.
    df_facturas.reset_index(drop=True, inplace=True)
    df_geo.reset_index(drop=True, inplace=True)
    df_facturas_enriquecido = pd.concat([df_facturas, df_geo], axis=1)

    # Eliminar columna temporal
    df_facturas_enriquecido = df_facturas_enriquecido.drop(columns=['destino_busqueda'])
    print(f"Facturas enriquecidas en {time.time() - start_time_clasif:.2f} segundos.")

    # --- 4. Guardar archivos FINALES ---
    os.makedirs(carpeta_salida, exist_ok=True)
    print(f"\nGuardando archivos finales enriquecidos en: '{carpeta_salida}'")
    try:
        # Renombrar archivos de salida para claridad
        ruta_clientes_out = os.path.join(carpeta_salida, 'clientes_enriquecido.csv')
        df_clientes.to_csv(ruta_clientes_out, index=False, encoding='utf-8-sig')

        ruta_facturas_out = os.path.join(carpeta_salida, 'facturas_enriquecido.csv')
        df_facturas_enriquecido.to_csv(ruta_facturas_out, index=False, encoding='utf-8-sig')

        # La tabla de proveedores no se modifica, pero la guardamos en la nueva carpeta con nombre consistente
        ruta_proveedores_out = os.path.join(carpeta_salida, 'proveedores_por_factura_enriquecido.csv')
        df_proveedores.to_csv(ruta_proveedores_out, index=False, encoding='utf-8-sig')

        print(f"¡Archivos finales guardados con éxito en '{carpeta_salida}'!")
        print(f" - {os.path.basename(ruta_clientes_out)}")
        print(f" - {os.path.basename(ruta_facturas_out)}")
        print(f" - {os.path.basename(ruta_proveedores_out)}")

    except Exception as e:
        print(f"Error CRÍTICO al guardar los archivos CSV finales: {e}")

    print(f"\nTiempo total Script 2: {time.time() - start_script_time:.2f} segundos.")


# -----------------------------------------------------------------------------
# 3. EJECUCIÓN PRINCIPAL (Sin cambios)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    CARPETA_DATOS_ENTRADA = 'datos_limpios' # Lee de la salida del Script 1
    CARPETA_DATOS_SALIDA = 'datos_enriquecidos' # Carpeta final con datos mejorados
    enriquecer_datos(CARPETA_DATOS_ENTRADA, CARPETA_DATOS_SALIDA)
    print("\n" + "="*30 + "\n¡SCRIPT 2 (Enriquecimiento) COMPLETADO!\n" +
          f"Tus 3 archivos CSV finales están en: '{CARPETA_DATOS_SALIDA}'\n" +
          "¡Este es el final del proceso!\n" + "="*30)
