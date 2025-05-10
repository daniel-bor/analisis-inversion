"""
Módulo de preprocesamiento para datos financieros.

Este módulo proporciona funciones para:
- Unificar datos de múltiples fuentes
- Calcular retornos diarios
- Detectar y manejar valores atípicos
- Manejar datos faltantes (NaN)
- Preparar el conjunto de datos para análisis estadísticos posteriores
"""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import yaml
from datetime import datetime
import warnings
from scipy import stats


def cargar_datos_brutos(ruta_datos=None, config=None):
    """
    Carga los datos crudos desde la carpeta data/raw/ y los unifica en un solo DataFrame.
    
    Args:
        ruta_datos (str): Ruta a la carpeta con los datos crudos. Si es None, se obtiene de config.
        config (dict): Configuración del proyecto. Si es None, se carga del archivo config.yaml.
        
    Returns:
        pd.DataFrame: DataFrame unificado con todos los datos.
    """
    if config is None:
        # Cargar configuración desde archivo
        script_dir = os.path.dirname(__file__)
        config_path = os.path.join(os.path.dirname(script_dir), 'config.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    
    if ruta_datos is None:
        ruta_datos = config['rutas']['datos_crudos']
        if not os.path.isabs(ruta_datos):
            # Convertir ruta relativa a absoluta
            script_dir = os.path.dirname(__file__)
            project_root = os.path.dirname(script_dir)
            ruta_datos = os.path.join(project_root, ruta_datos)
    
    # Buscar todos los archivos CSV en la carpeta de datos crudos
    archivos_csv = glob.glob(os.path.join(ruta_datos, '*_datos_historicos.csv'))
    
    if not archivos_csv:
        raise FileNotFoundError(f"No se encontraron archivos CSV en {ruta_datos}")
    
    # Lista para almacenar los DataFrames por cada archivo
    dataframes = []
    
    # Leer cada archivo y agregarlo a la lista
    for archivo in archivos_csv:
        df = pd.read_csv(archivo)
        # Verificar si el archivo tiene las columnas esperadas
        columnas_requeridas = ['Date', 'Ticker', 'Close', 'High', 'Low', 'Open', 'Volume']
        if not all(col in df.columns for col in columnas_requeridas):
            warnings.warn(f"El archivo {archivo} no tiene todas las columnas requeridas. Se omitirá.")
            continue
        
        # Añadir información del sector según la configuración
        ticker = df['Ticker'].iloc[0] if not df.empty else os.path.basename(archivo).split('_')[0]
        
        # Asignar el sector según el ticker
        sector = None
        for sector_nombre, empresas in config['empresas'].items():
            if any(empresa['ticker'] == ticker for empresa in empresas):
                sector = sector_nombre
                break
        
        if sector:
            df['Sector'] = sector
        else:
            warnings.warn(f"No se pudo determinar el sector para el ticker {ticker}")
            df['Sector'] = 'Desconocido'
        
        # Añadir el DataFrame a la lista
        dataframes.append(df)
    
    # Combinar todos los DataFrames en uno solo
    if not dataframes:
        raise ValueError("No se pudieron cargar datos válidos")
    
    datos_combinados = pd.concat(dataframes, ignore_index=True)
    
    # Convertir la columna Date a datetime
    datos_combinados['Date'] = pd.to_datetime(datos_combinados['Date'])
    
    # Ordenar datos por fecha y ticker
    datos_combinados = datos_combinados.sort_values(['Date', 'Ticker'])
    
    return datos_combinados


def calcular_retornos(df, metodo='simple', columna_precio='Close'):
    """
    Calcula los retornos diarios para cada acción.
    
    Args:
        df (pd.DataFrame): DataFrame con datos históricos de precios.
        metodo (str): Método para calcular retornos: 'simple' o 'logaritmico'.
        columna_precio (str): Columna que contiene los precios para calcular retornos.
    
    Returns:
        pd.DataFrame: DataFrame original con columnas adicionales de retornos.
    """
    # Verificar que el DataFrame tiene las columnas necesarias
    if columna_precio not in df.columns:
        raise ValueError(f"El DataFrame no contiene la columna {columna_precio}")
    
    if 'Ticker' not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'Ticker'")
    
    if 'Date' not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'Date'")
    
    # Crear una copia para no modificar el original
    df_retornos = df.copy()
    
    # Calcular retornos por ticker
    for ticker in df_retornos['Ticker'].unique():
        mask = df_retornos['Ticker'] == ticker
        
        # Ordenar por fecha para ese ticker específico
        df_ticker = df_retornos[mask].sort_values('Date')
        
        if metodo == 'simple':
            # Retornos simples: (P_t / P_{t-1}) - 1
            df_retornos.loc[mask, 'Retorno'] = df_ticker[columna_precio].pct_change()
        elif metodo == 'logaritmico':
            # Retornos logarítmicos: log(P_t / P_{t-1})
            df_retornos.loc[mask, 'Retorno'] = np.log(df_ticker[columna_precio] / df_ticker[columna_precio].shift(1))
        else:
            raise ValueError("El método debe ser 'simple' o 'logaritmico'")
    
    # Calcular retorno acumulado
    for ticker in df_retornos['Ticker'].unique():
        mask = df_retornos['Ticker'] == ticker
        df_ticker = df_retornos[mask].sort_values('Date')
        
        if metodo == 'simple':
            # Retorno acumulado simple
            df_retornos.loc[mask, 'RetornoAcumulado'] = (1 + df_ticker['Retorno']).cumprod() - 1
        elif metodo == 'logaritmico':
            # Retorno acumulado logarítmico
            df_retornos.loc[mask, 'RetornoAcumulado'] = df_ticker['Retorno'].cumsum()
    
    return df_retornos


def detectar_outliers(df, columna, metodo='zscore', umbral=3.0):
    """
    Detecta valores atípicos (outliers) en una columna específica.
    
    Args:
        df (pd.DataFrame): DataFrame con datos.
        columna (str): Nombre de la columna a analizar.
        metodo (str): Método para detectar outliers: 'zscore', 'iqr', o 'percentil'.
        umbral (float): Umbral para considerar un valor como outlier.
            - Para 'zscore': número de desviaciones estándar (por defecto 3.0)
            - Para 'iqr': multiplicador del rango intercuartílico (por defecto 1.5)
            - Para 'percentil': no se usa
            
    Returns:
        pd.Series: Serie booleana con True donde hay outliers.
    """
    if columna not in df.columns:
        raise ValueError(f"La columna {columna} no existe en el DataFrame")
    
    # Obtener la serie numérica sin NaN
    serie = df[columna].dropna()
    
    # Inicializar la serie de outliers
    outliers = pd.Series(False, index=df.index)
    
    if metodo == 'zscore':
        # Método de Z-score
        z_scores = np.abs(stats.zscore(serie))
        outliers[serie.index] = z_scores > umbral
    
    elif metodo == 'iqr':
        # Método del rango intercuartílico
        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - umbral * iqr
        upper_bound = q3 + umbral * iqr
        outliers[serie.index] = (serie < lower_bound) | (serie > upper_bound)
    
    elif metodo == 'percentil':
        # Método de percentiles
        lower_bound = serie.quantile(0.01)  # Percentil 1
        upper_bound = serie.quantile(0.99)  # Percentil 99
        outliers[serie.index] = (serie < lower_bound) | (serie > upper_bound)
    
    else:
        raise ValueError("Método no válido. Use 'zscore', 'iqr', o 'percentil'")
    
    return outliers


def manejar_outliers(df, columna, metodo_deteccion='zscore', metodo_manejo='clip', umbral=3.0):
    """
    Detecta y maneja los valores atípicos en una columna.
    
    Args:
        df (pd.DataFrame): DataFrame con datos.
        columna (str): Nombre de la columna a analizar.
        metodo_deteccion (str): Método para detectar outliers: 'zscore', 'iqr', o 'percentil'.
        metodo_manejo (str): Método para manejar outliers: 'clip', 'eliminar', o 'reemplazar'.
        umbral (float): Umbral para considerar un valor como outlier.
        
    Returns:
        pd.DataFrame: DataFrame con outliers manejados.
    """
    # Detectar outliers
    outliers = detectar_outliers(df, columna, metodo_deteccion, umbral)
    
    # Crear una copia del DataFrame para no modificar el original
    df_limpio = df.copy()
    
    if metodo_manejo == 'clip':
        # Método de recorte (clip)
        serie = df[columna].dropna()
        if metodo_deteccion == 'zscore':
            # Calcular límites basados en desviaciones estándar
            media = serie.mean()
            std = serie.std()
            lower_bound = media - umbral * std
            upper_bound = media + umbral * std
        elif metodo_deteccion == 'iqr':
            # Calcular límites basados en IQR
            q1 = serie.quantile(0.25)
            q3 = serie.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - umbral * iqr
            upper_bound = q3 + umbral * iqr
        elif metodo_deteccion == 'percentil':
            # Calcular límites basados en percentiles
            lower_bound = serie.quantile(0.01)
            upper_bound = serie.quantile(0.99)
        
        # Aplicar el recorte
        df_limpio[columna] = df_limpio[columna].clip(lower=lower_bound, upper=upper_bound)
    
    elif metodo_manejo == 'eliminar':
        # Método de eliminación
        df_limpio = df_limpio[~outliers]
    
    elif metodo_manejo == 'reemplazar':
        # Método de reemplazo por NaN
        df_limpio.loc[outliers, columna] = np.nan
    
    else:
        raise ValueError("Método de manejo no válido. Use 'clip', 'eliminar', o 'reemplazar'")
    
    return df_limpio


def manejar_datos_faltantes(df, metodo='interpolar', columnas=None):
    """
    Maneja los valores faltantes (NaN) en el DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame con datos.
        metodo (str): Método para manejar datos faltantes: 'interpolar', 'eliminar', 'forward_fill', 
                      'backward_fill', o 'media_grupo'.
        columnas (list): Lista de columnas a procesar. Si es None, se procesan todas las columnas numéricas.
        
    Returns:
        pd.DataFrame: DataFrame con datos faltantes manejados.
    """
    # Crear una copia del DataFrame para no modificar el original
    df_completo = df.copy()
    
    # Si no se especifican columnas, usar las columnas numéricas
    if columnas is None:
        columnas = df_completo.select_dtypes(include=[np.number]).columns.tolist()
    
    # Verificar que las columnas existen en el DataFrame
    columnas_existentes = [col for col in columnas if col in df_completo.columns]
    if not columnas_existentes:
        warnings.warn("Ninguna de las columnas especificadas existe en el DataFrame")
        return df_completo
    
    # Manejar datos faltantes por método
    if metodo == 'eliminar':
        # Eliminar filas con valores NA en las columnas especificadas
        df_completo = df_completo.dropna(subset=columnas_existentes)
    
    elif metodo == 'interpolar':
        # Para cada ticker, interpolar los valores faltantes
        for ticker in df_completo['Ticker'].unique():
            mask = df_completo['Ticker'] == ticker
            for col in columnas_existentes:
                # Usar interpolación lineal para datos faltantes
                df_completo.loc[mask, col] = df_completo.loc[mask, col].interpolate(method='linear')
    
    elif metodo == 'forward_fill':
        # Para cada ticker, propagar el último valor válido hacia adelante
        for ticker in df_completo['Ticker'].unique():
            mask = df_completo['Ticker'] == ticker
            df_completo.loc[mask, columnas_existentes] = df_completo.loc[mask, columnas_existentes].fillna(method='ffill')
    
    elif metodo == 'backward_fill':
        # Para cada ticker, propagar el próximo valor válido hacia atrás
        for ticker in df_completo['Ticker'].unique():
            mask = df_completo['Ticker'] == ticker
            df_completo.loc[mask, columnas_existentes] = df_completo.loc[mask, columnas_existentes].fillna(method='bfill')
    
    elif metodo == 'media_grupo':
        # Reemplazar por la media del grupo (por ticker y sector)
        for ticker in df_completo['Ticker'].unique():
            mask = df_completo['Ticker'] == ticker
            for col in columnas_existentes:
                media = df_completo.loc[mask, col].mean()
                df_completo.loc[mask, col] = df_completo.loc[mask, col].fillna(media)
    
    else:
        raise ValueError("Método no válido. Use 'interpolar', 'eliminar', 'forward_fill', 'backward_fill', o 'media_grupo'")
    
    return df_completo


def calcular_indicadores_tecnicos(df):
    """
    Calcula indicadores técnicos comunes para análisis financiero.
    
    Args:
        df (pd.DataFrame): DataFrame con datos históricos de precios.
        
    Returns:
        pd.DataFrame: DataFrame con indicadores técnicos añadidos.
    """
    # Crear una copia del DataFrame para no modificar el original
    df_indicadores = df.copy()
    
    # Para cada ticker, calcular los indicadores
    for ticker in df_indicadores['Ticker'].unique():
        mask = df_indicadores['Ticker'] == ticker
        df_ticker = df_indicadores[mask].sort_values('Date')
        
        # Medias móviles (7, 21 y 50 días)
        df_indicadores.loc[mask, 'MA7'] = df_ticker['Close'].rolling(window=7).mean()
        df_indicadores.loc[mask, 'MA21'] = df_ticker['Close'].rolling(window=21).mean()
        df_indicadores.loc[mask, 'MA50'] = df_ticker['Close'].rolling(window=50).mean()
        
        # Volatilidad (desviación estándar de retornos en ventana de 21 días)
        if 'Retorno' in df_ticker.columns:
            df_indicadores.loc[mask, 'Volatilidad21'] = df_ticker['Retorno'].rolling(window=21).std()
        
        # Volumen promedio (7 días)
        df_indicadores.loc[mask, 'VolumenPromedio7'] = df_ticker['Volume'].rolling(window=7).mean()
        
        # RSI (Relative Strength Index) - 14 días
        delta = df_ticker['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        # Evitar división por cero
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
        
        df_indicadores.loc[mask, 'RSI'] = rsi
    
    return df_indicadores


def unificar_datos(df, metodo_retornos='simple', manejar_outliers_retornos=True,
                  manejar_na='interpolar', calcular_tecnicos=True):
    """
    Unifica el proceso de preprocesamiento de datos financieros en una sola función.
    
    Args:
        df (pd.DataFrame): DataFrame con datos históricos de precios.
        metodo_retornos (str): Método para calcular retornos ('simple' o 'logaritmico').
        manejar_outliers_retornos (bool): Si es True, se manejan los outliers en la columna de retornos.
        manejar_na (str): Método para manejar datos faltantes.
        calcular_tecnicos (bool): Si es True, se calculan indicadores técnicos.
        
    Returns:
        pd.DataFrame: DataFrame procesado y listo para análisis posteriores.
    """
    # 1. Calcular retornos
    df_procesado = calcular_retornos(df, metodo=metodo_retornos)
    
    # 2. Manejar outliers en retornos si está especificado
    if manejar_outliers_retornos and 'Retorno' in df_procesado.columns:
        df_procesado = manejar_outliers(df_procesado, 'Retorno', 
                                        metodo_deteccion='zscore',
                                        metodo_manejo='clip',
                                        umbral=3.0)
    
    # 3. Manejar datos faltantes
    df_procesado = manejar_datos_faltantes(df_procesado, metodo=manejar_na)
    
    # 4. Calcular indicadores técnicos si está especificado
    if calcular_tecnicos:
        df_procesado = calcular_indicadores_tecnicos(df_procesado)
    
    return df_procesado


def guardar_datos_procesados(df, ruta_salida=None, formato='csv', config=None):
    """
    Guarda los datos procesados en la ubicación especificada.
    
    Args:
        df (pd.DataFrame): DataFrame con datos procesados.
        ruta_salida (str): Ruta donde guardar los datos. Si es None, se obtiene de config.
        formato (str): Formato de salida ('csv' o 'parquet').
        config (dict): Configuración del proyecto. Si es None, se carga del archivo config.yaml.
        
    Returns:
        str: Ruta donde se guardaron los datos.
    """
    if config is None:
        # Cargar configuración desde archivo
        script_dir = os.path.dirname(__file__)
        config_path = os.path.join(os.path.dirname(script_dir), 'config.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    
    if ruta_salida is None:
        ruta_salida = config['rutas']['datos_procesados']
        if not os.path.isabs(ruta_salida):
            # Convertir ruta relativa a absoluta
            script_dir = os.path.dirname(__file__)
            project_root = os.path.dirname(script_dir)
            ruta_salida = os.path.join(project_root, ruta_salida)
    
    # Asegurar que el directorio existe
    os.makedirs(ruta_salida, exist_ok=True)
    
    # Generar nombre de archivo con la fecha actual
    fecha_actual = datetime.now().strftime('%Y%m%d')
    nombre_archivo = f"datos_financieros_procesados_{fecha_actual}"
    
    # Guardar en el formato especificado
    ruta_completa = None
    if formato == 'csv':
        ruta_completa = os.path.join(ruta_salida, f"{nombre_archivo}.csv")
        df.to_csv(ruta_completa, index=False)
    elif formato == 'parquet':
        ruta_completa = os.path.join(ruta_salida, f"{nombre_archivo}.parquet")
        df.to_parquet(ruta_completa, index=False)
    else:
        raise ValueError(f"Formato {formato} no soportado. Use 'csv' o 'parquet'")
    
    print(f"Datos guardados en: {ruta_completa}")
    
    # Generar y guardar metadatos
    metadatos = {
        'fecha_procesamiento': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'filas': len(df),
        'columnas': len(df.columns),
        'empresas': df['Ticker'].unique().tolist(),
        'sectores': df['Sector'].unique().tolist(),
        'fecha_inicio': df['Date'].min().strftime('%Y-%m-%d') if not df.empty else None,
        'fecha_fin': df['Date'].max().strftime('%Y-%m-%d') if not df.empty else None,
        'formato': formato,
        'ruta_archivo': ruta_completa
    }
    
    ruta_metadatos = os.path.join(ruta_salida, f"{nombre_archivo}_metadatos.json")
    import json
    with open(ruta_metadatos, 'w') as file:
        json.dump(metadatos, file, indent=4)
    
    return ruta_completa


# Función principal para procesar datos desde la línea de comandos
if __name__ == '__main__':
    # Cargar configuración
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(os.path.dirname(script_dir), 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    print("Cargando datos brutos...")
    datos_brutos = cargar_datos_brutos(config=config)
    print(f"Datos cargados: {datos_brutos.shape[0]} filas, {datos_brutos.shape[1]} columnas")
    
    print("Procesando datos...")
    datos_procesados = unificar_datos(datos_brutos)
    print(f"Datos procesados: {datos_procesados.shape[0]} filas, {datos_procesados.shape[1]} columnas")
    
    print("Guardando datos procesados...")
    ruta_salida = guardar_datos_procesados(datos_procesados, config=config)
    print(f"Proceso completado. Datos guardados en: {ruta_salida}")