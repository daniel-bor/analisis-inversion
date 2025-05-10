\
import yfinance as yf
import pandas as pd
import yaml
from datetime import datetime
import os

def cargar_configuracion(ruta_config='../config.yaml'):
    """
    Carga la configuración desde un archivo YAML.

    Args:
        ruta_config (str): Ruta al archivo de configuración.

    Returns:
        dict: Diccionario con la configuración.
    """
    # Ajustar la ruta si es relativa al script actual
    if not os.path.isabs(ruta_config):
        script_dir = os.path.dirname(__file__)
        ruta_config = os.path.join(script_dir, ruta_config)
        
    with open(ruta_config, 'r') as file:
        config = yaml.safe_load(file)
    return config

def validar_ticker(ticker):
    """
    Valida si un ticker está disponible en Yahoo Finance.

    Args:
        ticker (str): Símbolo del ticker.

    Returns:
        bool: True si el ticker es válido, False en caso contrario.
    """
    try:
        data = yf.Ticker(ticker)
        if data.history(period="1d").empty:
            print(f"Advertencia: No se encontraron datos para el ticker {ticker}.")
            return False
        return True
    except Exception as e:
        print(f"Error al validar el ticker {ticker}: {e}")
        return False

def descargar_datos_empresa(ticker, fecha_inicio, fecha_fin):
    """
    Descarga datos históricos para un ticker específico.

    Args:
        ticker (str): Símbolo del ticker.
        fecha_inicio (str): Fecha de inicio (YYYY-MM-DD).
        fecha_fin (str): Fecha de fin (YYYY-MM-DD).

    Returns:
        pd.DataFrame: DataFrame con los datos históricos, o None si falla.
    """
    try:
        if not validar_ticker(ticker):
            return None
        
        data = yf.download(ticker, start=fecha_inicio, end=fecha_fin, progress=False)
        if data.empty:
            print(f"No se descargaron datos para {ticker} en el período {fecha_inicio} - {fecha_fin}.")
            return None
        print(f"Datos descargados para {ticker} desde {fecha_inicio} hasta {fecha_fin}")
        return data
    except Exception as e:
        print(f"Error al descargar datos para {ticker}: {e}")
        return None

def obtener_tickers_de_config(config):
    """
    Obtiene la lista de todos los tickers de la configuración.

    Args:
        config (dict): Diccionario de configuración cargado.

    Returns:
        list: Lista de tuplas (ticker, nombre_empresa, sector).
    """
    tickers_info = []
    for sector, empresas in config.get('empresas', {}).items():
        for empresa in empresas:
            tickers_info.append((empresa['ticker'], empresa['nombre'], sector))
    return tickers_info

if __name__ == '__main__':
    # Ejemplo de uso (esto se ejecutará solo si se corre data_loader.py directamente)
    config = cargar_configuracion()

    if config:
        print("Configuración cargada exitosamente.")
        
        fecha_inicio = config['periodo_analisis']['fecha_inicio']
        fecha_fin = config['periodo_analisis']['fecha_fin']
        
        print(f"Período de análisis: {fecha_inicio} a {fecha_fin}")

        empresas_info = obtener_tickers_de_config(config)
        
        if not empresas_info:
            print("No se encontraron empresas en la configuración.")
        else:
            print(f"Empresas a procesar: {empresas_info}")

            for ticker_symbol, nombre_empresa, sector in empresas_info:
                print(f"\\nProcesando: {nombre_empresa} ({ticker_symbol}) - Sector: {sector}")
                datos = descargar_datos_empresa(ticker_symbol, fecha_inicio, fecha_fin)
                if datos is not None and not datos.empty:
                    print(f"Primeras 5 filas de datos para {nombre_empresa}:")
                    print(datos.head())
                    
                    # Aquí podrías añadir la lógica para guardar los datos si es necesario
                    # Por ejemplo, guardarlos en la carpeta data/raw/
                    ruta_guardado = config['rutas']['datos_crudos']
                    if not os.path.isabs(ruta_guardado):
                        # Asumimos que la ruta en config.yaml es relativa a la raíz del proyecto
                        # y data_loader.py está en src/
                        script_dir = os.path.dirname(__file__)
                        project_root = os.path.dirname(script_dir) 
                        ruta_guardado_abs = os.path.join(project_root, ruta_guardado)
                    else:
                        ruta_guardado_abs = ruta_guardado
                        
                    if not os.path.exists(ruta_guardado_abs):
                        os.makedirs(ruta_guardado_abs)
                        print(f"Directorio creado: {ruta_guardado_abs}")
                        
                    nombre_archivo = f"{ticker_symbol}_datos_historicos.csv"
                    ruta_completa_archivo = os.path.join(ruta_guardado_abs, nombre_archivo)
                    datos.to_csv(ruta_completa_archivo)
                    print(f"Datos de {nombre_empresa} guardados en: {ruta_completa_archivo}")
                else:
                    print(f"No se pudieron descargar o no hay datos para {nombre_empresa}.")
    else:
        print("No se pudo cargar la configuración.")

