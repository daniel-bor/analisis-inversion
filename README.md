# Análisis y Optimización de Inversiones

Proyecto para el análisis estadístico de datos financieros y optimización de carteras de inversión en los sectores tecnológico y farmacéutico.

## Descripción del Proyecto

Este proyecto realiza un análisis completo de datos financieros históricos para empresas de los sectores tecnológico y farmacéutico. El objetivo es crear un modelo predictivo para optimizar una cartera de inversión con un presupuesto definido, considerando el balance entre rendimiento y riesgo.

## Estructura del Proyecto

```
analisis-inversion/
│
├── config.yaml                 # Configuración general del proyecto
├── README.md                   # Este archivo
├── requirements.txt            # Dependencias del proyecto
│
├── data/                       # Datos del proyecto
│   ├── processed/              # Datos procesados y listos para análisis
│   └── raw/                    # Datos crudos descargados de Yahoo Finance
│
├── notebooks/                  # Notebooks de Jupyter para análisis
│   ├── 1_extraccion_datos.ipynb       # Extracción de datos históricos
│   ├── 2_limpieza_analisis.ipynb      # Limpieza y preprocesamiento
│   ├── 3_estadisticas_descriptivas.ipynb  # Análisis estadístico
│   └── 4_modelado_prediccion.ipynb    # Modelado y predicción
│
├── reports/                    # Informes y visualizaciones
│   ├── figures/                # Gráficas generadas
│   ├── estadisticas_descriptivas_empresas.csv
│   └── estadisticas_descriptivas_sectores.csv
│
└── src/                        # Módulos de código fuente
    ├── data_loader.py          # Carga y validación de datos
    ├── preprocess.py           # Funciones de preprocesamiento
    └── utils.py                # Utilidades generales
```

## Flujo de Trabajo

1. **Extracción de Datos**: Descarga de datos históricos de cotizaciones usando Yahoo Finance.
2. **Limpieza y Preprocesamiento**: Tratamiento de valores faltantes, cálculo de retornos y normalización.
3. **Análisis Estadístico**: Cálculo de estadísticas descriptivas por empresa y sector.
4. **Modelado y Predicción**: Desarrollo de modelos predictivos para optimizar inversiones.

## Empresas Analizadas

### Sector Tecnológico
- Microsoft (MSFT)
- Alphabet (GOOG)
- Meta Platforms (META)
- Amazon (AMZN)
- NVIDIA (NVDA)

### Sector Farmacéutico
- Johnson & Johnson (JNJ)
- Merck & Co. (MRK)
- Pfizer (PFE)
- Bristol Myers Squibb (BMY)
- Eli Lilly (LLY)

## Periodo de Análisis

El análisis cubre datos desde mayo de 2022 hasta mayo de 2025.

## Principales Características

- Extracción automatizada de datos financieros históricos
- Cálculo de métricas estadísticas y financieras
- Visualización de tendencias y patrones por empresa y sector
- Modelo predictivo para optimización de carteras de inversión
- Recomendaciones de inversión basadas en análisis estadístico

## Requisitos

Para ejecutar este proyecto se necesitan las siguientes dependencias (detalladas en `requirements.txt`):
- Python 3.10+
- pandas
- numpy
- matplotlib
- seaborn
- yfinance
- scikit-learn
- jupyter

## Ejecución

Los notebooks deben ejecutarse en orden secuencial:

1. `1_extraccion_datos.ipynb`
2. `2_limpieza_analisis.ipynb`
3. `3_estadisticas_descriptivas.ipynb`
4. `4_modelado_prediccion.ipynb`

## Resultados

Los resultados incluyen:
- Datos procesados en formato CSV
- Estadísticas descriptivas por empresa y sector
- Visualizaciones de evolución de precios y rentabilidades
- Modelo predictivo para optimizar inversiones
- Recomendación final de distribución de cartera