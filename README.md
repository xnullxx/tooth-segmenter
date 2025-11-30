# Separador Automático de Dientes FDI

Script en Python para separar automáticamente imágenes de dientes en dos partes: **corona** y **raíz**, detectando la orientación y encontrando la línea cervical (cuello del diente).

## 📋 Descripción

Este script procesa imágenes PNG de dientes numerados según el sistema **FDI (Fédération Dentaire Internationale)** y las separa automáticamente en dos archivos:

- `XX_corona.png` - Parte superior del diente (corona)
- `XX_raiz.png` - Parte inferior del diente (raíz)

Donde `XX` es el número FDI del diente (11, 12, 13, etc.).

## ✨ Características

- ✅ **Detección automática de orientación**: Identifica si la corona está arriba o abajo
- ✅ **Detección inteligente del cuello**: Encuentra la línea cervical (separación entre corona y raíz)
- ✅ **Soporte para todos los tipos de dientes**: Incisivos, caninos, premolares, molares (2 y 3 raíces), temporales
- ✅ **Manejo de fondos diversos**: Soporta fondos negros y transparentes (canal alpha)
- ✅ **Imágenes de debug**: Genera imágenes con la línea de corte marcada para verificación
- ✅ **Procesamiento por lotes**: Procesa todas las imágenes de una carpeta automáticamente

## 🛠️ Requisitos

- Python 3.7 o superior
- OpenCV (cv2)
- NumPy

## 📦 Instalación

1. Clona o descarga este repositorio
2. Instala las dependencias:

```bash
pip install -r requirements.txt
```

O manualmente:

```bash
pip install opencv-python numpy
```

## 🚀 Uso

### Uso básico

Coloca tus imágenes PNG de dientes en una carpeta llamada `FDI/` y ejecuta:

```bash
python separar_diente.py
```

El script procesará todas las imágenes PNG en la carpeta `FDI/` y guardará los resultados en `FDI_SPLIT/`.

### Estructura de archivos

```
proyecto/
├── FDI/
│   ├── 11.png
│   ├── 12.png
│   └── ...
├── FDI_SPLIT/
│   ├── 11_corona.png
│   ├── 11_raiz.png
│   ├── 11_debug.png
│   └── ...
├── separar_diente.py
└── requirements.txt
```

### Personalización

Puedes modificar los parámetros en el script:

```python
procesar_carpeta(
    carpeta_entrada="FDI",           # Carpeta de entrada
    carpeta_salida="FDI_SPLIT",      # Carpeta de salida
    generar_debug=True                # Generar imágenes de debug
)
```

## 🔬 Cómo funciona

### 1. Detección de orientación

El script analiza el perfil de ancho del diente para determinar si la corona está arriba o abajo:

- **Corona arriba**: El tercio superior es más ancho que el inferior
- **Corona abajo**: El tercio inferior es más ancho que el superior

### 2. Detección del cuello (línea cervical)

El algoritmo busca el punto de transición entre la corona y la raíz:

- **Para corona arriba**: Busca donde la corona (ancha arriba) comienza a estrecharse (40-50%)
- **Para corona abajo**: Busca donde la raíz (estrecha arriba) se ensancha hacia la corona (58-68%)
- **Para molares**: Detecta expansión de raíces múltiples y ajusta el cuello apropiadamente

### 3. Separación

Una vez detectado el cuello, el script:

1. Crea dos imágenes del mismo tamaño que la original
2. Copia la parte superior (corona) en una imagen
3. Copia la parte inferior (raíz) en otra imagen
4. Mantiene el fondo original (negro o transparente)

## 📊 Algoritmo

El script utiliza las siguientes técnicas de procesamiento de imágenes:

- **Máscaras binarias**: Para separar el diente del fondo
- **Perfiles de ancho**: Para analizar la forma del diente verticalmente
- **Suavizado gaussiano**: Para reducir ruido en el análisis
- **Análisis de derivadas**: Para encontrar puntos de transición
- **Detección de mínimos locales**: Para identificar el cuello del diente

## 📝 Formato de imágenes

- **Formato**: PNG
- **Fondo**: Negro sólido o transparente (con canal alpha)
- **Orientación**: Puede variar (corona arriba o abajo)
- **Resolución**: Alta resolución recomendada para mejor precisión

## 🎯 Ejemplos de uso

### Procesar una carpeta específica

```python
from separar_diente import procesar_carpeta

procesar_carpeta(
    carpeta_entrada="mis_dientes",
    carpeta_salida="resultados",
    generar_debug=True
)
```

### Procesar una imagen individual

```python
from separar_diente import procesar_imagen
from pathlib import Path

procesar_imagen(
    ruta_imagen=Path("FDI/11.png"),
    carpeta_salida=Path("FDI_SPLIT"),
    generar_debug=True
)
```

## 📈 Resultados

Para cada imagen procesada, se generan:

1. **`XX_corona.png`**: Imagen con solo la parte de la corona
2. **`XX_raiz.png`**: Imagen con solo la parte de la raíz
3. **`XX_debug.png`**: Imagen de debug con:
   - Línea roja marcando el cuello detectado
   - Texto indicando la orientación detectada
   - Posición del cuello en píxeles y porcentaje

## ⚙️ Configuración avanzada

Puedes ajustar los parámetros en la sección de configuración del script:

```python
DEFAULT_NECK_RATIO = 0.57        # Ratio por defecto para el cuello
THRESHOLD_BACKGROUND = 30       # Umbral para detectar el fondo
CROWN_WIDTH_THRESHOLD = 0.85     # Umbral para identificar la corona
```

## 🐛 Solución de problemas

### El cuello no se detecta correctamente

- Verifica que la imagen tenga buen contraste entre el diente y el fondo
- Asegúrate de que el diente esté centrado en la imagen
- Revisa las imágenes de debug para ver dónde se está detectando el cuello

### Error al leer imágenes

- Verifica que las imágenes sean PNG válidas
- Asegúrate de que la carpeta `FDI/` exista y contenga imágenes

## 📄 Licencia

Este proyecto es de código abierto y está disponible para uso libre.

## 👤 Autor

**xnullxx**

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Si encuentras algún problema o tienes sugerencias, no dudes en abrir un issue o crear un pull request.

## 📚 Referencias

- Sistema de numeración FDI (Fédération Dentaire Internationale)
- OpenCV Documentation: https://docs.opencv.org/
- NumPy Documentation: https://numpy.org/doc/

