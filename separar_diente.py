"""
Script para separar imágenes de dientes en corona y raíz.

Este script analiza imágenes PNG de dientes numerados según FDI y las separa
automáticamente en dos partes: corona y raíz, detectando la orientación y
encontrando la línea cervical (cuello del diente).
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple

DEFAULT_NECK_RATIO = 0.57
NECK_SEARCH_START = 0.30
NECK_SEARCH_END = 0.70
THRESHOLD_BACKGROUND = 30
CROWN_WIDTH_THRESHOLD = 0.85


def crear_mascara_diente(imagen: np.ndarray) -> np.ndarray:
    """
    Crea una máscara binaria del diente, manejando fondos negros y transparentes.
    
    Args:
        imagen: Imagen BGR o BGRA
        
    Returns:
        Máscara binaria (255 = diente, 0 = fondo)
    """
    if imagen.shape[2] == 4:
        b, g, r, alpha = cv2.split(imagen)
        mascara_alpha = alpha > 0
        gris = cv2.cvtColor(imagen[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        mascara_alpha = np.ones_like(gris, dtype=bool)
    
    _, mascara_brillo = cv2.threshold(gris, THRESHOLD_BACKGROUND, 255, cv2.THRESH_BINARY)
    mascara_final = np.logical_and(mascara_alpha, mascara_brillo > 0)
    
    return mascara_final.astype(np.uint8) * 255


def calcular_perfil_ancho(mascara: np.ndarray) -> np.ndarray:
    """
    Calcula el perfil de ancho vertical del diente.
    
    Para cada fila (altura), cuenta cuántos píxeles del diente hay.
    Esto nos da el ancho del diente en cada nivel vertical.
    
    Args:
        mascara: Máscara binaria del diente
        
    Returns:
        Array 1D con el ancho del diente en cada fila
    """
    altura = mascara.shape[0]
    perfil = np.zeros(altura, dtype=np.float32)
    
    for fila in range(altura):
        perfil[fila] = np.sum(mascara[fila, :] > 0)
    
    return perfil


def detect_orientation(perfil_ancho: np.ndarray) -> str:
    """
    Detecta si la corona está arriba o abajo.
    
    Criterios:
    1. La corona es siempre más ancha que la raíz
    2. Analizamos el ancho promedio en el tercio superior vs inferior
    3. La corona tiene más área y es más redondeada
    
    Args:
        perfil_ancho: Perfil de ancho vertical del diente
        
    Returns:
        'crown_up' o 'crown_down'
    """
    altura = len(perfil_ancho)
    
    tercio_superior = perfil_ancho[:altura//3]
    tercio_inferior = perfil_ancho[2*altura//3:]
    
    ancho_superior = np.mean(tercio_superior) if len(tercio_superior) > 0 else 0
    ancho_inferior = np.mean(tercio_inferior) if len(tercio_inferior) > 0 else 0
    
    max_superior = np.max(tercio_superior) if len(tercio_superior) > 0 else 0
    max_inferior = np.max(tercio_inferior) if len(tercio_inferior) > 0 else 0
    
    if ancho_superior > ancho_inferior * CROWN_WIDTH_THRESHOLD and max_superior > max_inferior:
        return 'crown_up'
    elif ancho_inferior > ancho_superior * CROWN_WIDTH_THRESHOLD and max_inferior > max_superior:
        return 'crown_down'
    else:
        indice_max = np.argmax(perfil_ancho)
        if indice_max < altura // 2:
            return 'crown_up'
        else:
            return 'crown_down'


def find_neck_line(perfil_ancho: np.ndarray, orientacion: str) -> int:
    """
    Encuentra la línea cervical (cuello del diente) buscando el punto de transición
    entre la corona (más ancha) y la raíz (más estrecha).
    
    El cuello es la constricción donde el diente cambia de la forma de la corona
    a la forma de la raíz. Este punto está en la zona de transición.
    
    Args:
        perfil_ancho: Perfil de ancho vertical del diente
        orientacion: 'crown_up' o 'crown_down'
        
    Returns:
        Índice de fila donde está el cuello
    """
    altura = len(perfil_ancho)
    
    perfil_suavizado = cv2.GaussianBlur(perfil_ancho.reshape(-1, 1), (15, 1), 0).flatten()
    
    ancho_maximo = np.max(perfil_suavizado)
    ancho_minimo = np.min(perfil_suavizado)
    ancho_promedio = np.mean(perfil_suavizado)
    
    if orientacion == 'crown_up':
        inicio_busqueda = int(altura * 0.35)
        fin_busqueda = int(altura * 0.58)
        
        zona_busqueda = perfil_suavizado[inicio_busqueda:fin_busqueda]
        
        if len(zona_busqueda) > 10:
            umbral_corona = ancho_maximo * 0.92
            
            zona_corona_fin = None
            for i in range(len(zona_busqueda)):
                if zona_busqueda[i] < umbral_corona:
                    zona_corona_fin = i
                    break
            
            if zona_corona_fin is not None and zona_corona_fin > 3:
                zona_despues_corona = zona_busqueda[zona_corona_fin:min(zona_corona_fin + 60, len(zona_busqueda))]
                
                if len(zona_despues_corona) > 20:
                    indice_min_despues = np.argmin(zona_despues_corona)
                    zona_despues_min = zona_despues_corona[indice_min_despues:]
                    
                    if len(zona_despues_min) > 10:
                        aumento_despues = np.max(zona_despues_min) - zona_despues_corona[indice_min_despues]
                        minimos_locales = []
                        for j in range(5, len(zona_despues_corona) - 5):
                            if (zona_despues_corona[j] < zona_despues_corona[j-3] and 
                                zona_despues_corona[j] < zona_despues_corona[j+3]):
                                minimos_locales.append(j)
                        
                        es_molar = (aumento_despues > ancho_maximo * 0.08) or (len(minimos_locales) > 1)
                        
                        if es_molar:
                            mitad_zona = len(zona_despues_corona) // 2
                            zona_segunda_mitad = zona_despues_corona[mitad_zona:]
                            if len(zona_segunda_mitad) > 0:
                                indice_min_segunda = np.argmin(zona_segunda_mitad)
                                indice_cuello = inicio_busqueda + zona_corona_fin + mitad_zona + indice_min_segunda
                                indice_cuello = max(int(altura * 0.55), min(int(altura * 0.60), indice_cuello))
                            else:
                                indice_cuello = inicio_busqueda + zona_corona_fin + indice_min_despues
                                indice_cuello = max(int(altura * 0.55), min(int(altura * 0.60), indice_cuello))
                        else:
                            zona_transicion = zona_busqueda[max(0, zona_corona_fin - 5):min(zona_corona_fin + 20, len(zona_busqueda))]
                            
                            if len(zona_transicion) > 5:
                                derivada_transicion = np.diff(zona_transicion)
                                derivada_suavizada = cv2.GaussianBlur(derivada_transicion.reshape(-1, 1), (5, 1), 0).flatten()
                                
                                if len(derivada_suavizada) > 0:
                                    indice_max_decrecimiento = np.argmin(derivada_suavizada)
                                    offset_inicio = max(0, zona_corona_fin - 5)
                                    indice_cuello = inicio_busqueda + offset_inicio + indice_max_decrecimiento + 1
                                else:
                                    indice_cuello = inicio_busqueda + zona_corona_fin
                            else:
                                indice_cuello = inicio_busqueda + zona_corona_fin
                    else:
                        zona_transicion = zona_busqueda[max(0, zona_corona_fin - 5):min(zona_corona_fin + 20, len(zona_busqueda))]
                        if len(zona_transicion) > 5:
                            derivada_transicion = np.diff(zona_transicion)
                            derivada_suavizada = cv2.GaussianBlur(derivada_transicion.reshape(-1, 1), (5, 1), 0).flatten()
                            if len(derivada_suavizada) > 0:
                                indice_max_decrecimiento = np.argmin(derivada_suavizada)
                                offset_inicio = max(0, zona_corona_fin - 5)
                                indice_cuello = inicio_busqueda + offset_inicio + indice_max_decrecimiento + 1
                            else:
                                indice_cuello = inicio_busqueda + zona_corona_fin
                        else:
                            indice_cuello = inicio_busqueda + zona_corona_fin
                else:
                    indice_cuello = inicio_busqueda + zona_corona_fin
            else:
                derivada = np.diff(zona_busqueda)
                derivada_suavizada = cv2.GaussianBlur(derivada.reshape(-1, 1), (9, 1), 0).flatten()
                
                indice_max_decrecimiento = np.argmin(derivada_suavizada)
                indice_cuello = inicio_busqueda + indice_max_decrecimiento + 2
            
            zona_despues_cuello = perfil_suavizado[indice_cuello:min(indice_cuello + 60, altura)]
            if len(zona_despues_cuello) > 20:
                ancho_en_cuello = perfil_suavizado[indice_cuello]
                ancho_min_despues = np.min(zona_despues_cuello[:20]) if len(zona_despues_cuello) > 20 else ancho_en_cuello
                ancho_max_despues = np.max(zona_despues_cuello)
                
                aumento_desde_min = ancho_max_despues - ancho_min_despues
                
                if indice_cuello >= altura * 0.48 and indice_cuello <= altura * 0.55:
                    inicio_busqueda_corona = int(altura * 0.30)
                    fin_busqueda_corona = int(altura * 0.50)
                    zona_corona = perfil_suavizado[inicio_busqueda_corona:fin_busqueda_corona]
                    
                    if len(zona_corona) > 10:
                        umbral_corona = ancho_maximo * 0.92
                        zona_corona_fin = None
                        for i in range(len(zona_corona) - 1, max(0, len(zona_corona) - 30), -1):
                            if zona_corona[i] >= umbral_corona:
                                zona_corona_fin = i
                                break
                        
                        if zona_corona_fin is not None:
                            indice_cuello = inicio_busqueda_corona + zona_corona_fin + 3
                        else:
                            derivada_corona = np.diff(zona_corona)
                            if len(derivada_corona) > 5:
                                derivada_suavizada = cv2.GaussianBlur(derivada_corona.reshape(-1, 1), (7, 1), 0).flatten()
                                indice_max_dec = np.argmin(derivada_suavizada)
                                indice_cuello = inicio_busqueda_corona + indice_max_dec + 2
                            else:
                                indice_cuello = int(altura * 0.425)
                    else:
                        indice_cuello = int(altura * 0.425)
                    
                    indice_cuello = max(int(altura * 0.40), min(int(altura * 0.45), indice_cuello))
                elif indice_cuello < altura * 0.50:
                    indice_cuello = max(int(altura * 0.40), min(int(altura * 0.50), indice_cuello))
            else:
                if indice_cuello < altura * 0.50:
                    indice_cuello = max(int(altura * 0.40), min(int(altura * 0.50), indice_cuello))
        else:
            indice_cuello = int(altura * 0.47)
    
    else:
        inicio_busqueda = int(altura * 0.50)
        fin_busqueda = int(altura * 0.70)
        
        zona_busqueda = perfil_suavizado[inicio_busqueda:fin_busqueda]
        
        if len(zona_busqueda) > 20:
            umbral_corona = ancho_maximo * 0.90
            
            zona_corona_inicio = None
            for i in range(len(zona_busqueda) - 1, -1, -1):
                if zona_busqueda[i] >= umbral_corona:
                    zona_corona_inicio = i
                    break
            
            if zona_corona_inicio is not None and zona_corona_inicio > 10:
                zona_antes_corona = zona_busqueda[:zona_corona_inicio]
                
                if len(zona_antes_corona) > 10:
                    inicio_minimos = max(0, len(zona_antes_corona) - len(zona_antes_corona) // 2)
                    
                    minimos_candidatos = []
                    for i in range(inicio_minimos, len(zona_antes_corona) - 3):
                        if (zona_antes_corona[i] < zona_antes_corona[i-2] and 
                            zona_antes_corona[i] < zona_antes_corona[i+2]):
                            minimos_candidatos.append((i, zona_antes_corona[i]))
                    
                    if minimos_candidatos:
                        minimos_candidatos.sort(key=lambda x: x[1])
                        indice_min_local = minimos_candidatos[0][0]
                        indice_cuello = inicio_busqueda + indice_min_local + 3
                    else:
                        zona_tercio = zona_antes_corona[inicio_minimos:]
                        if len(zona_tercio) > 0:
                            indice_min_tercio = np.argmin(zona_tercio)
                            indice_cuello = inicio_busqueda + inicio_minimos + indice_min_tercio + 2
                        else:
                            indice_cuello = inicio_busqueda + zona_corona_inicio - 5
                else:
                    indice_cuello = inicio_busqueda + zona_corona_inicio - 5
            else:
                derivada = np.diff(zona_busqueda)
                derivada_suavizada = cv2.GaussianBlur(derivada.reshape(-1, 1), (7, 1), 0).flatten()
                
                mitad_zona = len(derivada_suavizada) // 2
                zona_crecimiento = derivada_suavizada[mitad_zona:]
                
                if len(zona_crecimiento) > 0:
                    indice_max_crecimiento_relativo = np.argmax(zona_crecimiento)
                    indice_max_crecimiento = mitad_zona + indice_max_crecimiento_relativo
                    
                    if indice_max_crecimiento > 10:
                        zona_antes = zona_busqueda[:indice_max_crecimiento]
                        if len(zona_antes) > 5:
                            inicio_busqueda_min = max(0, len(zona_antes) - len(zona_antes) // 3)
                            zona_min = zona_antes[inicio_busqueda_min:]
                            if len(zona_min) > 0:
                                indice_min = np.argmin(zona_min)
                                indice_cuello = inicio_busqueda + inicio_busqueda_min + indice_min + 2
                            else:
                                indice_cuello = inicio_busqueda + indice_max_crecimiento - 8
                        else:
                            indice_cuello = inicio_busqueda + indice_max_crecimiento - 8
                    else:
                        indice_cuello = int(altura * 0.63)
                else:
                    indice_cuello = int(altura * 0.63)
            
            indice_cuello = max(int(altura * 0.58), min(int(altura * 0.68), indice_cuello))
        else:
            indice_cuello = int(altura * 0.60)
    
    if indice_cuello < altura * 0.20 or indice_cuello > altura * 0.80:
        indice_cuello = int(altura * DEFAULT_NECK_RATIO)
    
    return int(indice_cuello)


def split_tooth(imagen: np.ndarray, indice_cuello: int, orientacion: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separa el diente en corona y raíz según la línea cervical.
    
    Mantiene el tamaño original del lienzo, solo recorta verticalmente.
    
    Args:
        imagen: Imagen original del diente
        indice_cuello: Índice de fila donde está el cuello
        orientacion: 'crown_up' o 'crown_down'
        
    Returns:
        Tupla (corona, raiz) - ambas con el mismo tamaño que la imagen original
    """
    altura, ancho = imagen.shape[:2]
    
    if imagen.shape[2] == 4:
        corona = np.zeros_like(imagen)
        raiz = np.zeros_like(imagen)
    else:
        corona = np.zeros_like(imagen)
        raiz = np.zeros_like(imagen)
    
    if orientacion == 'crown_up':
        corona[0:indice_cuello, :] = imagen[0:indice_cuello, :]
        raiz[indice_cuello:altura, :] = imagen[indice_cuello:altura, :]
    else:
        raiz[0:indice_cuello, :] = imagen[0:indice_cuello, :]
        corona[indice_cuello:altura, :] = imagen[indice_cuello:altura, :]
    
    return corona, raiz


def save_debug(imagen: np.ndarray, indice_cuello: int, orientacion: str, 
               ruta_salida: Path) -> None:
    """
    Guarda una imagen de debug con la línea de corte y la orientación detectada.
    
    Args:
        imagen: Imagen original del diente
        indice_cuello: Índice de fila donde está el cuello
        orientacion: 'crown_up' o 'crown_down'
        ruta_salida: Ruta donde guardar la imagen de debug
    """
    debug_img = imagen.copy()
    
    if debug_img.shape[2] == 4:
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGRA2BGR)
    
    altura, ancho = debug_img.shape[:2]
    
    cv2.line(debug_img, (0, indice_cuello), (ancho, indice_cuello), (0, 0, 255), 2)
    
    texto_orientacion = f"Corona: {'ARRIBA' if orientacion == 'crown_up' else 'ABAJO'}"
    cv2.putText(debug_img, texto_orientacion, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    texto_cuello = f"Cuello: {indice_cuello}px ({indice_cuello/altura*100:.1f}%)"
    cv2.putText(debug_img, texto_cuello, (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imwrite(str(ruta_salida), debug_img)


def procesar_imagen(ruta_imagen: Path, carpeta_salida: Path, generar_debug: bool = True) -> bool:
    """
    Procesa una imagen de diente: detecta orientación, encuentra el cuello,
    separa en corona y raíz, y guarda los resultados.
    
    Args:
        ruta_imagen: Ruta a la imagen del diente
        carpeta_salida: Carpeta donde guardar los resultados
        generar_debug: Si True, genera imagen de debug
        
    Returns:
        True si el procesamiento fue exitoso, False en caso contrario
    """
    imagen = cv2.imread(str(ruta_imagen), cv2.IMREAD_UNCHANGED)
    
    if imagen is None:
        print(f"❌ Error: No se pudo leer {ruta_imagen}")
        return False
    
    mascara = crear_mascara_diente(imagen)
    perfil_ancho = calcular_perfil_ancho(mascara)
    
    if len(perfil_ancho) == 0 or np.max(perfil_ancho) == 0:
        print(f"❌ Error: No se detectó el diente en {ruta_imagen}")
        return False
    
    orientacion = detect_orientation(perfil_ancho)
    indice_cuello = find_neck_line(perfil_ancho, orientacion)
    corona, raiz = split_tooth(imagen, indice_cuello, orientacion)
    
    nombre_base = ruta_imagen.stem
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    
    ruta_corona = carpeta_salida / f"{nombre_base}_corona.png"
    ruta_raiz = carpeta_salida / f"{nombre_base}_raiz.png"
    
    if imagen.shape[2] == 4:
        cv2.imwrite(str(ruta_corona), corona)
        cv2.imwrite(str(ruta_raiz), raiz)
    else:
        cv2.imwrite(str(ruta_corona), corona)
        cv2.imwrite(str(ruta_raiz), raiz)
    
    if generar_debug:
        ruta_debug = carpeta_salida / f"{nombre_base}_debug.png"
        save_debug(imagen, indice_cuello, orientacion, ruta_debug)
    
    orientacion_texto = "ARRIBA" if orientacion == 'crown_up' else "ABAJO"
    print(f"✓ {nombre_base}.png -> corona ({orientacion_texto}), cuello en {indice_cuello}px")
    
    return True


def procesar_carpeta(carpeta_entrada: str = "FDI", carpeta_salida: str = "FDI_SPLIT", 
                     generar_debug: bool = True) -> None:
    """
    Procesa todas las imágenes PNG en una carpeta.
    
    Args:
        carpeta_entrada: Carpeta con las imágenes de entrada
        carpeta_salida: Carpeta donde guardar las imágenes separadas
        generar_debug: Si True, genera imágenes de debug
    """
    carpeta_path = Path(carpeta_entrada)
    carpeta_salida_path = Path(carpeta_salida)
    
    if not carpeta_path.exists():
        print(f"❌ Error: La carpeta {carpeta_entrada} no existe")
        return
    
    imagenes = [img for img in carpeta_path.glob("*.png") 
                if not img.name.endswith("_corona.png") and 
                   not img.name.endswith("_raiz.png") and
                   not img.name.endswith("_debug.png")]
    
    if not imagenes:
        print(f"⚠ No se encontraron imágenes PNG en {carpeta_entrada}")
        return
    
    print(f"🔍 Procesando {len(imagenes)} imágenes...\n")
    
    exitosas = 0
    for imagen in sorted(imagenes):
        if procesar_imagen(imagen, carpeta_salida_path, generar_debug):
            exitosas += 1
    
    print(f"\n✅ Procesadas {exitosas} de {len(imagenes)} imágenes correctamente")
    print(f"📁 Resultados guardados en: {carpeta_salida_path.absolute()}")


if __name__ == "__main__":
    procesar_carpeta(
        carpeta_entrada="FDI",
        carpeta_salida="FDI_SPLIT",
        generar_debug=True
    )
