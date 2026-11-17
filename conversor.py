import os
import json
from dotenv import load_dotenv
from google.cloud import storage

# CARGAMOS LAS VARIABLES DE CONFIGURACIÓN

load_dotenv()

# Guardamos en una variable la dirección de nuestra carpeta en Google Cloud
BUCKET_URI = os.getenv("GCS_BUCKET_URI")

# Escribimos las instrucciones fijas que queremos que Gemini aprenda
PROMPT_SISTEMA = """Clasifica la incidencia y responde SOLO con un JSON válido con estas claves:
- "categoria": "facturacion", "soporte_tecnico", "cuenta" o "envios"
- "prioridad": "baja", "media" o "alta"
- "respuesta": respuesta corta para el cliente.
No escribas nada fuera del JSON."""


#  FUNCIÓN PARA PREPARAR UN SOLO EJEMPLO

def formatear_para_gemini(ejemplo):
    """
    Recibe un diccionario con 'incidencia' y 'salida'.
    Devuelve ese mismo ejemplo pero con la estructura exacta que pide Google.
    """
    # Sacamos los datos de nuestro ejemplo original para tenerlos a mano
    texto_pregunta = ejemplo["incidencia"]
    diccionario_respuesta = ejemplo["salida"]

    # Convertimos el diccionario de respuesta en un texto plano (formato JSON)
    texto_respuesta = json.dumps(diccionario_respuesta, ensure_ascii=False)

    # Construimos el molde oficial de Google Gemini
    molde_gemini = {
        "systemInstruction": {
            "parts": [
                {"text": PROMPT_SISTEMA}
            ]
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": texto_pregunta}
                ]
            },
            {
                "role": "model",
                "parts": [
                    {"text": texto_respuesta}
                ]
            }
        ]
    }

    return molde_gemini


# FUNCIÓN PARA TRANSFORMAR UN ARCHIVO COMPLETO

def convertir_a_jsonl_gemini(ruta_entrada, ruta_salida_jsonl, prefijo_id):
    """
    Lee un archivo .json completo, transforma cada ejemplo y lo guarda
    línea por línea en un archivo nuevo (.jsonl).
    """
    # Abrimos el archivo original en modo lectura ("r") y cargamos la lista de ejemplos
    with open(ruta_entrada, "r", encoding="utf-8") as archivo_lectura:
        lista_ejemplos = json.load(archivo_lectura)

    # Creamos una lista vacía donde iremos apuntando el control de cada línea
    mapa_identificadores = []

    # Abrimos el archivo nuevo en modo escritura  para ir guardando los resultados
    with open(ruta_salida_jsonl, "w", encoding="utf-8") as archivo_escritura:
        
        # Recorremos la lista de ejemplos usando enumerate para tener el número de línea automáticamente
        for numero_linea, ejemplo in enumerate(lista_ejemplos, start=1):
            
            # Si el ejemplo ya trae un "id", lo usamos. Si no, creamos uno (ej: train_001)
            if "id" in ejemplo:
                id_actual = ejemplo["id"]
            else:
                id_actual = f"{prefijo_id}_{numero_linea:03d}"

            # Guardamos los datos de control en nuestro mapa para revisarlos en el futuro
            ficha_control = {
                "linea_jsonl": numero_linea,
                "id_original": id_actual,
                "categoria_esperada": ejemplo["salida"]["categoria"],
                "prioridad_esperada": ejemplo["salida"]["prioridad"]
            }
            mapa_identificadores.append(ficha_control)

            # Llamamos a nuestra función FORMATEAR_PARA_GEMINI  para darle forma de Gemini
            ejemplo_listo = formatear_para_gemini(ejemplo)

            # Convertimos ese ejemplo a una línea de texto 
            linea_texto = json.dumps(ejemplo_listo, ensure_ascii=False)
            archivo_escritura.write(linea_texto + "\n")

    # 4. Devolvemos la lista con el control de todas las líneas procesadas
    return mapa_identificadores


# FUNCIÓN PARA SUBIR UN ARCHIVO A GOOGLE CLOUD

def subir_a_bucket_gcs(archivo_local, nombre_en_nube):
    """
    Toma un archivo de nuestro ordenador y lo envía a nuestra carpeta en Google Cloud.
    """
    #  Limpiamos la dirección del bucket 
    sin_prefijo = BUCKET_URI.replace("gs://", "")
    partes_ruta = sin_prefijo.split("/")
    nombre_bucket = partes_ruta[0].strip()

    #  Nos conectamos paso a paso con Google Cloud Storage
    cliente_storage = storage.Client()
    mi_bucket = cliente_storage.bucket(nombre_bucket)
    archivo_destino = mi_bucket.blob(nombre_en_nube)

    # Subimos el archivo desde nuestro ordenador hacia ese destino en la nube
    archivo_destino.upload_from_filename(archivo_local)
    print(f" Subido correctamente a: gs://{nombre_bucket}/{nombre_en_nube}")


# PROGRAMA DONDE SE EJECUTA TODO, PROGRAMA PRINCIPAL
if __name__ == "__main__":
    # Creamos la carpeta 'datos_gemini' si aún no existe en nuestro ordenador
    os.makedirs("datos_gemini", exist_ok=True)
    print(" Iniciando conversión de archivos...")

    # Procesamos el archivo de ENTRENAMIENTO
    mapa_train = convertir_a_jsonl_gemini(
        ruta_entrada="datos/entrenamiento_ejemplos.json",
        ruta_salida_jsonl="datos_gemini/train_gemini.jsonl",
        prefijo_id="train"
    )

    # Procesamos el archivo de VALIDACIÓN
    mapa_val = convertir_a_jsonl_gemini(
        ruta_entrada="datos/validacion_ejemplos.json",
        ruta_salida_jsonl="datos_gemini/val_gemini.jsonl",
        prefijo_id="val"
    )

    # Guardamos un archivo resumen con los identificadores de ambos procesos
    resumen_completo = {
        "entrenamiento": mapa_train,
        "validacion": mapa_val
    }
    with open("datos_gemini/relacion_identificadores.json", "w", encoding="utf-8") as archivo_mapa:
        json.dump(resumen_completo, archivo_mapa, ensure_ascii=False, indent=2)

    print(" Archivos creados correctamente en la carpeta 'datos_gemini/'.")

    #  Si tenemos configurada una carpeta en Google Cloud, subimos los dos archivos
    if BUCKET_URI:
        print(" Subiendo archivos a Google Cloud...")
        subir_a_bucket_gcs("datos_gemini/train_gemini.jsonl", "datos/train_gemini.jsonl")
        subir_a_bucket_gcs("datos_gemini/val_gemini.jsonl", "datos/val_gemini.jsonl")
    else:
        print(" No se encontró GCS_BUCKET_URI en el .env. Se omitió la subida a la nube.")
