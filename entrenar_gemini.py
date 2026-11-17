import os
import time
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

# CONFIGURACIÓN Y CONECTAR CON GOOGLE CLOUD

load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
# Quitamos "/" final si lo hubiera
BUCKET_URI = os.getenv("GCS_BUCKET_URI").rstrip("/")  
BASE_MODEL = os.getenv("GEMINI_BASE_MODEL")

# Creamos el cliente para comunicarnos con Vertex AI (la plataforma de IA de Google)
cliente = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION
)


# FUNCIONES PARA ORGANIZAR EL PROCESO
def esperar_fin_entrenamiento(trabajo):
    """
    Revisa cada 60 segundos cómo va el entrenamiento en Google Cloud
    hasta que termine (ya sea con éxito o con error).
    """
    # 3 estados 
    estados_en_marcha = {
        "JOB_STATE_QUEUED",  
        "JOB_STATE_PENDING",  
        "JOB_STATE_RUNNING"   
    }

    print(" Esperando a que Google Cloud termine de entrenar el modelo...")
    
    # Mientras el estado actual esté dentro de los estados en marcha, seguimos esperando
    while trabajo.state.name in estados_en_marcha:
         # Pausamos el programa 1 minuto para no saturar a Google con preguntas
        time.sleep(60) 
        # Pedimos el estado actualizado
        trabajo = cliente.tunings.get(name=trabajo.name)  
        print(f" Estado actual: {trabajo.state.name}...")

    return trabajo


def guardar_evidencia(trabajo):
    """
    Guarda en un archivo JSON local todos los datos importantes del modelo
    ya entrenado para poder usarlo después en el examen final.
    """
    datos_evidencia = {
        "estado": trabajo.state.name,
        "region_entrenamiento": LOCATION,
        "modelo_base": BASE_MODEL,
        "trabajo_tuning_id": trabajo.name,
        "modelo_ajustado_endpoint": trabajo.tuned_model.endpoint,
        "modelo_ajustado_nombre": trabajo.tuned_model.model
    }

    with open("evidencia_tuning_gemini.json", "w", encoding="utf-8") as archivo:
        json.dump(datos_evidencia, archivo, ensure_ascii=False, indent=2)

    print("💾 Evidencia guardada correctamente en 'evidencia_tuning_gemini.json'.")


#  PROGRAMA PRINCIPAL (LANZAR Y CONTROLAR EL ENTRENAMIENTO)

if __name__ == "__main__":
    #  Definimos dónde están nuestros archivos ya subidos en la nube
    ruta_train_nube = f"{BUCKET_URI}/datos/train_gemini.jsonl"
    ruta_val_nube = f"{BUCKET_URI}/datos/val_gemini.jsonl"

    print(f"Iniciando Fine-Tuning sobre '{BASE_MODEL}'...")

    #  Preparamos las piezas de configuración por separado para que sea fácil de leer
    datos_entrenamiento = types.TuningDataset(gcs_uri=ruta_train_nube)
    datos_validacion = types.TuningValidationDataset(gcs_uri=ruta_val_nube)
    
    configuracion_tuning = types.CreateTuningJobConfig(
        validation_dataset=datos_validacion,
        tuned_model_display_name="gemini-31-flash-lite-incidencias-erika"
    )

    # Enviamos la orden oficial a Google Cloud para crear el trabajo de entrenamiento
    trabajo_tuning = cliente.tunings.tune(
        base_model=BASE_MODEL,
        training_dataset=datos_entrenamiento,
        config=configuracion_tuning
    )

    print(f" Trabajo creado con éxito.")

    #  Llamamos a nuestra función para esperar tranquilamente a que termine
    trabajo_finalizado = esperar_fin_entrenamiento(trabajo_tuning)

    #  Revisamos el resultado final
    if trabajo_finalizado.state.name == "JOB_STATE_SUCCEEDED":
        print("\n ¡ENTRENAMIENTO COMPLETADO CON ÉXITO!")
        print("Endpoint del modelo ajustado:", trabajo_finalizado.tuned_model.endpoint)
        
        guardar_evidencia(trabajo_finalizado)
    else:
        print(f"\n El entrenamiento falló. Estado final: {trabajo_finalizado.state.name}")
        print("Detalle del error:", trabajo_finalizado.error)
