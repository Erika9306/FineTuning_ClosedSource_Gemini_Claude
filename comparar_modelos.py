import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI

# LEEMOS LAS CLAVES DESDE .ENV
load_dotenv()

ID_PROYECTO = os.getenv("GOOGLE_CLOUD_PROJECT")
REGION_GEMINI = os.getenv("GEMINI_ENDPOINT_LOCATION")  
MODELO_GEMINI_BASE = os.getenv("GEMINI_BASE_MODEL")    
OPENROUTER_URL = os.getenv("OPENROUTER_BASE_URL")
CLAVE_OPENROUTER = os.getenv("OPENROUTER_API_KEY")
MODELO_CLAUDE = os.getenv("CLAUDE_MODEL_ID")           

PROMPT = """Clasifica la incidencia y responde SOLO con un JSON válido con estas claves:
- "categoria": "facturacion", "soporte_tecnico", "cuenta" o "envios"
- "prioridad": "baja", "media" o "alta"
- "respuesta": respuesta corta para el cliente.
No escribas nada fuera del JSON."""

# CARGAMOS LOS ARCHIVOS
with open("datos/entrenamiento_ejemplos.json", "r", encoding="utf-8") as archivo:
    datos_entrenamiento = json.load(archivo)

with open("datos/test_ejemplos.json", "r", encoding="utf-8") as archivo:
    datos_test = json.load(archivo)

with open("evidencia_tuning_gemini.json", "r", encoding="utf-8") as archivo:
    datos_entrenamiento_gemini = json.load(archivo)
MODELO_GEMINI_ENTRENO = datos_entrenamiento_gemini["modelo_ajustado_endpoint"]

# Elegimos 4 ejemplos (uno de cada categoría) para Claude
ejemplos = []
categorias = []

for ejemplo in datos_entrenamiento:
    categoria = ejemplo["salida"]["categoria"]
    if categoria not in categorias:
        ejemplos.append(ejemplo)
        categorias.append(categoria)
    if len(ejemplos) == 4:
        break

#  CONECTAMOS CON GEMINI Y CLAUDE
cliente_google = genai.Client(vertexai=True, project=ID_PROYECTO, location=REGION_GEMINI)
cliente_openrouter = OpenAI(base_url=OPENROUTER_URL, api_key=CLAVE_OPENROUTER)

reglas_gemini = types.GenerateContentConfig(
    system_instruction=PROMPT,
    temperature=0.0,
    max_output_tokens=150,
    thinking_config=types.ThinkingConfig(thinking_level="MINIMAL")
)

#  FUNCIONES PARA EXAMINAR A LOS MODELOS 
def examinar_gemini(nombre_modelo, titulo_prueba):
    print(f"\n--- Probando {titulo_prueba} ---")
    respuestas_guardadas = []
    tiempos_respuesta = []
    tokens_entrada = 0
    tokens_salida = 0

    for numero, caso in enumerate(datos_test):
        hora_inicio = time.perf_counter()
        
        respuesta = cliente_google.models.generate_content(
            model=nombre_modelo,
            contents=caso["incidencia"],
            config=reglas_gemini
        )
        
        tiempos_respuesta.append(time.perf_counter() - hora_inicio)
        respuestas_guardadas.append(respuesta.text if respuesta.text else "")

        if respuesta.usage_metadata:
            tokens_entrada += respuesta.usage_metadata.prompt_token_count or 0
            tokens_salida += respuesta.usage_metadata.candidates_token_count or 0

        if (numero + 1) % 10 == 0:
            print(f"   Llevamos {numero + 1} de 30 casos...")

    # Devolvemos todo guardado en un diccionario
    return {
        "respuestas": respuestas_guardadas,
        "media_tiempo": round(sum(tiempos_respuesta) / len(datos_test), 2),
        "tokens": f"{tokens_entrada} / {tokens_salida}",
        "coste": "Ver consola GCP"
    }

def examinar_claude():
    print("\n--- Probando Claude Haiku 4.5 (Con 4 ejemplos) ---")
    respuestas_guardadas = []
    tiempos_respuesta = []
    tokens_entrada = 0
    tokens_salida = 0
    coste_dolares = 0.0
    hay_datos_coste = False

    historial_ejemplos = [{"role": "system", "content": PROMPT}]
    for ej in ejemplos:
        historial_ejemplos.append({"role": "user", "content": ej["incidencia"]})
        historial_ejemplos.append({"role": "assistant", "content": json.dumps(ej["salida"], ensure_ascii=False)})

    for numero, caso in enumerate(datos_test):
        mensajes_enviar = historial_ejemplos + [{"role": "user", "content": caso["incidencia"]}]
        
        hora_inicio = time.perf_counter()
        respuesta = cliente_openrouter.chat.completions.create(
            model=MODELO_CLAUDE,
            messages=mensajes_enviar,
            temperature=0.0,
            max_tokens=150
        )
        tiempos_respuesta.append(time.perf_counter() - hora_inicio)
        respuestas_guardadas.append(respuesta.choices[0].message.content or "")

        if respuesta.usage:
            tokens_entrada += respuesta.usage.prompt_tokens or 0
            tokens_salida += respuesta.usage.completion_tokens or 0
            coste = getattr(respuesta.usage, "cost", None)
            if coste is not None:
                coste_dolares += float(coste)
                hay_datos_coste = True

        if (numero + 1) % 10 == 0:
            print(f"   Llevamos {numero + 1} de 30 casos...")

    texto_coste = f"${coste_dolares:.6f}" if hay_datos_coste else "No devuelto por API"
    
    # Devolvemos todo guardado en un diccionario
    return {
        "respuestas": respuestas_guardadas,
        "media_tiempo": round(sum(tiempos_respuesta) / len(datos_test), 2),
        "tokens": f"{tokens_entrada} / {tokens_salida}",
        "coste": texto_coste
    }

# FUNCIÓN PARA CORREGIR CALCULAR NOTAS
def calcular_notas(respuestas, casos_test):
    total = len(casos_test)
    validos = 0
    aciertos_cat = 0
    aciertos_prio = 0

    for resp_texto, caso in zip(respuestas, casos_test):
        try:
            datos = json.loads(resp_texto)
            validos += 1
            if datos.get("categoria") == caso["salida"]["categoria"]:
                aciertos_cat += 1
            if datos.get("prioridad") == caso["salida"]["prioridad"]:
                aciertos_prio += 1
        except json.JSONDecodeError:
            pass

    porcentaje_json = round((validos / total) * 100, 1)
    porcentaje_cat = round((aciertos_cat / total) * 100, 1)
    porcentaje_prio = round((aciertos_prio / total) * 100, 1)
    
    return porcentaje_json, porcentaje_cat, porcentaje_prio

#  EJECUTA TODO, GUARDA RESULTADOS E IMPRIME LA TABLA FINAL
if __name__ == "__main__":
    # Hacemos las 3 pruebas y guardamos cada resultado en una sola variable
    res_gb = examinar_gemini(MODELO_GEMINI_BASE, " Gemini Base")
    res_ge = examinar_gemini(MODELO_GEMINI_ENTRENO, " Gemini Entrenado")
    res_cl = examinar_claude()

    # Guardamos todas las respuestas en un archivo
    with open("resultados_originales_sesion8.json", "w", encoding="utf-8") as archivo:
        json.dump({
            "4_gemini_base": res_gb["respuestas"],
            "5_gemini_entrenado": res_ge["respuestas"],
            "6_claude_haiku_fewshot": res_cl["respuestas"]
        }, archivo, ensure_ascii=False, indent=2)

    # Calculamos las notas usando el nombre correcto de la función
    nota_gb = calcular_notas(res_gb["respuestas"], datos_test)
    nota_ge = calcular_notas(res_ge["respuestas"], datos_test)
    nota_cl = calcular_notas(res_cl["respuestas"], datos_test)

    # Mostramos la tabla final
    print("\n" + "="*90)
    print("### TABLA COMPARATIVA FINAL DE 6 CONDICIONES (Sobre 30 casos de Test)")
    print("="*90)
    print("| Condición Evaluada | JSON Válido | Cat. Exacta | Prio. Exacta | Latencia Media | Tokens (In / Out) | Coste Predicción |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    print("| 1. Qwen3-0.6B Base (Zero-Shot) | 80.0% | 20.0% | 40.0% | — (Sesión 7) | Local / Colab T4 | $0 (GPU gratuita) |")
    print("| 2. Qwen3-0.6B Base (Few-Shot) | 100.0% | 80.0% | 30.0% | — (Sesión 7) | Local / Colab T4 | $0 (GPU gratuita) |")
    print("| 3. Qwen3-0.6B Ajustado (LoRA) | 100.0% | 56.7% | 50.0% | — (Sesión 7) | Local / Colab T4 | $0 (GPU gratuita) |")
    print(f"| 4. Gemini 3.1 Flash-Lite Base | {nota_gb[0]}% | {nota_gb[1]}% | {nota_gb[2]}% | {res_gb['media_tiempo']}s | {res_gb['tokens']} | {res_gb['coste']} |")
    print(f"| 5. Gemini 3.1 Flash-Lite Tuned | {nota_ge[0]}% | {nota_ge[1]}% | {nota_ge[2]}% | {res_ge['media_tiempo']}s | {res_ge['tokens']} | {res_ge['coste']} |")
    print(f"| 6. Claude Haiku 4.5 (Few-Shot) | {nota_cl[0]}% | {nota_cl[1]}% | {nota_cl[2]}% | {res_cl['media_tiempo']}s | {res_cl['tokens']} | {res_cl['coste']} |")
