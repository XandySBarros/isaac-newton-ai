import os
import time
import httpx
from fastapi import FastAPI, Request
from google import genai
from google.genai import types

app = FastAPI()

# Inicialização de Clientes
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

# Instruções de Sistema do Isaac
SYSTEM_INSTRUCTION = """
Tu és o Isaac, um assistente virtual de elite focado em apoio técnico, robótica, automação e organização profissional.
Responda sempre de forma clara, objetiva, prestativa e profissional.
"""

# Lista de modelos em ordem de preferência (com reserva caso haja sobrecarga 503)
MODELS_TO_TRY = ["gemini-3.8-flash", "gemini-flash-latest"]

# =====================================================================
# NÚCLEO CENTRAL DE IA (Com Fallback e Tratamento de Erros)
# =====================================================================
def process_with_isaac(prompt: str) -> str:
    for model_name in MODELS_TO_TRY:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION
                    )
                )
                if response.text:
                    return response.text
            except Exception as e:
                error_str = str(e)
                # Se for erro 503 (sobrecarga), aguarda 1s e tenta novamente
                if "503" in error_str:
                    time.sleep(1)
                    continue
                # Se o modelo falhar por outro motivo, passa para o próximo modelo da lista
                break

    return "O Isaac está temporariamente indisponível devido a alta demanda nos servidores da Google. Por favor, tente novamente em alguns instantes."


# =====================================================================
# ENDPOINTS DAS PLATAFORMAS
# =====================================================================
@app.get("/")
def home():
    return {"status": "Isaac está online!"}

@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    
    if "message" in data and "text" in data["message"]:
        chat_id = data["message"]["chat"]["id"]
        user_message = data["message"]["text"]
        
        # Processa a resposta usando o sistema de fallback
        reply = process_with_isaac(user_message)
        
        # Envia a resposta no Telegram
        if TELEGRAM_BOT_TOKEN:
            telegram_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            async with httpx.AsyncClient() as http_client:
                await http_client.post(telegram_url, json={
                    "chat_id": chat_id,
                    "text": reply
                })
                
    return {"status": "ok"}
