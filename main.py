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

# =====================================================================
# NÚCLEO CENTRAL DE IA (Com tratamento de erro e retentativa)
# =====================================================================
def process_with_isaac(prompt: str) -> str:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION
                )
            )
            return response.text
        except Exception as e:
            # Se for erro de sobrecarga (503) e ainda houver tentativas, aguarda 2s e tenta de novo
            if "503" in str(e) and attempt < max_retries - 1:
                time.sleep(2)
                continue
            
            # Se esgotar as tentativas ou for outro erro
            return "Ocorreu um pico temporário nos servidores do Google. Por favor, envie sua mensagem novamente em alguns instantes."


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
        
        # Processa a resposta
        reply = process_with_isaac(user_message)
        
        # Envia de volta ao Telegram
        if TELEGRAM_BOT_TOKEN:
            telegram_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            async with httpx.AsyncClient() as http_client:
                await http_client.post(telegram_url, json={
                    "chat_id": chat_id,
                    "text": reply
                })
                
    return {"status": "ok"}
