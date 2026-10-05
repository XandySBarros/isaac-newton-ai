import os
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
# NÚCLEO CENTRAL DE IA (Reutilizável por qualquer plataforma)
# =====================================================================
def process_with_isaac(prompt: str) -> str:
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION
            )
        )
        return response.text
    except Exception as e:
        return f"Erro interno no Isaac: {str(e)}"


# =====================================================================
# PLATAFORMA 1: Navegador / Teste REST API
# =====================================================================
@app.get("/")
def home():
    return {"status": "Isaac esta online e pronto para servir!"}

@app.get("/chat")
def web_chat(prompt: str = "Ola"):
    return {"resposta": process_with_isaac(prompt)}


# =====================================================================
# PLATAFORMA 2: Telegram Bot (Webhook)
# =====================================================================
@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    print(f"--> [WEBHOOK RECEBIDO]: {data}")
    
    if "message" in data and "text" in data["message"]:
        chat_id = data["message"]["chat"]["id"]
        user_message = data["message"]["text"]
        
        # Gera resposta
        reply = process_with_isaac(user_message)
        print(f"--> [ISAAC RESPOSTA]: {reply}")
        
        # Verifica Token
        if not TELEGRAM_BOT_TOKEN:
            print("--> [ERRO CRÍTICO]: TELEGRAM_BOT_TOKEN não foi encontrado nas variáveis do Render!")
        else:
            telegram_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            async with httpx.AsyncClient() as http_client:
                res = await http_client.post(telegram_url, json={
                    "chat_id": chat_id,
                    "text": reply
                })
                print(f"--> [ENVIO TELEGRAM]: Código {res.status_code} - {res.text}")
                
    return {"status": "ok"}
