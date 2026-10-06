import os
import httpx
from fastapi import FastAPI, Request

app = FastAPI()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()

conversation_history = {}
MAX_HISTORY_TURNS = 10 

SYSTEM_INSTRUCTION = """
Tu és o Isaac, um assistente virtual de elite focado em apoio técnico, robótica, automação e organização profissional.
Responde sempre de forma clara, objetiva, prestativa e profissional.
"""

def get_gemini_contents(chat_id: int, new_prompt: str):
    contents = []
    history = conversation_history.get(chat_id, [])
    
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["text"]}]})
        
    contents.append({"role": "user", "parts": [{"text": new_prompt}]})
    return contents

def save_to_history(chat_id: int, user_text: str, bot_text: str):
    if chat_id not in conversation_history:
        conversation_history[chat_id] = []
    conversation_history[chat_id].append({"role": "user", "text": user_text})
    conversation_history[chat_id].append({"role": "model", "text": bot_text})
    if len(conversation_history[chat_id]) > MAX_HISTORY_TURNS * 2:
        conversation_history[chat_id] = conversation_history[chat_id][-(MAX_HISTORY_TURNS * 2):]

def clear_history(chat_id: int):
    if chat_id in conversation_history:
        conversation_history[chat_id] = []

async def process_with_isaac(chat_id: int, prompt: str) -> str:
    if not GEMINI_API_KEY:
        return "Erro: A chave GEMINI_API_KEY não foi configurada no Render."

    contents = get_gemini_contents(chat_id, prompt)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    payload = {
        "contents": contents,
        "systemInstruction": {
            "parts": [{"text": SYSTEM_INSTRUCTION}]
        },
        "generationConfig": {
            "temperature": 0.7
        }
    }
    
    headers = {"Content-Type": "application/json"}
    
    async with httpx.AsyncClient() as http_client:
        try:
            response = await http_client.post(url, json=payload, headers=headers, timeout=30.0)
            if response.status_code == 200:
                data = response.json()
                bot_reply = data["candidates"][0]["content"]["parts"][0]["text"]
                save_to_history(chat_id, prompt, bot_reply)
                return bot_reply
            else:
                print(f"Erro Gemini: {response.status_code} - {response.text}")
                err_msg = response.json().get('error', {}).get('message', 'Erro desconhecido')
                return f"Erro Gemini {response.status_code}: {err_msg}"
        except Exception as e:
            print(f"Exceção ao chamar Gemini: {e}")
            return "O Isaac está temporariamente indisponível. Tente novamente em instantes."

@app.get("/")
def home():
    return {"status": "Isaac está online via Gemini!"}

@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    if "message" in data and "text" in data["message"]:
        chat_id = data["message"]["chat"]["id"]
        user_message = data["message"]["text"].strip()
        
        if user_message == "/start":
            clear_history(chat_id)
            reply = (
                "👋 **Olá! Eu sou o Isaac.**\n\n"
                "Estou pronto para ajudar em robótica, automação, programação e organização de projetos.\n\n"
                "💡 **Dica:** Para reiniciar a nossa conversa e limpar a memória, envia `/limpar`."
            )
        elif user_message in ["/limpar", "/reset"]:
            clear_history(chat_id)
            reply = "🧹 **Memória limpa com sucesso!** Podemos começar um novo assunto do zero."
        else:
            reply = await process_with_isaac(chat_id, user_message)
        
        if TELEGRAM_BOT_TOKEN:
            telegram_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            async with httpx.AsyncClient() as http_client:
                await http_client.post(telegram_url, json={
                    "chat_id": chat_id,
                    "text": reply,
                    "parse_mode": "Markdown"
                })
                
    return {"status": "ok"}
