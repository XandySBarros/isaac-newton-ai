import os
import httpx
from fastapi import FastAPI, Request

app = FastAPI()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

conversation_history = {}
MAX_HISTORY_TURNS = 10 

SYSTEM_INSTRUCTION = """
Tu és o Isaac, um assistente virtual de elite focado em apoio técnico, robótica, automação e organização profissional.
Responda sempre de forma clara, objetiva, prestativa e profissional.
"""

def get_messages_payload(chat_id: int, new_prompt: str):
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    history = conversation_history.get(chat_id, [])
    for msg in history:
        messages.append(msg)
    messages.append({"role": "user", "content": new_prompt})
    return messages

def save_to_history(chat_id: int, user_text: str, bot_text: str):
    if chat_id not in conversation_history:
        conversation_history[chat_id] = []
    conversation_history[chat_id].append({"role": "user", "text": user_text})
    conversation_history[chat_id].append({"role": "assistant", "content": bot_text})
    if len(conversation_history[chat_id]) > MAX_HISTORY_TURNS * 2:
        conversation_history[chat_id] = conversation_history[chat_id][-(MAX_HISTORY_TURNS * 2):]

def clear_history(chat_id: int):
    if chat_id in conversation_history:
        conversation_history[chat_id] = []

async def process_with_isaac(chat_id: int, prompt: str) -> str:
    messages = get_messages_payload(chat_id, prompt)
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": messages,
        "temperature": 0.7
    }
    
    async with httpx.AsyncClient() as http_client:
        try:
            response = await http_client.post(url, json=payload, headers=headers, timeout=30.0)
            if response.status_code == 200:
                data = response.json()
                bot_reply = data["choices"][0]["message"]["content"]
                save_to_history(chat_id, prompt, bot_reply)
                return bot_reply
            else:
                print(f"Erro Groq: {response.status_code} - {response.text}")
                return f"Erro Groq {response.status_code}. Verifique as configurações no Render."
        except Exception as e:
            print(f"Exceção ao chamar Groq: {e}")
            return "O Isaac está temporariamente indisponível. Tente novamente em instantes."

@app.get("/")
def home():
    return {"status": "Isaac está online via Groq!"}

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
                "💡 **Dica:** Para reiniciar a nossa conversa e limpar a memória, envie `/limpar`."
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
