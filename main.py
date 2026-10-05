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

# Armazenamento em memória do histórico por utilizador { chat_id: [ {role, text} ] }
conversation_history = {}

# Limite do histórico (últimas 10 trocas de mensagens para manter o servidor leve)
MAX_HISTORY_TURNS = 10 

SYSTEM_INSTRUCTION = """
Tu és o Isaac, um assistente virtual de elite focado em apoio técnico, robótica, automação e organização profissional.
Responda sempre de forma clara, objetiva, prestativa e profissional.
Mantém o contexto das mensagens anteriores da conversa para responder de forma contínua.
"""

MODELS_TO_TRY = ["gemini-3.8-flash", "gemini-flash-latest"]

# =====================================================================
# GESTÃO DE MEMÓRIA E HISTÓRICO
# =====================================================================
def format_history_for_gemini(chat_id: int, new_prompt: str):
    """Constrói a estrutura de mensagens do histórico para a API do Gemini."""
    history = conversation_history.get(chat_id, [])
    contents = []
    
    for item in history:
        contents.append(
            types.Content(
                role=item["role"],
                parts=[types.Part.from_text(text=item["text"])]
            )
        )
    
    # Adiciona a mensagem atual
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=new_prompt)]
        )
    )
    return contents

def save_to_history(chat_id: int, user_text: str, bot_text: str):
    """Guarda a interação no histórico do chat_id."""
    if chat_id not in conversation_history:
        conversation_history[chat_id] = []
    
    conversation_history[chat_id].append({"role": "user", "text": user_text})
    conversation_history[chat_id].append({"role": "model", "text": bot_text})
    
    # Limita o tamanho da memória
    if len(conversation_history[chat_id]) > MAX_HISTORY_TURNS * 2:
        conversation_history[chat_id] = conversation_history[chat_id][-(MAX_HISTORY_TURNS * 2):]

def clear_history(chat_id: int):
    """Limpa a memória da conversa do utilizador."""
    if chat_id in conversation_history:
        conversation_history[chat_id] = []


# =====================================================================
# NÚCLEO CENTRAL DE IA (Com Memória)
# =====================================================================
def process_with_isaac(chat_id: int, prompt: str) -> str:
    contents = format_history_for_gemini(chat_id, prompt)
    
    for model_name in MODELS_TO_TRY:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION
                    )
                )
                if response.text:
                    save_to_history(chat_id, prompt, response.text)
                    return response.text
            except Exception as e:
                if "503" in str(e):
                    time.sleep(1)
                    continue
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
        user_message = data["message"]["text"].strip()
        
        # --- COMANDOS DO TELEGRAM ---
        if user_message == "/start":
            clear_history(chat_id)
            reply = (
                "👋 **Olá! Eu sou o Isaac.**\n\n"
                "Estou pronto para ajudar em robótica, automação, programação e organização de projetos.\n\n"
                "💡 **Dica:** Agora eu lembro-me do contexto das nossas conversas! "
                "Para reiniciar o assunto e apagar a minha memória, envie `/limpar`."
            )
        elif user_message in ["/limpar", "/reset"]:
            clear_history(chat_id)
            reply = "🧹 **Memória limpa com sucesso!** Podemos começar um novo assunto do zero."
        else:
            # Processa mensagem normal com histórico de conversa
            reply = process_with_isaac(chat_id, user_message)
        
        # Envia a resposta de volta ao Telegram
        if TELEGRAM_BOT_TOKEN:
            telegram_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            async with httpx.AsyncClient() as http_client:
                await http_client.post(telegram_url, json={
                    "chat_id": chat_id,
                    "text": reply,
                    "parse_mode": "Markdown"
                })
                
    return {"status": "ok"}
