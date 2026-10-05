import os
import asyncio
import logging
from fastapi import FastAPI, Request
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from google import genai
from google.genai import types

# Configuração de Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Inicialização do Cliente Gemini SDK Oficial
client = genai.Client(api_key=GEMINI_API_KEY)

# Instância da App FastAPI
app = FastAPI()

# Histórico em memória: { chat_id: [ {"role": "user"/"model", "content": "..."}, ... ] }
chat_history = {}

SYSTEM_INSTRUCTION = (
    "Tu és o Isaac, um assistente virtual inteligente, prestativo, educado e bem-humorado. "
    "Responde sempre em português de Portugal. "
    "Trata o utilizador com proximidade e cordialidade. "
    "Quando fores questionado sobre a tua identidade ou criador, responde de forma simpática "
    "que foste desenvolvido pelo Alexandre para ajudar no dia a dia."
)

def build_gemini_contents(user_id: int, user_message: str) -> list[types.Content]:
    """
    Converte o histórico guardado em memória para a estrutura oficial
    de objetos types.Content e types.Part do SDK google-genai.
    """
    history = chat_history.get(user_id, [])
    contents = []

    # Adiciona o histórico anterior convertido para objetos types.Content
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])]
            )
        )

    # Adiciona a mensagem atual do utilizador
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)]
        )
    )
    return contents

async def process_with_isaac(user_id: int, user_message: str) -> str:
    contents = build_gemini_contents(user_id, user_message)
    
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.7,
    )

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",  # Pode utilizar "gemini-2.5-flash" ou "gemini-2.0-flash"
                contents=contents,
                config=config,
            )
            
            bot_reply = response.text
            
            # Atualiza o histórico na memória após resposta bem-sucedida
            if user_id not in chat_history:
                chat_history[user_id] = []
            
            chat_history[user_id].append({"role": "user", "content": user_message})
            chat_history[user_id].append({"role": "model", "content": bot_reply})
            
            # Mantém apenas as últimas 20 mensagens (10 turnos de conversa)
            if len(chat_history[user_id]) > 20:
                chat_history[user_id] = chat_history[user_id][-20:]
                
            return bot_reply

        except Exception as e:
            # Imprime o erro detalhado nos Logs do Render
            logger.error(f"--> [ERRO GEMINI - Tentativa {attempt + 1}]: {e}", exc_info=True)
            if attempt < 2:
                await asyncio.sleep(2)

    return "O Isaac encontrou um problema técnico ao processar a mensagem. Por favor, tente novamente em instantes."
