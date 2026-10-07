import asyncio
import base64
import io
import os

import httpx
from fastapi import BackgroundTasks, FastAPI, Request
from gtts import gTTS

app = FastAPI()

# ---------------------------------------------------------------------------
# Configuração (variáveis de ambiente do Render)
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# Plano B: Groq (usado só quando todos os modelos do Gemini falham)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODELS = [
    m.strip()
    for m in os.getenv("GROQ_MODELS", "llama-3.3-70b-versatile,llama-3.1-8b-instant").split(",")
    if m.strip()
]

# Ordem de modelos configurada por você
MODELS_TO_TRY = [
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-flash-latest",
]

# Personalidade / regras do bot.
SYSTEM_INSTRUCTION = (
    "Você é o JARVIS, um assistente pessoal inteligente que responde "
    "comandos e perguntas pelo Telegram. Responda sempre em português, "
    "de forma clara, objetiva e amigável. Ajude também com projetos de "
    "eletrônica e programação, como ESP32."
)

# ---------------------------------------------------------------------------
# Memória de conversa (em RAM: zera quando o Render reinicia)
# ---------------------------------------------------------------------------
MAX_HISTORY_MESSAGES = 20
chat_histories: dict[int, list[dict]] = {}

# Evita processar a mesma atualização do Telegram duas vezes
processed_updates: set[int] = set()


def get_gemini_contents(chat_id: int, user_parts: list[dict]) -> list[dict]:
    """Histórico da conversa + a nova mensagem do usuário (texto/foto/áudio)."""
    history = chat_histories.get(chat_id, [])
    return history + [{"role": "user", "parts": user_parts}]


def save_to_history(chat_id: int, prompt_text: str, reply_text: str) -> None:
    history = chat_histories.setdefault(chat_id, [])
    history.append({"role": "user", "parts": [{"text": prompt_text}]})
    history.append({"role": "model", "parts": [{"text": reply_text}]})
    if len(history) > MAX_HISTORY_MESSAGES:
        del history[: len(history) - MAX_HISTORY_MESSAGES]


# ---------------------------------------------------------------------------
# Gemini (com retry e fallback entre modelos)
# ---------------------------------------------------------------------------
async def process_with_jarvis(chat_id: int, user_parts: list[dict], history_prompt: str) -> str:
    if not GEMINI_API_KEY:
        return "⚠️ A GEMINI_API_KEY não está configurada no Render."

    contents = get_gemini_contents(chat_id, user_parts)
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }
    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
    }

    last_status = None

    async with httpx.AsyncClient(timeout=30.0) as client:
        for model in MODELS_TO_TRY:
            url = (
                "https://generativelanguage.googleapis.com/"
                f"v1beta/models/{model}:generateContent"
            )

            for _attempt in range(2):  # 2 tentativas por modelo
                try:
                    r = await client.post(url, json=payload, headers=headers)
                except Exception as e:
                    print(f"JARVIS: {model} erro de rede: {repr(e)}")
                    last_status = "timeout"
                    await asyncio.sleep(1.5)
                    continue

                print(f"JARVIS: {model} -> HTTP {r.status_code}")

                if r.status_code == 200:
                    try:
                        parts = r.json()["candidates"][0]["content"]["parts"]
                        reply = "".join(p.get("text", "") for p in parts).strip()
                    except Exception:
                        print(f"JARVIS: resposta inesperada: {r.text[:1000]}")
                        break  # tenta o próximo modelo

                    if not reply:
                        print(f"JARVIS: resposta vazia: {r.text[:1000]}")
                        break

                    save_to_history(chat_id, history_prompt, reply)
                    return reply

                last_status = r.status_code
                print(f"JARVIS: {r.text[:500]}")

                if r.status_code in (429, 500, 503):
                    await asyncio.sleep(1.5)  # congestionado: tenta de novo
                    continue

                break  # 400/401/403/404: vai para o próximo modelo

    # Gemini falhou em todos os modelos: tenta o Groq
    print("JARVIS: Gemini indisponível, tentando Groq...")
    groq_reply = await process_with_groq(chat_id, history_prompt)
    if groq_reply:
        return groq_reply

    return (
        f"⚠️ Todos os modelos falharam (último erro Gemini: {last_status}). "
        "Tente de novo em instantes."
    )


async def process_with_groq(chat_id: int, prompt: str) -> str | None:
    """Plano B. Retorna None se o Groq também falhar."""
    if not GROQ_API_KEY:
        print("GROQ: GROQ_API_KEY não configurada, pulando fallback.")
        return None

    # Converte o histórico do formato Gemini para o formato OpenAI/Groq
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    for item in chat_histories.get(chat_id, []):
        role = "assistant" if item["role"] == "model" else "user"
        text = "".join(p.get("text", "") for p in item["parts"])
        messages.append({"role": role, "content": text})
    messages.append({"role": "user", "content": prompt})

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}",
    }
    url = "https://api.groq.com/openai/v1/chat/completions"

    async with httpx.AsyncClient(timeout=30.0) as client:
        for model in GROQ_MODELS:
            try:
                r = await client.post(
                    url, json={"model": model, "messages": messages}, headers=headers
                )
            except Exception as e:
                print(f"GROQ: {model} erro de rede: {repr(e)}")
                continue

            print(f"GROQ: {model} -> HTTP {r.status_code}")

            if r.status_code == 200:
                try:
                    reply = r.json()["choices"][0]["message"]["content"].strip()
                except Exception:
                    print(f"GROQ: resposta inesperada: {r.text[:500]}")
                    continue
                if reply:
                    save_to_history(chat_id, prompt, reply)
                    return reply
            else:
                print(f"GROQ: {r.text[:500]}")

    return None


# ---------------------------------------------------------------------------
# Telegram: Auxiliares para Baixar Arquivos e Enviar Respostas
# ---------------------------------------------------------------------------
async def download_telegram_file(file_id: str) -> bytes | None:
    """Baixa fotos ou áudios enviados pelos usuários no Telegram."""
    if not TELEGRAM_BOT_TOKEN:
        return None
    async with httpx.AsyncClient(timeout=30.0) as client:
        get_file_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile?file_id={file_id}"
        r = await client.get(get_file_url)
        if r.status_code != 200:
            return None
        file_path = r.json().get("result", {}).get("file_path")
        if not file_path:
            return None

        download_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
        r_file = await client.get(download_url)
        if r_file.status_code == 200:
            return r_file.content
    return None


async def send_telegram_message(chat_id: int, text: str) -> None:
    if not TELEGRAM_BOT_TOKEN:
        print("ERRO: TELEGRAM_BOT_TOKEN não configurado no Render.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    chunks = [text[i : i + 4000] for i in range(0, len(text), 4000)] or [""]

    async with httpx.AsyncClient(timeout=30.0) as client:
        for chunk in chunks:
            try:
                r = await client.post(url, json={"chat_id": chat_id, "text": chunk})
                if r.status_code != 200:
                    print(f"TELEGRAM: erro {r.status_code}: {r.text[:500]}")
            except Exception as e:
                print(f"TELEGRAM: falha ao enviar texto: {repr(e)}")


import edge_tts  # Remova a linha "from gtts import gTTS" e coloque esta no topo

# ... (restante do código) ...

async def send_telegram_voice(chat_id: int, text_to_speak: str) -> None:
    """Converte o texto da resposta em áudio com voz neural realista do Edge TTS."""
    if not TELEGRAM_BOT_TOKEN:
        return

    try:
        # Vozes em PT-BR disponíveis:
        # "pt-BR-AntonioNeural" (Masculina - perfeita para o JARVIS)
        # "pt-BR-FranciscaNeural" (Feminina)
        VOICE = "pt-BR-AntonioNeural"

        communicate = edge_tts.Communicate(text_to_speak, VOICE)
        
        # Gera o áudio na memória
        audio_bytes = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes.extend(chunk["data"])

        audio_fp = io.BytesIO(audio_bytes)
        audio_fp.name = "voice.ogg"

        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVoice"
        files = {"voice": ("voice.ogg", audio_fp, "audio/ogg")}
        data = {"chat_id": chat_id}

        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(url, data=data, files=files)
            
    except Exception as e:
        print(f"TELEGRAM: falha ao enviar voz: {repr(e)}")


# ---------------------------------------------------------------------------
# Handlers de Atualizações do Telegram
# ---------------------------------------------------------------------------
async def handle_update(update: dict) -> None:
    """Processa uma mensagem (texto, foto ou áudio) em segundo plano."""
    try:
        message = update.get("message") or update.get("edited_message")
        if not message:
            return

        chat_id = message["chat"]["id"]
        user_message = message.get("text")
        caption_text = message.get("caption", "")
        voice_info = message.get("voice") or message.get("audio")
        photo_info = message.get("photo")

        # 1. Comandos de Texto
        if user_message:
            cmd = user_message.strip()
            if cmd == "/start":
                await send_telegram_message(
                    chat_id, "Olá! Eu sou o JARVIS. Pode me mandar mensagens de texto, fotos ou áudios! 🤖📸🎙️"
                )
                return

            if cmd == "/limpar":
                chat_histories.pop(chat_id, None)
                await send_telegram_message(chat_id, "Memória da conversa apagada. ✅")
                return

        # 2. Processar Fotos
        if photo_info:
            best_photo = photo_info[-1]  # Foto em maior resolução
            file_id = best_photo.get("file_id")
            print(f"JARVIS: Foto recebida de {chat_id}")
            image_bytes = await download_telegram_file(file_id)

            if not image_bytes:
                await send_telegram_message(chat_id, "⚠️ Não consegui baixar a foto enviada.")
                return

            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            prompt_text = caption_text.strip() if caption_text else "Analise esta imagem e me ajude com o que estiver nela."

            user_parts = [
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64_image,
                    }
                },
                {"text": prompt_text},
            ]

            history_prompt = f"[Foto enviada] {prompt_text}"
            reply = await process_with_jarvis(chat_id, user_parts, history_prompt)
            await send_telegram_message(chat_id, reply)
            return

        # 3. Processar Mensagens de Voz / Áudio
        if voice_info:
            file_id = voice_info.get("file_id")
            mime_type = voice_info.get("mime_type", "audio/ogg")
            print(f"JARVIS: Áudio recebido de {chat_id}")
            audio_bytes = await download_telegram_file(file_id)

            if not audio_bytes:
                await send_telegram_message(chat_id, "⚠️ Não consegui baixar o áudio enviado.")
                return

            base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
            user_parts = [
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64_audio,
                    }
                },
                {"text": "Ouça este áudio e responda ao que foi solicitado."},
            ]

            history_prompt = "[Áudio do usuário]"
            reply = await process_with_jarvis(chat_id, user_parts, history_prompt)
            await send_telegram_message(chat_id, reply)
            await send_telegram_voice(chat_id, reply)
            return

        # 4. Processar Texto Normal
        if user_message:
            print(f"JARVIS: mensagem recebida de {chat_id}: {user_message[:200]}")
            user_parts = [{"text": user_message}]
            reply = await process_with_jarvis(chat_id, user_parts, user_message)
            await send_telegram_message(chat_id, reply)
            return

    except Exception as e:
        print(f"ERRO em handle_update: {repr(e)}")


@app.post("/telegram/webhook")
async def telegram_webhook(request: Request, background_tasks: BackgroundTasks):
    update = await request.json()
    update_id = update.get("update_id")

    if update_id is not None:
        if update_id in processed_updates:
            return {"ok": True}
        processed_updates.add(update_id)
        if len(processed_updates) > 1000:
            processed_updates.clear()

    background_tasks.add_task(handle_update, update)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Health check (Render / UptimeRobot)
# ---------------------------------------------------------------------------
@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"status": "JARVIS online"}
