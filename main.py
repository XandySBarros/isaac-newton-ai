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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

MODELS_TO_TRY = [
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-pro"
]

SYSTEM_INSTRUCTION = (
    "Você é o JARVIS, um assistente pessoal inteligente que responde "
    "comandos e perguntas pelo Telegram. Responda sempre em português, "
    "de forma clara, objetiva e amigável. Ajude também com projetos de "
    "eletrônica e programação, como ESP32."
)

MAX_HISTORY_MESSAGES = 20
chat_histories: dict[int, list[dict]] = {}
processed_updates: set[int] = set()


# ---------------------------------------------------------------------------
# Auxiliares do Histórico
# ---------------------------------------------------------------------------
def get_gemini_contents(chat_id: int, user_parts: list[dict]) -> list[dict]:
    history = chat_histories.get(chat_id, [])
    return history + [{"role": "user", "parts": user_parts}]


def save_to_history(chat_id: int, user_text: str, reply_text: str) -> None:
    history = chat_histories.setdefault(chat_id, [])
    history.append({"role": "user", "parts": [{"text": user_text}]})
    history.append({"role": "model", "parts": [{"text": reply_text}]})
    if len(history) > MAX_HISTORY_MESSAGES:
        del history[: len(history) - MAX_HISTORY_MESSAGES]


# ---------------------------------------------------------------------------
# Processamento com Gemini (Texto, Foto ou Áudio)
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

    async with httpx.AsyncClient(timeout=60.0) as client:
        for model in MODELS_TO_TRY:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

            for _attempt in range(2):
                try:
                    r = await client.post(url, json=payload, headers=headers)
                except Exception as e:
                    print(f"JARVIS: {model} erro de rede: {repr(e)}")
                    last_status = "timeout"
                    await asyncio.sleep(1.0)
                    continue

                if r.status_code == 200:
                    try:
                        parts = r.json()["candidates"][0]["content"]["parts"]
                        reply = "".join(p.get("text", "") for p in parts).strip()
                    except Exception:
                        break

                    if not reply:
                        break

                    save_to_history(chat_id, history_prompt, reply)
                    return reply

                last_status = r.status_code

                if r.status_code in (429, 500, 503):
                    await asyncio.sleep(1.0)
                    continue

                break

    return (
        f"⚠️ Todos os modelos falharam (último erro: {last_status}). "
        "Tente de novo em instantes."
    )


# ---------------------------------------------------------------------------
# Telegram: Baixar Arquivos (Áudio ou Foto)
# ---------------------------------------------------------------------------
async def download_telegram_file(file_id: str) -> bytes | None:
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


# ---------------------------------------------------------------------------
# Telegram: Enviar Mensagens
# ---------------------------------------------------------------------------
async def send_telegram_message(chat_id: int, text: str) -> None:
    if not TELEGRAM_BOT_TOKEN:
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    chunks = [text[i : i + 4000] for i in range(0, len(text), 4000)] or [""]

    async with httpx.AsyncClient(timeout=30.0) as client:
        for chunk in chunks:
            try:
                await client.post(url, json={"chat_id": chat_id, "text": chunk})
            except Exception as e:
                print(f"TELEGRAM: falha ao enviar texto: {repr(e)}")


async def send_telegram_voice(chat_id: int, text_to_speak: str) -> None:
    if not TELEGRAM_BOT_TOKEN:
        return

    try:
        tts = gTTS(text=text_to_speak, lang="pt", slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)

        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVoice"
        files = {"voice": ("voice.ogg", audio_fp, "audio/ogg")}
        data = {"chat_id": chat_id}

        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(url, data=data, files=files)
    except Exception as e:
        print(f"TELEGRAM: falha ao enviar voz: {repr(e)}")


# ---------------------------------------------------------------------------
# Webhook Handler
# ---------------------------------------------------------------------------
async def handle_update(update: dict) -> None:
    try:
        message = update.get("message") or update.get("edited_message")
        if not message:
            return

        chat_id = message["chat"]["id"]
        user_text = message.get("text")
        caption_text = message.get("caption", "")
        voice_info = message.get("voice") or message.get("audio")
        photo_info = message.get("photo")

        # 1. Comandos
        if user_text:
            cmd = user_text.strip()
            if cmd == "/start":
                await send_telegram_message(
                    chat_id, "Olá! Eu sou o JARVIS. Pode me mandar texto, fotos ou mensagens de áudio! 🤖📸🎙️"
                )
                return
            if cmd in ["/limpar", "/reset"]:
                chat_histories.pop(chat_id, None)
                await send_telegram_message(chat_id, "Memória da conversa apagada. ✅")
                return

        # 2. Processar Fotos
        if photo_info:
            # Seleciona a foto com melhor resolução (última da lista)
            best_photo = photo_info[-1]
            file_id = best_photo.get("file_id")
            image_bytes = await download_telegram_file(file_id)

            if not image_bytes:
                await send_telegram_message(chat_id, "⚠️ Não consegui baixar a foto enviada.")
                return

            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            prompt_text = caption_text if caption_text else "Analise esta imagem e me ajude com o que estiver nela."

            user_parts = [
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64_image,
                    }
                },
                {"text": prompt_text}
            ]

            reply = await process_with_jarvis(
                chat_id, 
                user_parts, 
                history_prompt=f"[Foto enviada] {prompt_text}"
            )
            await send_telegram_message(chat_id, reply)
            return

        # 3. Processar Entrada de Áudio (Voz)
        if voice_info:
            file_id = voice_info.get("file_id")
            audio_bytes = await download_telegram_file(file_id)
            if not audio_bytes:
                await send_telegram_message(chat_id, "⚠️ Não consegui baixar o áudio enviado.")
                return

            base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
            user_parts = [
                {
                    "inline_data": {
                        "mime_type": "audio/ogg",
                        "data": base64_audio,
                    }
                }
            ]

            reply = await process_with_jarvis(
                chat_id, 
                user_parts, 
                history_prompt="[Áudio do Usuário]"
            )
            await send_telegram_message(chat_id, reply)
            await send_telegram_voice(chat_id, reply)
            return

        # 4. Processar Entrada de Texto
        if user_text:
            user_parts = [{"text": user_text}]
            reply = await process_with_jarvis(chat_id, user_parts, history_prompt=user_text)
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


@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"status": "JARVIS online"}
