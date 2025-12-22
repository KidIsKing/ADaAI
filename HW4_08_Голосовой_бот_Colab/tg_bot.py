import requests
import uuid
import io
import os
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
import numpy as np
import scipy.io
from langchain_gigachat.chat_models import GigaChat
from langchain_core.messages import HumanMessage

# Токены
AUTH_SPEECH = "MDE5YjQxYTMtNTQ0NS03MjNkLTgyOTEtY2EwZTBlNzM5M2VjOjg2NDVjNzcwLWE2ZTItNDJkNy1hY2MyLTdmYWI0ZmY5MmZhNQ=="
AUTH_GIGA = "MDE5YWNiMjYtYzQ4YS03Njc3LWE0MTMtZTM1OTQxZjdjMDFlOjA1NTUyZTk4LTVlZmMtNDY4Ni04ZTFkLWU1MDYwMmEwOWQ1Ng=="
TELEGRAM_TOKEN = "8585747582:AAF4JnswBHlkCvDBRj7QYGSzMfSZX7Df9to"


# Получение токена SaluteSpeech
def get_speech_token(auth_token, scope="SALUTE_SPEECH_PERS"):
    rq_uid = str(uuid.uuid4())
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "RqUID": rq_uid,
        "Authorization": f"Basic {auth_token}",
    }
    payload = {"scope": scope}

    response = requests.post(url, headers=headers, data=payload, verify=False)
    if response.status_code == 200:
        return response.json()["access_token"]
    return None


# Инициализация GigaChat
def init_gigachat():
    llm = GigaChat(credentials=AUTH_GIGA, verify_ssl_certs=False, timeout=30)
    return llm


# Распознавание речи - ОТПРАВКА OGG НАПРЯМУЮ
def speech_to_text(audio_bytes, token, content_type="audio/ogg;codecs=opus"):
    """Распознавание речи через SaluteSpeech API"""
    url = "https://smartspeech.sber.ru/rest/v1/speech:recognize"

    headers = {"Authorization": f"Bearer {token}", "Content-Type": content_type}

    print(f"\n=== DEBUG Speech2Text ===")
    print(f"Content-Type: {headers['Content-Type']}")
    print(f"Размер аудио: {len(audio_bytes)} байт")

    try:
        response = requests.post(
            url, headers=headers, data=audio_bytes, verify=False, timeout=30
        )

        print(f"=== RESPONSE DEBUG ===")
        print(f"Статус ответа: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"Успешный ответ: {result}")
            return result["result"][0]
        else:
            print(f"ОШИБКА {response.status_code}: {response.text[:200]}")
            return None

    except Exception as e:
        print(f"Исключение: {type(e).__name__}: {e}")
        return None


# Синтез речи
def text_to_speech(text, token):
    url = "https://smartspeech.sber.ru/rest/v1/text:synthesize"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/text"}
    params = {"format": "wav16", "voice": "Bys_24000"}

    try:
        response = requests.post(
            url,
            headers=headers,
            params=params,
            data=text.encode("utf-8"),
            verify=False,
            timeout=30,
        )
        if response.status_code == 200:
            return response.content
        else:
            print(f"Text2Speech error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Text2Speech exception: {e}")
        return None


# Обработчик команды /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я голосовой бот.\n"
        "Отправь мне голосовое сообщение, и я постараюсь его распознать и ответить.\n"
        "Также могу отвечать на текстовые сообщения.\n\n"
        "Использую SaluteSpeech для распознавания/синтеза речи и GigaChat для ответов."
    )


# Обработчик голосовых сообщений - УПРОЩЕННЫЙ И РАБОЧИЙ
async def voice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Упрощенный обработчик голосовых сообщений"""
    try:
        await update.message.reply_text("🎤 Получил ваше голосовое сообщение...")

        # 1. Скачиваем голосовое сообщение
        voice_file = await update.message.voice.get_file()
        voice_bytes_io = io.BytesIO()
        await voice_file.download_to_memory(out=voice_bytes_io)
        ogg_bytes = voice_bytes_io.getvalue()

        if len(ogg_bytes) == 0:
            await update.message.reply_text("❌ Не удалось загрузить аудио")
            return

        await update.message.reply_text("🔑 Получаю доступ к API...")

        # 2. Получаем токен
        speech_token = get_speech_token(AUTH_SPEECH)
        if not speech_token:
            await update.message.reply_text("❌ Не удалось получить токен API")
            return

        await update.message.reply_text("👂 Распознаю речь...")

        # 3. Пробуем отправить OGG напрямую (самый простой способ)
        # Telegram отправляет OGG с кодеком Opus, а API SaluteSpeech его принимает
        text = speech_to_text(
            audio_bytes=ogg_bytes,
            token=speech_token,
            content_type="audio/ogg;codecs=opus",
        )

        # 4. Если OGG не работает, пробуем конвертировать через pydub
        if not text:
            await update.message.reply_text("⚠️ Пробую альтернативный метод...")
            try:
                from pydub import AudioSegment

                # Загружаем OGG
                audio = AudioSegment.from_file(io.BytesIO(ogg_bytes), format="ogg")
                # Конвертируем в RAW PCM (сырые данные)
                audio = audio.set_frame_rate(16000).set_channels(1)
                raw_pcm_bytes = audio.raw_data

                # Отправляем как PCM
                text = speech_to_text(
                    audio_bytes=raw_pcm_bytes,
                    token=speech_token,
                    content_type="audio/x-pcm;bit=16;rate=16000",
                )

            except ImportError:
                await update.message.reply_text(
                    "❌ Установите pydub: pip install pydub"
                )
                return

        # 5. Обрабатываем результат
        if text:
            await update.message.reply_text(
                f"🗣️ Вы сказали: *{text}*", parse_mode="Markdown"
            )

            # Получаем ответ от GigaChat
            await update.message.reply_text("🤔 Думаю над ответом...")
            llm = init_gigachat()
            messages = [HumanMessage(content=text)]
            response = llm.invoke(messages)
            ai_response = response.content

            # Отправляем текстовый ответ
            await update.message.reply_text(f"💬 {ai_response}")

            # Пробуем синтезировать голос
            if len(ai_response) > 100:
                short_response = ai_response[:100] + "..."
            else:
                short_response = ai_response

            speech_bytes = text_to_speech(short_response, speech_token)

            if speech_bytes:
                await update.message.reply_voice(
                    voice=io.BytesIO(speech_bytes), caption="🎤 Ответ голосом"
                )
            else:
                await update.message.reply_text("⚠️ Текстовый ответ отправлен")
        else:
            await update.message.reply_text(
                "❌ Не удалось распознать речь.\n"
                "Попробуйте:\n"
                "1. Отправить текстовое сообщение\n"
                "2. Проверить подключение к интернету"
            )

    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {str(e)[:200]}")
        print(f"Voice handler error: {e}")


# Обработчик текстовых сообщений
async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        text = update.message.text
        if not text or len(text.strip()) == 0:
            return

        await update.message.reply_text("🤔 Думаю над ответом...")

        llm = init_gigachat()
        messages = [HumanMessage(content=text)]
        response = llm.invoke(messages)
        ai_response = response.content

        await update.message.reply_text(f"💬 {ai_response}")

    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {str(e)[:100]}...")


# Основная функция
def main():
    # Создаем приложение
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Добавляем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.VOICE, voice_handler))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler)
    )

    # Запускаем бота
    print("Бот запущен...")
    print("Отправьте /start в Telegram")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
