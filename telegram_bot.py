import telebot
import model
from docx import Document as DocxDocument
from io import BytesIO

TOKEN = "6289925879:AAHUUPHLxu8WV36nUALlvUREvHnf3597kgE"  # Замените на ваш токен

bot = telebot.TeleBot(TOKEN)


@bot.message_handler(commands=['start'])
def send_welcome(message):
    welcome_text = """Добро пожаловать в Обработчик Резюме! Пришлите мне ваше резюме в формате DOCX, и я подскажу, на какую профессию вы лучше всего подойдёте!"""
    bot.reply_to(message, welcome_text)


@bot.message_handler(content_types=['text'])
def process_message(message):
    bot.reply_to(message, "Я пока понимаю и принимаю резюме в формате DOCX. Пришлите мне его скорее!")


@bot.message_handler(content_types=['document'])
def process_document(message):
    bot.reply_to(message, "Мои мыслительные процессы начали обрабатывать ваше резюме... Это может занять время, ожидайте!")
    if message.document.mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        with BytesIO(downloaded_file) as docx_file:
            docx_file.seek(0)  # Устанавливаем указатель на начало файла
            document = DocxDocument(docx_file)
            full_text = []

            for paragraph in document.paragraphs:
                full_text.append(paragraph.text)

            text = "\n".join(full_text)

        bot_response = model.process_text(text)
        bot.reply_to(message, bot_response)
    else:
        bot.reply_to(message, "Я пока понимаю и принимаю ваше резюме в формате DOCX. Пришлите мне его скорее!")


if __name__ == "__main__":
    print("Бот запущен...")
    bot.polling(none_stop=True)
