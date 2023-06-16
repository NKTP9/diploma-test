import telebot
import model
from docx import Document as DocxDocument
from io import BytesIO

# Необходимо подставить приватный токен бота
TOKEN = "{ТОКЕН_БОТА}"

bot = telebot.TeleBot(TOKEN)


# Метод, который реагирует на команду /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    welcome_text = """Добро пожаловать в Обработчик Резюме! Пришлите ваше резюме в формате DOCX, нажав на скрепку в поле чата и выбрав файл, и система подберёт наилучшее совпадение!"""
    bot.reply_to(message, welcome_text)


# Метод, который реагирует на сообщение пользователя
@bot.message_handler(content_types=['text'])
def process_message(message):
    bot.reply_to(message, "В настоящее время система принимает только файлы в формате DOCX. Пришлите файл, нажав на скрепку в поле чата!")


# Метод, который реагирует на присланный DOCX-файл
@bot.message_handler(content_types=['document'])
def process_document(message):
    print("Сообщение принято...")
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

        print("Сообщение отправлено в модель...")
        bot_response = model.process_text(text)
        bot.reply_to(message, bot_response)
    else:
        bot.reply_to(message, "В настоящее время система принимает только файлы в формате DOCX. Пришлите файл, нажав на скрепку в поле чата!")


# Запуск бота
if __name__ == "__main__":
    print("Бот запущен...")
    bot.polling(none_stop=True)
