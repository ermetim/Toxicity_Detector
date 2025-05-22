Для запуска бота необходимо создать в корне проекта файл .env и внутри прописать

    BOT_TOKEN = 'YOUR TOKEN'    

Токен должен быть такого вида: 

    '123456789:AABBCCDDEEFFaabbccddeeff-1234567890'

Для запуска через терминал войти в папку tg_bot выполнить команду

    python bot.py

Для создания и запуска докер образа

    docker build --no-cache -t image_name .
    docker run --env BOT_TOKEN='YOUR TOKEN' image_name
