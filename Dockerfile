FROM python:3.10-slim

# setup working directory
WORKDIR app
# copy src directory to the container
COPY ./src /app/src
# copy models directory to the container
COPY ./models /app/models
COPY ./tg_bot /app/tg_bot
COPY ./config.py /app/

ENV PYTHONPATH "${PYTHONPATH}:/app"

RUN apt-get update && apt-get install -y libglib2.0-0 libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install -r /app/tg_bot/bot_requirements.txt

# Run bot when docker run
CMD ["python3", "/app/tg_bot/bot.py"]
