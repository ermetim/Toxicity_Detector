import asyncio
import logging
import os
import random
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message, ReplyKeyboardRemove
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

from text_predictor import predict_text_toxicity

load_dotenv()
BOT_TOKEN = os.getenv('BOT_TOKEN')

logging.basicConfig(level=logging.INFO)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())

def generate_keyboard(button_names, n=2):
    if type(button_names) is dict:
        button_names = list(button_names.keys())
    keyboard_buttons = []
    for i in range(0, len(button_names), n):
        lines = []
        for button in button_names[i:i+n]:
            lines.append(KeyboardButton(text=button))
        keyboard_buttons.append(lines)
    keyboard = ReplyKeyboardMarkup(
        keyboard=keyboard_buttons,
        resize_keyboard=True,
        input_field_placeholder="Выберите ответ",
        one_time_keyboard=True
    )
    return keyboard

class States(StatesGroup):
    yes_no = State()
    text_input = State()

@dp.message(Command(commands=["start"]))
async def command_handler(message: Message, state: FSMContext) -> None:
    await message.answer(
        text="Привет! Я бот для детекции токсичных сообщений. Хотите проверить текст?",
        reply_markup=generate_keyboard(["Да", "Нет"])
    )
    await state.set_state(States.yes_no)

@dp.message(F.text.lower() == "да", States.yes_no)
async def continue_yes(message: Message, state: FSMContext) -> None:
    await message.reply(
        text="Отправьте сообщение, которое нужно проверить.",
        reply_markup=ReplyKeyboardRemove()
    )
    await state.set_state(States.text_input)

@dp.message(F.text.lower() == "нет", States.yes_no)
async def continue_no(message: Message, state: FSMContext) -> None:
    await message.reply(
        text="Жаль! До встречи в следующий раз.",
        reply_markup=ReplyKeyboardRemove()
    )
    await state.clear()

@dp.message(States.yes_no)
async def continue_unexpected(message: Message) -> None:
    await message.reply(
        text="Введите 'Да' для продолжения или 'Нет' для выхода",
        reply_markup=generate_keyboard(["Да", "Нет"])
    )

@dp.message(States.text_input)
async def handle_text(message: Message, state: FSMContext):
    text = message.text.strip()
    pred = predict_text_toxicity(text)

    if pred == 1:
        with open("responses/toxic_responses.txt", "r", encoding="utf-8") as f:
            responses = f.readlines()
    else:
        with open("responses/non_toxic_responses.txt", "r", encoding="utf-8") as f:
            responses = f.readlines()

    answer = random.choice(responses).strip()
    await message.answer(answer)
    await message.answer(
        text="Хотите проверить еще одно сообщение?",
        reply_markup=generate_keyboard(["Да", "Нет"])
    )
    await state.set_state(States.yes_no)

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
