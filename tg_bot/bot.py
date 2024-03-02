import asyncio
import logging
import os
import pickle
import cv2
import numpy as np
from dotenv import load_dotenv, dotenv_values

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.filters.callback_data import CallbackData
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message, ReplyKeyboardRemove
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from src.ml_functions import ImageProcessing

MODEL_NAMES = ['lgbm_model_hog_best.pkl',
               'lgbm_model_pic_best.pkl',
               'rfc_model_hog_best.pkl',
               'rfc_model_pic_best.pkl',
               ]
load_dotenv()
config = dotenv_values("../.env")
BOT_TOKEN = config['BOT_TOKEN']

# fake token
# BOT_TOKEN = '123456789:AABBCCDDEEFFaabbccddeeff-1234567890'

# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


class TestCallbackData(CallbackData, prefix="test_callback_data"):
    id: int
    name: str


dp = Dispatcher(storage=MemoryStorage())


def generate_keyboard(button_names, n=2):
    """
    Создание списка кнопок по строкам и возвращение объекта keyboard
    :param button_names: список кнопок
    :param n: Количество кнопок в ряду
    :return: список кнопок типа KeyboardButton(text="button_name")
    """
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
        input_field_placeholder="'Выберите ответ'",
        one_time_keyboard=True
    )
    return keyboard


class States(StatesGroup):
    yes_no = State()
    type_detection = State()
    ml_foto = State()


@dp.message(Command(commands=["start"]))
async def command_handler(message: Message, state: FSMContext) -> None:
    await message.answer(
        text="Это бот по детекции Дипфейков на фотографии! \
             \nХотите попробовать?",
        reply_markup=generate_keyboard(["Да", "Нет"])
    )
    await state.set_state(States.yes_no)


@dp.message(F.text.lower() == "да", States.yes_no)
async def continue_yes(message: Message,
                       state: FSMContext) -> None:
    await message.reply(
        text="Отличный выбор! \nВыберите метод или нажмите Отмена для выхода",
        reply_markup=generate_keyboard(["ML", "DL", "Отмена"], 3)
    )
    await state.set_state(States.type_detection)


@dp.message(F.text.lower() == "нет", States.yes_no)
async def continue_no(message: Message,
                      state: FSMContext) -> None:
    await message.reply(
        text="Жаль! \nДо встречи в следующий раз",
        reply_markup=ReplyKeyboardRemove()
    )
    await state.clear()


@dp.message(States.yes_no)
async def continue_unexpected(message: Message) -> None:
    await message.reply(
        text="Введите 'Да' для продолжения или 'Нет' для выхода",
        reply_markup=generate_keyboard(["Да", "Нет"])
    )


@dp.message(F.text.lower() == "ml", States.type_detection)
async def ml_method(message: Message,
                    state: FSMContext) -> None:
    await message.reply(
        text="Machine Learning - отличный выбор!\nДобавьте фотографию",
        reply_markup=ReplyKeyboardRemove()
    )
    await state.set_state(States.ml_foto)


@dp.message(F.text.lower() == "dl", States.type_detection)
async def dl_method(message: Message) -> None:
    await message.reply(
        text="Deep Learning. \
            \nК сожалению, данный метод пока еще в разработке.\
            \nВведите 'ML' для продолжения или 'Отмена' для выхода",
        reply_markup=generate_keyboard(["ML", "DL", "Отмена"], 3)
    )


@dp.message(F.text.lower() == "отмена", States.type_detection)
async def quit_method(message: Message, state: FSMContext) -> None:
    await message.reply(
        text="Жаль! \nДо встречи в следующий раз",
        reply_markup=ReplyKeyboardRemove()
    )
    await state.clear()


@dp.message(States.type_detection)
async def unexpected_method(message: Message) -> None:
    await message.reply(
        text="Выберите метод или нажмите Отмена для выхода",
        reply_markup=generate_keyboard(["ML", "DL", "Отмена"], 3)
    )


@dp.message(F.photo, States.ml_foto)
async def ml_photo(message: Message, bot: Bot, state: FSMContext):

    await message.answer("Ваша фотография получена. \
                         \nИдет процесс обработки....")

    # сохраняем в формате class '_io.BytesIO'
    bytes_io_object = await bot.download(message.photo[-1])

    # переводим в numpy
    np_photo = np.frombuffer(bytes_io_object.getvalue(), dtype=np.uint8)

    # кодируем в фотографию
    img = cv2.imdecode(np_photo, cv2.IMREAD_COLOR)

    # сохраняем в корень
    # cv2.imwrite('pic.jpg', img)

    image, face_image, hog_image = ImageProcessing().transform_image(img)
    models_path = os.path.join('../models', 'ML')
    predictions = np.array([])
    for model_name in MODEL_NAMES:
        model = pickle.load(open(os.path.join(models_path, model_name), 'rb'))
        if 'hog' in model_name:
            predictions = np.append(predictions,
                                    model.predict_proba([hog_image.ravel()])[:, 1])
        else:
            predictions = np.append(predictions,
                                    model.predict_proba([image.ravel()])[:, 1])
    avg_pred = np.mean(predictions)
    await message.answer(f"Ваша фотография является Фейком с вероятностью {round(avg_pred,3) * 100}%")
    await message.answer(
        text="Хотите попробовать еще?",
        reply_markup=generate_keyboard(["Да", "Нет"])
    )
    await state.set_state(States.yes_no)


async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
