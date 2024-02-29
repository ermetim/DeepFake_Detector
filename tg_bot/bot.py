import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.enums import ParseMode
from aiogram.types import Message
from aiogram.filters import Command
from aiogram import F
from config_reader import config
import os
import pickle
import cv2
import numpy as np
from src.ml_functions import ImageProcessing
MODEL_NAMES = ['lgbm_model_hog_best.pkl',
               'lgbm_model_pic_best.pkl',
               'rfc_model_hog_best.pkl',
               'rfc_model_pic_best.pkl',
               ]


# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)
# Объект бота
bot = Bot(token=config.bot_token.get_secret_value())
# Диспетчер
dp = Dispatcher()

# Хэндлер на команду /start
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    kb = [
        [
            types.KeyboardButton(text="Попробуем"),
            types.KeyboardButton(text="В следующий раз")
        ],
    ]
    keyboard = types.ReplyKeyboardMarkup(
        keyboard=kb,
        resize_keyboard=True,
        input_field_placeholder="'Выберите ответ'"
    )
    await message.answer(
        f"Привет, <b>{message.from_user.full_name}</b>",
        parse_mode=ParseMode.HTML
    )
    await message.answer("Это бот по детекции Дипфейков на фотографии", reply_markup=keyboard)
    await message.answer("Хотите попробовать", reply_markup=keyboard)

@dp.message(F.text.lower() == "попробуем")
async def ml_method(message: types.Message):
    await message.reply("Отличный выбор! Давайте начнем", reply_markup=types.ReplyKeyboardRemove())
    # await message.answer("/start_detection'")
    await start_detection(message)

@dp.message(F.text.lower() == "в следующий раз")
async def dl_method(message: types.Message):
    await message.reply("Очень жаль.", reply_markup=types.ReplyKeyboardRemove())
    await message.answer("Пока.")

@dp.message(Command("start_detection"))
async def cmd_start_detection(message: types.Message):
    await start_detection(message)

async def start_detection(message: types.Message):
    kb = [
        [
            types.KeyboardButton(text="ML"),
            types.KeyboardButton(text="DL")
        ],
    ]
    keyboard = types.ReplyKeyboardMarkup(
        keyboard=kb,
        resize_keyboard=True,
        input_field_placeholder="'Выберите метод детекции'"
    )
    await message.answer("Какой метод хотите попробовать", reply_markup=keyboard)

@dp.message(F.text.lower() == "ml")
async def ml_method(message: types.Message):
    await message.reply("Machine Learning - отличный выбор!", reply_markup=types.ReplyKeyboardRemove())
    await message.answer("Добавьте фотографию")

@dp.message(F.text.lower() == "dl")
async def dl_method(message: types.Message):
    await message.reply("Deep Learning. К сожалению, данный метод пока еще в разработке. Попробуйте ML метод")

# хендлер на текстовое сообщение от пользователя
@dp.message(F.photo)
async def download_photo(message: Message, bot: Bot):
    # сохраняем в формате class '_io.BytesIO'
    bytes_io_object = await bot.download(message.photo[-1])
    # переводим в numpy
    np_photo = np.frombuffer(bytes_io_object.getvalue(), dtype=np.uint8)
    # кодируем в фотографию
    img = cv2.imdecode(np_photo, cv2.IMREAD_COLOR)
    # сохраняем в корень
    # cv2.imwrite('pic.jpg', img)

    image, face_image, hog_image = ImageProcessing().transform_image(img)
    models_path = os.path.join('models', 'ML')
    predictions = np.array([])
    for model_name in MODEL_NAMES:
        model = pickle.load(open(os.path.join(models_path, model_name), 'rb'))
        if 'hog' in model_name:
            predictions = np.append(predictions, model.predict_proba([hog_image.ravel()])[:,1])
        else:
            predictions = np.append(predictions, model.predict_proba([image.ravel()])[:,1])
    avg_pred = np.mean(predictions)
    await message.answer(f"Ваша фотография является Фейком с вероятностью {round(avg_pred,3) * 100}%")

    kb = [
        [
            types.KeyboardButton(text="Попробуем"),
            types.KeyboardButton(text="В следующий раз")
        ],
    ]
    keyboard = types.ReplyKeyboardMarkup(
        keyboard=kb,
        resize_keyboard=True,
        input_field_placeholder="'Выберите ответ'"
    )
    await message.answer("Хотите попробовать еще?", reply_markup=keyboard)

# Запуск процесса поллинга новых апдейтов
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
