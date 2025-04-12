from email import message
import os
from aiogram import Bot, Router
from aiogram import types
from dotenv import load_dotenv
from aiogram  import F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np
import logging
from aiogram.fsm.context import FSMContext
from database.orm import MessageRepository
from database.engine import Database
from keyboard import admin_kb
from aiogram.fsm.state import State, StatesGroup

load_dotenv()

filter_router = Router()
# Создаем экземпляр базы данных и репозитория сообщений
db = Database()
message_repo = MessageRepository(db)

# ID модератора, которому будут пересылаться сообщения на проверку
MODERATOR_ID = int(os.getenv("MODERATOR_ID", "192659790"))
# ID канала (должен быть указан в .env файле)
CHANNEL_ID = os.getenv("CHANNEL_ID")

class ChannelID(StatesGroup):
    channel_id = State()

async def preprocess(text):
    # Улучшенная предобработка текста
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    # Можно добавить удаление стоп-слов или лемматизацию по необходимости
    return text


async def calculate_similarity(texts):
    """Вычисляет матрицу схожести между всеми текстами."""
    # Предобработка всех текстов
    processed_texts = [await preprocess(text) for text in texts]
    
    # Проверка на пустые тексты
    if not processed_texts or all(not text for text in processed_texts):
        return np.zeros((len(texts), len(texts))), []
    
    # Настройка векторизатора с улучшенными параметрами
    vectorizer = TfidfVectorizer(
        min_df=1,       # Минимальная частота слова
        ngram_range=(1, 2)  # Учитывать одиночные слова и биграммы
    )
    
    try:
        # Вычисление TF-IDF матрицы
        tfidf_matrix = vectorizer.fit_transform(processed_texts)
        
        # Вычисление матрицы схожести между всеми текстами
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        return similarity_matrix, vectorizer.get_feature_names_out()
    except Exception as e:
        logging.error(f"Ошибка при расчете схожести: {e}")
        return np.zeros((len(texts), len(texts))), []


async def compare_message_with_all(new_message, all_messages):
    """Сравнивает одно сообщение со всеми имеющимися и возвращает сходства и индекс наиболее похожего"""
    # Добавляем новое сообщение в начало списка
    all_texts = [new_message] + all_messages
    
    # Вычисляем матрицу схожести для всех текстов
    similarity_matrix, features = await calculate_similarity(all_texts)
    
    # Проверка на пустую матрицу
    if similarity_matrix.size == 0 or len(all_messages) == 0:
        return [], 0, 0.0, []
    
    # Извлекаем значения схожести между новым сообщением и всеми остальными
    similarities = similarity_matrix[0, 1:] # Первая строка без первого элемента
    
    # Находим индекс наиболее похожего сообщения
    most_similar_idx = np.argmax(similarities)
    max_similarity = similarities[most_similar_idx]
    
    return similarities, most_similar_idx, max_similarity, features


# Обработчик для сообщений в канале
@filter_router.channel_post()
async def filter_channel_post(message: types.Message, bot: Bot, state: FSMContext):
    """Обрабатывает сообщения, опубликованные в канале"""
    logging.info(f"Получено новое сообщение в канале {message.chat.id}: {message.text}")
    
    if not message.text:
        logging.info("Сообщение без текста, пропускаем")
        return
    
    # Получаем текст текущего сообщения
    current_message = message.text
    
    # Инициализируем базу данных, если ещё не инициализирована
    await db.init()
    
    # Получаем все сообщения из базы данных через экземпляр репозитория
    db_messages = await message_repo.get_all_messages()
    
    # Получаем только тексты сообщений из объектов Message
    message_texts = [msg.text for msg in db_messages]
    
    # Если в базе нет сообщений, просто добавляем текущее сообщение
    if not message_texts:
        logging.info("База сообщений пуста, добавляем первое сообщение")
        # При добавлении передаем текст и ID сообщения
        await message_repo.add_message(current_message, message.message_id)
        return
    
    # Сохраняем ID канала в состоянии
    await state.update_data(channel_id=message.chat.id)
    
    # Сравниваем текущее сообщение со всеми сообщениями в базе
    similarities, most_similar_idx, max_similarity, features = await compare_message_with_all(current_message, message_texts)
    
    # Порог для определения дубликатов
    threshold = 0.3
    
    logging.info(f"Максимальная схожесть: {max_similarity:.2f}, порог: {threshold}")
    
    # Проверяем, является ли сообщение новым (уникальным)
    if max_similarity >= threshold:
        logging.info(f"Обнаружено похожее сообщение! Сходство: {max_similarity:.2f} >= {threshold}")
        
        try:
            # Пересылаем сообщение модератору для проверки
            await bot.copy_message(
                chat_id=MODERATOR_ID,
                from_chat_id=message.chat.id,
                message_id=message.message_id,
                reply_markup=await admin_kb()
            )
            logging.info(f"Сообщение переслано модератору {MODERATOR_ID}")
            
            # Удаляем сообщение из канала
            await bot.delete_message(
                chat_id=message.chat.id,
                message_id=message.message_id
            )
            logging.info(f"Сообщение удалено из канала {message.chat.id}")
            
        except Exception as e:
            logging.error(f"Ошибка при обработке сообщения: {e}")
    else:
        logging.info(f"Сообщение уникально: {max_similarity:.2f} < {threshold}")
        # Добавляем сообщение в базу данных с ID сообщения
        await message_repo.add_message(current_message, message.message_id)
        logging.info(f"Сообщение добавлено в базу данных")


# Обработчик для обычных сообщений (в чатах и личных сообщениях)
@filter_router.message()
async def filter_message(message: types.Message, bot: Bot, state: FSMContext):
    """Обрабатывает обычные сообщения в чатах и личных сообщениях"""
    # Получаем текст текущего сообщения
    current_message = message.text
    
    if not current_message:
        return
    
    # Инициализируем базу данных, если ещё не инициализирована
    await db.init()
    
    # Получаем все сообщения из базы данных через экземпляр репозитория
    db_messages = await message_repo.get_all_messages()
    
    # Получаем только тексты сообщений из объектов Message
    message_texts = [msg.text for msg in db_messages]
    
    # Если в базе нет сообщений, просто добавляем текущее сообщение
    if not message_texts:
        # При добавлении передаем текст и ID сообщения
        await message_repo.add_message(current_message, message.message_id)
        return
    
    # Сравниваем текущее сообщение со всеми сообщениями в базе
    similarities, most_similar_idx, max_similarity, features = await compare_message_with_all(current_message, message_texts)
    
    # Порог для определения дубликатов
    threshold = 0.3
    
    # Проверяем, является ли сообщение новым (уникальным)
    if max_similarity >= threshold:
        # Отправляем модератору для проверки
        await bot.copy_message(
            chat_id=MODERATOR_ID,
            from_chat_id=message.chat.id,
            message_id=message.message_id,
            reply_markup=await admin_kb()
        )
        await message.delete()
    else:
        # Добавляем сообщение в базу данных с ID сообщения
        await message_repo.add_message(current_message, message.message_id)


@filter_router.callback_query(F.data == 'confirm')
async def confirm_message(callback: types.CallbackQuery, bot: Bot, state: FSMContext):
    GROUP_ID = os.getenv("GROUP_ID")
    await callback.answer("Сообщение подтверждено")
    
    # Получаем ID канала из состояния
    data = await state.get_data()
    channel_id = data.get("channel_id", GROUP_ID)
    
    # Если ID канала не найден, используем GROUP_ID из .env
    if not channel_id:
        channel_id = GROUP_ID
    
    try:
        await bot.copy_message(
            chat_id=int(channel_id), 
            from_chat_id=callback.message.chat.id, 
            message_id=callback.message.message_id
        )
        await MessageRepository(db).add_message(text=callback.message.text, message_id=callback.message.message_id)
        await callback.message.delete()
    except Exception as e:
        logging.error(f"Ошибка при подтверждении сообщения: {e}")
        await callback.message.answer(f"Ошибка при подтверждении сообщения: {e}")

@filter_router.callback_query(F.data == 'reject')
async def reject_message(callback: types.CallbackQuery):
    await callback.answer("Сообщение отклонено")
    await callback.message.delete()