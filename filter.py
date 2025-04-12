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

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
    # Настройка векторизатора с улучшенными параметрами
    vectorizer = TfidfVectorizer(
        min_df=1,       # Минимальная частота слова
        ngram_range=(1, 2)  # Учитывать одиночные слова и биграммы
    )
    
    # Вычисление TF-IDF матрицы
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    
    # Вычисление матрицы схожести между всеми текстами
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    return similarity_matrix, vectorizer.get_feature_names_out()


async def compare_message_with_all(new_message, all_messages):
    """Сравнивает одно сообщение со всеми имеющимися и возвращает сходства и индекс наиболее похожего"""
    # Добавляем новое сообщение в начало списка
    all_texts = [new_message] + all_messages
    
    # Вычисляем матрицу схожести для всех текстов
    similarity_matrix, features = await calculate_similarity(all_texts)
    
    # Извлекаем значения схожести между новым сообщением и всеми остальными
    similarities = similarity_matrix[0, 1:] # Первая строка без первого элемента
    
    # Находим индекс наиболее похожего сообщения
    most_similar_idx = np.argmax(similarities)
    max_similarity = similarities[most_similar_idx]
    
    return similarities, most_similar_idx, max_similarity, features


# Удаляем старый обработчик channel_post и создаем новый
@filter_router.channel_post()
async def process_channel_post(message: types.Message, bot: Bot, state: FSMContext):
    """Обрабатывает сообщения из канала"""
    logger.info(f"Обработка сообщения из канала: {message.message_id}")
    
    # Если нет текста, пропускаем обработку
    if not message.text:
        logger.info("Сообщение без текста, пропускаем")
        return
    
    current_message = message.text
    
    try:
        # Инициализируем базу данных
        await db.init()
        
        # Получаем сообщения из базы данных
        db_messages = await message_repo.get_all_messages()
        message_texts = [msg.text for msg in db_messages]
        
        # Если база пуста, добавляем первое сообщение
        if not message_texts:
            logger.info("База сообщений пуста, добавляем первое сообщение")
            await message_repo.add_message(current_message, message.message_id)
            return
        
        # Сохраняем ID канала в состоянии
        await state.update_data(channel_id=message.chat.id)
        
        # Сравниваем с существующими сообщениями
        similarities, most_similar_idx, max_similarity, features = await compare_message_with_all(current_message, message_texts)
        
        # Пороговое значение для определения схожести
        threshold = 0.3
        
        # Проверяем схожесть
        if max_similarity >= threshold:
            logger.info(f"Обнаружено похожее сообщение! Схожесть: {max_similarity:.2f}")
            
            # Отправляем модератору
            await bot.copy_message(
                chat_id=6264939461,
                from_chat_id=message.chat.id,
                message_id=message.message_id,
                reply_markup=await admin_kb()
            )
            
            # Удаляем из канала
            await bot.delete_message(
                chat_id=message.chat.id,
                message_id=message.message_id
            )
            
            logger.info(f"Сообщение удалено из канала {message.chat.id}")
        else:
            logger.info(f"Сообщение уникально (схожесть: {max_similarity:.2f})")
            await message_repo.add_message(current_message, message.message_id)
            
    except Exception as e:
        logger.error(f"Ошибка при обработке сообщения: {e}")


# Обработчик обычных сообщений
@filter_router.message()
async def process_message(message: types.Message, bot: Bot, state: FSMContext):
    """Обрабатывает обычные сообщения"""
    logger.info(f"Обработка обычного сообщения: {message.message_id}")
    
    # Если нет текста, пропускаем обработку
    if not message.text:
        return
    
    current_message = message.text
    
    try:
        # Инициализируем базу данных
        await db.init()
        
        # Получаем сообщения из базы данных
        db_messages = await message_repo.get_all_messages()
        message_texts = [msg.text for msg in db_messages]
        
        # Если база пуста, добавляем первое сообщение
        if not message_texts:
            await message_repo.add_message(current_message, message.message_id)
            return
        
        # Сравниваем с существующими сообщениями
        similarities, most_similar_idx, max_similarity, features = await compare_message_with_all(current_message, message_texts)
        
        # Пороговое значение для определения схожести
        threshold = 0.3
        
        # Проверяем схожесть
        if max_similarity >= threshold:
            # Отправляем модератору
            await bot.copy_message(
                chat_id=6264939461,
                from_chat_id=message.chat.id,
                message_id=message.message_id,
                reply_markup=await admin_kb()
            )
            
            # Удаляем сообщение
            await message.delete()
        else:
            await message_repo.add_message(current_message, message.message_id)
            
    except Exception as e:
        logger.error(f"Ошибка при обработке сообщения: {e}")


@filter_router.callback_query(F.data == 'confirm')
async def confirm_message(callback: types.CallbackQuery, bot: Bot, state: FSMContext):
    """Обработчик для подтверждения сообщения"""
    GROUP_ID = os.getenv("GROUP_ID")
    
    try:
        await callback.answer("Сообщение подтверждено")
        
        # Получаем ID канала из состояния
        data = await state.get_data()
        channel_id = data.get("channel_id", GROUP_ID)
        
        if not channel_id:
            channel_id = GROUP_ID
            
        logger.info(f"Подтверждение сообщения, отправка в канал {channel_id}")
        
        # Копируем сообщение в канал/группу
        await bot.copy_message(
            chat_id=int(channel_id), 
            from_chat_id=callback.message.chat.id, 
            message_id=callback.message.message_id
        )
        
        # Сохраняем сообщение в базе данных
        await MessageRepository(db).add_message(
            text=callback.message.text, 
            message_id=callback.message.message_id
        )
        
        # Удаляем сообщение с кнопками
        await callback.message.delete()
        
    except Exception as e:
        logger.error(f"Ошибка при подтверждении сообщения: {e}")
        await callback.message.answer(f"Ошибка: {e}")


@filter_router.callback_query(F.data == 'reject')
async def reject_message(callback: types.CallbackQuery, state: FSMContext):
    """Обработчик для отклонения сообщения"""
    try:
        await callback.answer("Сообщение отклонено")
        await callback.message.delete()
        logger.info("Сообщение отклонено и удалено")
    except Exception as e:
        logger.error(f"Ошибка при отклонении сообщения: {e}")