import os
from aiogram import Bot, Router
from aiogram import types
from dotenv import load_dotenv
from aiogram  import F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np

from database.orm import MessageRepository
from database.engine import Database
from keyboard import admin_kb


load_dotenv()

filter_router = Router()
# Создаем экземпляр базы данных и репозитория сообщений
db = Database()
message_repo = MessageRepository(db)


async def preprocess(text):
    # Улучшенная предобработка текста
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


@filter_router.channel_post()
async def filter_message(message: types.Message , bot: Bot  ):
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
        
        # При добавлении передаем текст и ID сообщения
        await message_repo.add_message(current_message, message.message_id)
        return
    
    # Сравниваем текущее сообщение со всеми сообщениями в базе
    similarities, most_similar_idx, max_similarity, features = await compare_message_with_all(current_message, message_texts)
    
    # Вывод результатов
   
    
    # Вывод сходства с каждым сообщением в базе
    
    
    
    # Вывод наиболее похожего сообщения
   
    
    # Порог для определения дубликатов
    threshold = 0.3
    
    # Проверяем, является ли сообщение новым (уникальным)
    if max_similarity >= threshold:
       
        # Можно добавить дополнительную логику обработки дубликата
        # Например, отправить предупреждение пользователю
        await bot.copy_message(
            chat_id=6264939461,
            from_chat_id=message.chat.id,
            message_id=message.message_id,
            reply_markup= await admin_kb()
            
        )
        
        await bot.delete_message(
            chat_id=message.chat.id,
            message_id=message.message_id
        )
          
        
    else:
        
        
        # Добавляем сообщение в базу данных с ID сообщения
        await message_repo.add_message(current_message, message.message_id)
        
        
        
       


@filter_router.callback_query(F.data == 'confirm')
async def confirm_message(callback: types.CallbackQuery, bot : Bot):
    GROUP_ID = os.getenv("GROUP_ID")
    await callback.answer("Сообщение подтверждено")
    await bot.copy_message(chat_id=int(GROUP_ID), from_chat_id=callback.message.chat.id, message_id=callback.message.message_id)
    await MessageRepository(db).add_message(text=callback.message.text, message_id=callback.message.message_id)
    await callback.message.delete()

@filter_router.callback_query(F.data == 'reject')
async def reject_message(callback: types.CallbackQuery):
    await callback.answer("Сообщение отклонено")
    await callback.message.delete()