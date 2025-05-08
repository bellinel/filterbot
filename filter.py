import asyncio

from itertools import product

import os
import re

from aiogram import Bot, Dispatcher, Router
from aiogram import types
from dotenv import load_dotenv
from aiogram  import F

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from aiogram.fsm.context import FSMContext
from database.orm import MessageRepository
from database.engine import Database
from aiogram.filters.command import Command

from keyboard import admin_kb, filter_admin_kb, menu_kb, reklama_filter_back_kb, reklama_filter_delete_kb, reklama_kb, relevant_filter_back_kb, relevant_filter_delete_kb, relevant_kb
from aiogram.fsm.state import State, StatesGroup
import pymorphy2

from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.storage.base import StorageKey


filter_router = Router()


morph = pymorphy2.MorphAnalyzer()
load_dotenv()


# Создаем экземпляр базы данных и репозитория сообщений
db = Database()
message_repo = MessageRepository(db)

class ChannelID(StatesGroup):
    
    add_reklama_filter = State()
    add_relevant_filter = State()
    message_id = State()




async def preprocess(text):
    # Улучшенная предобработка текста
    if text is None:
        return ""  # Возвращаем пустую строку, если текст None
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




async def generate_all_case_forms(phrase):
    """Генерирует все возможные падежные формы фразы"""
    words = phrase.split()
    if not words:
        return []
    
    # Получаем все возможные варианты склонения для каждого слова
    word_variants = []
    for word in words:
        parsed = morph.parse(word)[0]  # берем первый вариант разбора
        if parsed.tag.POS in {'NOUN', 'ADJF', 'ADJS', 'PRTF', 'PRTS', 'NUMR'}:
            cases = ['nomn', 'gent', 'datv', 'accs', 'ablt', 'loct']
            variants = []
            for case in cases:
                try:
                    inflected = parsed.inflect({case})
                    if inflected:
                        variants.append(inflected.word)
                except:
                    continue
            word_variants.append(variants if variants else [word])
        else:
            word_variants.append([word])  # для неизменяемых слов
    
    # Генерируем все возможные комбинации слов в разных падежах
    all_forms = []
    for combination in product(*word_variants):
        all_forms.append(' '.join(combination))
    
    return list(set(all_forms))  # убираем дубли





@filter_router.channel_post()
async def filter_message(message: types.Message, bot: Bot):

     
    if message.sender_chat and message.sender_chat.id == (await bot.get_me()).id:
        print("Сообщение от бота — не удаляем")
        return
    
    # Получаем текст из сообщения или подписи
    if message.text:
        text = message.text
        print('текст')
    elif message.caption:
        text = message.caption
        print('описание')
    else:
        print('Сообщение без текста')
        return  # Если нет ни текста, ни подписи, выходим
        
    # Приводим к нижнему регистру, если текст не None
    
    if text:
        text_for_message = text
        text = text.lower()

        
    
    current_message = text
    
    # Инициализируем базу данных, если ещё не инициализирована
    await db.init()
    
    # Получаем все сообщения из базы данных через экземпляр репозитория
    db_messages = await message_repo.get_all_messages()
    
    # Получаем только тексты сообщений из объектов Message
    message_texts = [msg.text for msg in db_messages]
    
    
    # Если в базе нет сообщений, просто добавляем текущее сообщение
    if  message_texts == None:
        
        # При добавлении передаем текст и ID сообщения
        await message_repo.add_message(current_message, message.message_id)
        return
    
    # Сравниваем текущее сообщение со всеми сообщениями в базе
    similarities, most_similar_idx, max_similarity, features = await compare_message_with_all(current_message, message_texts)
    # 
    # Вывод результатов
   
    
    # Вывод сходства с каждым сообщением в базе
    
    
    
    # Вывод наиболее похожего сообщения
   
    
    # Порог для определения дубликатов
    threshold = 0.3
    
    # Проверяем, является ли сообщение новым (уникальным)
    if max_similarity >= threshold:
       
        # Можно добавить дополнительную логику обработки дубликата
        # Например, отправить предупреждение пользователю
       
        
        a = await bot.forward_message(
            chat_id=192659790,
            from_chat_id=message.chat.id,
            message_id=message.message_id
            
            
            
            
        )
        await bot.send_message(
            chat_id=192659790,
            text="Подозрение на дубликат",
            reply_markup= await admin_kb(forward_message_id=a.message_id)
            
        )
        
        
        await bot.delete_message(
            chat_id=message.chat.id,
            message_id=message.message_id
        )
        
        return
    
        # Добавляем сообщение в базу данных с ID сообщения
        
        
    data = await message_repo.get_ne_relevant_filters()
    
    cycle = False
    relevant = False

    # Проверка наличия обязательных фраз (нерелевантные вакансии)
    for phrase in data:
        if cycle:
                cycle = False
                break
        phrase_text = phrase.text
        case_forms = await generate_all_case_forms(phrase_text)
        for i in case_forms:
            

            if i in text[:200]:
                cycle = True
                relevant = True
                break
        
    if relevant == False:
         
                
                a = await bot.forward_message(
                    chat_id=192659790,
                    from_chat_id=message.chat.id,
                    message_id=message.message_id
                    
                    
                    
                

                )
                
                await bot.send_message(
                    chat_id=192659790,
                    text=f"Подозрение на нерелевантную вакансию",
                    reply_markup= await filter_admin_kb(forward_message_id=a.message_id)
                )
                await bot.delete_message(
                    chat_id=message.chat.id,
                    message_id=message.message_id
                )
                

                return
         
                
        

              
        
         
        
      
    data  = await message_repo.get_reklama_filters()

    for phrase in data:
        phrase = phrase.text
        case_forms = await generate_all_case_forms(phrase)
        for i in case_forms:
            if i in text:

                
               
                a = await bot.forward_message(
                    chat_id=192659790,
                    from_chat_id=message.chat.id,
                    message_id=message.message_id
                    
                    
                   

                    

                )
                await bot.send_message(
                    chat_id=192659790,#192659790,
                    text=f"Подозрение на рекламу",
                    reply_markup= await filter_admin_kb(forward_message_id=a.message_id)
                    
                )
                await bot.delete_message(
                    chat_id=message.chat.id,
                    message_id=message.message_id
                )
                
                return
    
    await message_repo.add_message(current_message, message.message_id)







    # Получаем текст текущего сообщения

        
        
        
       


@filter_router.callback_query(F.data.startswith('confirm:'))
async def confirm_message(callback: types.CallbackQuery, bot : Bot):
    GROUP_ID = os.getenv("GROUP_ID")
    
    await callback.answer("Сообщение подтверждено")
    
    forward_message_id = callback.data.split(":")
    forward_message_id = int(forward_message_id[1])
    
    await bot.forward_message(chat_id=int(GROUP_ID), from_chat_id=callback.message.chat.id, message_id=forward_message_id)
    await bot.delete_message(chat_id=callback.message.chat.id, message_id=forward_message_id)
    await MessageRepository(db).add_message(text=callback.message.text, message_id=callback.message.message_id)
    await asyncio.sleep(2)
    await callback.message.delete()

@filter_router.callback_query(F.data.startswith('reject:'))
async def reject_message(callback: types.CallbackQuery,  bot: Bot):
    forward_message_id = callback.data.split(":")
    forward_message_id = int(forward_message_id[1])
    await bot.delete_message(chat_id=callback.message.chat.id, message_id=forward_message_id)
    await callback.answer("Сообщение отклонено")
    await asyncio.sleep(2)
    await callback.message.delete()
    


@filter_router.callback_query(F.data.startswith('confirm_filter:'))
async def confirm_filter(callback: types.CallbackQuery, bot : Bot):
    GROUP_ID = os.getenv("GROUP_ID")
    await callback.answer("Сообщение подтверждено")
    forward_message_id = callback.data.split(":")
    forward_message_id = int(forward_message_id[1])
    
   
    try:
        await bot.forward_message(chat_id=int(GROUP_ID), from_chat_id=callback.message.chat.id, message_id=forward_message_id)
    except:
        print(forward_message_id)
    await bot.delete_message(chat_id=callback.message.chat.id, message_id=forward_message_id)
    await asyncio.sleep(2)
    await callback.message.delete()
    



@filter_router.message(Command('start'))
async def start(message: types.Message):
    if message.from_user.id not in [192659790, 6264939461]:
        return
    await message.answer("Меню бота", reply_markup=await menu_kb())




"""РЕКЛАМНЫЕ ФИЛЬТРЫ"""

@filter_router.callback_query(F.data == 'reklama')
async def reklama(callback: types.CallbackQuery, state: FSMContext):
    await callback.message.edit_text("Управление фильтрами рекламы", reply_markup=await reklama_kb())


@filter_router.callback_query(F.data == 'add_filter_reklama')
async def add_reklama(callback: types.CallbackQuery, state: FSMContext):
    await callback.message.edit_text("Введите текст фильтра рекламы")
    await state.set_state(ChannelID.add_reklama_filter)


@filter_router.message(ChannelID.add_reklama_filter)
async def add_reklama_filter(message: types.Message, state: FSMContext):
    await message_repo.add_reklama_filter(message.text)
    await message.answer("Фильтр рекламы добавлен", reply_markup=await reklama_kb())
    await state.clear()

@filter_router.callback_query(F.data == 'all_filter_reklama')
async def show_reklama(callback: types.CallbackQuery, state: FSMContext):
    reklama_filters = await message_repo.get_reklama_filters()
    message_ids = []
    await callback.message.delete()
    for reklama_filter in reklama_filters:
        a = await callback.message.answer(reklama_filter.text, reply_markup=await reklama_filter_delete_kb())
        message_ids.append(a.message_id)
    await callback.message.answer("Нажмите чтобы вернутся в меню", reply_markup=await reklama_filter_back_kb())

    await state.update_data(message_ids=message_ids)



@filter_router.callback_query(F.data == 'delete_filter_reklama')
async def delete_reklama(callback: types.CallbackQuery, state: FSMContext):
    reklama_text = callback.message.text
    await message_repo.delete_reklama_filter(reklama_text)
    await callback.message.edit_text("Фильтр рекламы удален")
    await asyncio.sleep(1)
    await callback.message.delete()
    

@filter_router.callback_query(F.data == 'back_to_menu')
async def back_to_menu(callback: types.CallbackQuery, state: FSMContext):
    await callback.message.edit_text("Меню бота", reply_markup=await menu_kb())

@filter_router.callback_query(F.data == 'back_to_reklama_menu')
async def back_to_reklama_menu(callback: types.CallbackQuery, state: FSMContext, bot : Bot):
    await callback.message.delete()
    ids = await state.get_data()
    ids = ids.get('message_ids')
    try:
        for id in ids:
           await callback.bot.delete_message(chat_id=callback.message.chat.id, message_id=id)
        await callback.message.answer("Управление фильтрами рекламы", reply_markup=await reklama_kb())
    except:
        await callback.message.answer("Управление фильтрами рекламы", reply_markup=await reklama_kb())
    
    



"""НЕ РЕЛЕВАНТНЫЕ ВАКАНСИИ"""


@filter_router.callback_query(F.data == 'no_relevants_vac')
async def relevant(callback: types.CallbackQuery, state: FSMContext):
    await callback.message.edit_text("Управление фильтрами релевантных вакансий", reply_markup=await relevant_kb())


@filter_router.callback_query(F.data == 'add_filter_relevant')
async def add_relevant(callback: types.CallbackQuery, state: FSMContext):
    await callback.message.edit_text("Введите текст фильтра релевантных вакансий")
    await state.set_state(ChannelID.add_relevant_filter)


@filter_router.message(ChannelID.add_relevant_filter)
async def add_relevant_filter(message: types.Message, state: FSMContext):
    await message_repo.add_ne_relevant_filter(message.text)
    await message.answer("Фильтр добавлен", reply_markup=await relevant_kb())
    await state.clear()

@filter_router.callback_query(F.data == 'all_filter_relevant')
async def show_relevant(callback: types.CallbackQuery, state: FSMContext):
    reklama_filters = await message_repo.get_ne_relevant_filters()
    message_ids = []
    await callback.message.delete()
    for reklama_filter in reklama_filters:
        a = await callback.message.answer(reklama_filter.text, reply_markup=await relevant_filter_delete_kb())
        message_ids.append(a.message_id)
    await callback.message.answer("Нажмите чтобы вернутся в меню", reply_markup=await relevant_filter_back_kb())

    await state.update_data(message_ids=message_ids)



@filter_router.callback_query(F.data == 'delete_filter_relevant')
async def delete_relevant(callback: types.CallbackQuery, state: FSMContext):
    relevant_text = callback.message.text
    await message_repo.delete_ne_relevant_filter(relevant_text)
    await callback.message.edit_text("Фильтр удален")
    await asyncio.sleep(1)
    await callback.message.delete()
    

@filter_router.callback_query(F.data == 'back_to_relevant_menu')
async def back_to_relevant_menu(callback: types.CallbackQuery, state: FSMContext, bot : Bot):
    await callback.message.delete()
    ids = await state.get_data()
    ids = ids.get('message_ids')
    try:
        for id in ids:
            await callback.bot.delete_message(chat_id=callback.message.chat.id, message_id=id)
        await callback.message.answer("Управление фильтрами релевантных вакансий", reply_markup=await relevant_kb())
    except :
        await callback.message.answer("Управление фильтрами релевантных вакансий", reply_markup=await relevant_kb())