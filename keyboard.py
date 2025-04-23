from aiogram.utils.keyboard import InlineKeyboardBuilder
 

async def admin_kb():
    builder = InlineKeyboardBuilder()
    builder.button(text="✅ Оставить", callback_data="confirm")
    builder.button(text="❌ Удалить", callback_data="reject")
    builder.adjust(2)
    return builder.as_markup()


async def menu_kb():
    builder = InlineKeyboardBuilder()
    builder.button(text="Дубли", callback_data="dubli")
    builder.button(text="Релевантные вакансии", callback_data="no_relevants_vac")
    builder.button(text="Подозрения в рекламе", callback_data="reklama")
    builder.adjust(1)
    return builder.as_markup()


async def reklama_kb():
    builder = InlineKeyboardBuilder()
    builder.button(text="Вывести список фильтров", callback_data="all_filter_reklama")
    builder.button(text="+ Добавить фильтр", callback_data="add_filter_reklama")
    builder.button(text="⬅️ Назад", callback_data="back_to_menu")
    
    builder.adjust(2)
    return builder.as_markup()

async def reklama_filter_delete_kb():
    builder = InlineKeyboardBuilder()
    
    builder.button(text="❌ Удалить фильтр", callback_data="delete_filter_reklama")
    builder.adjust()
    return builder.as_markup()

async def reklama_filter_back_kb():
    builder = InlineKeyboardBuilder()
    builder.button(text="⬅️ Назад", callback_data="back_to_reklama_menu")
    builder.adjust()
    return builder.as_markup()  


async def relevant_kb():
    builder = InlineKeyboardBuilder()
    builder.button(text="Вывести список фильтров не релевантных вакансий", callback_data="all_filter_relevant")
    builder.button(text="+ Добавить фильтр", callback_data="add_filter_relevant")
    builder.button(text="⬅️ Назад", callback_data="back_to_menu")
    builder.adjust(1)
    return builder.as_markup()

async def relevant_filter_delete_kb():
    builder = InlineKeyboardBuilder()
    
    builder.button(text="❌ Удалить фильтр", callback_data="delete_filter_relevant")
    builder.adjust()
    return builder.as_markup()

async def relevant_filter_back_kb():
    builder = InlineKeyboardBuilder()
    builder.button(text="⬅️ Назад", callback_data="back_to_relevant_menu")
    builder.adjust()
    return builder.as_markup()



async def filter_admin_kb():
    builder = InlineKeyboardBuilder()
    builder.button(text="✅ Выложить", callback_data="confirm_filter")
    builder.button(text="❌ Удалить", callback_data="reject")
    builder.adjust(2)
    return builder.as_markup()


async def clean_day_kb():
    builder = InlineKeyboardBuilder()
    builder.button(text="Ввести новый период", callback_data="clean_day")
    builder.button(text="⬅️ Назад", callback_data="back_to_menu")
    builder.adjust(2)
    return builder.as_markup()