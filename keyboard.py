from aiogram.utils.keyboard import InlineKeyboardBuilder
 

async def admin_kb():
    builder = InlineKeyboardBuilder()
    builder.button(text="✅ Оставить", callback_data="confirm")
    builder.button(text="❌ Удалить", callback_data="reject")
    builder.adjust(2)
    return builder.as_markup()
