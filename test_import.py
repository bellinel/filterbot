import inspect_patch
from aiogram.enums import ChatMemberStatus

# Проверяем, что импорт работает корректно
print("Импорт ChatMemberStatus успешен")
print(f"Доступные значения: {ChatMemberStatus.ADMINISTRATOR}, {ChatMemberStatus.CREATOR}") 