from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from database.engine import Message, Database
import logging
from sqlalchemy import text


class MessageRepository:
    """
    Класс для работы с сообщениями в базе данных.
    """
    def __init__(self, db: Database):
        """
        Инициализирует репозиторий сообщений.
        
        Args:
            db (Database): Экземпляр базы данных
        """
        self.db = db
        self.logger = logging.getLogger(__name__)
    
    async def add_message(self, text: str, message_id: int) -> Message:
        """
        Добавляет новое сообщение в базу данных.
        
        Args:
            text (str): Текст сообщения
            message_id (int): ID сообщения
            
        Returns:
            Message: Созданный объект сообщения
        """
        async with self.db.session_factory() as session:
            message = Message(text=text, message_id=message_id)
            session.add(message)
            await session.commit()
            self.logger.info(f"Сообщение с ID {message_id} добавлено в базу данных")
            return message
    
    async def get_message_by_id(self, message_id: int) -> Message:
        """
        Получает сообщение по его ID.
        
        Args:
            message_id (int): ID сообщения
            
        Returns:
            Message: Найденное сообщение или None
        """
        async with self.db.session_factory() as session:
            query = select(Message).where(Message.message_id == message_id)
            result = await session.execute(query)
            message = result.scalar_one_or_none()
            return message
    
    async def get_all_messages(self) -> list[Message]:
        """
        Получает все сообщения из базы данных.
        
        Returns:
            list[Message]: Список всех сообщений
        """
        async with self.db.session_factory() as session:
            query = select(Message)
            result = await session.execute(query)
            messages = result.scalars().all()
            return messages
            
    async def clear_database(self) -> None:
        """
        Очищает всю базу данных сообщений.
        
        Returns:
            None
        """
        async with self.db.session_factory() as session:
            # Используем text() для создания правильного SQL выражения
            await session.execute(text("DELETE FROM messages"))
            await session.commit()
            self.logger.info("База данных сообщений очищена")


