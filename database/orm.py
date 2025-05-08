from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from database.engine import Message, Database, NeRelevant_filter, Reklama_filter
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
            if result is None:
                return None
            messages = result.scalars().all()
            return messages
    
    async def delete_message(self, message_id: int) -> None:
        """
        Удаляет сообщение из базы данных по его ID.

        Args:
            message_id (int): ID сообщения

        Returns:
            None
        """
        async with self.db.session_factory() as session:
            query = select(Message).where(Message.message_id == message_id)
            result = await session.execute(query)
            message = result.scalar_one_or_none()

            if message:
                await session.delete(message)
                await session.commit()
                self.logger.info(f"Сообщение с ID {message_id} удалено из базы данных")

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

    async def add_reklama_filter(self, text: str) -> Reklama_filter:
        """
        Добавляет новый фильтр рекламы в базу данных.

        Args:
            text (str): Текст фильтра рекламы

        Returns:
            Reklama_filter: Созданный объект фильтра рекламы
        """
        async with self.db.session_factory() as session:
            reklama_filter = Reklama_filter(text=text)
            session.add(reklama_filter)
            await session.commit()
            await session.refresh(reklama_filter)
            self.logger.info(f"Фильтр рекламы '{text}' добавлен в базу данных")
            
    async def get_reklama_filters(self) -> list[Reklama_filter]:
        """
        Получает все фильтры рекламы из базы данных.

        Returns:
            list[Reklama_filter]: Список всех фильтров рекламы
        """
        async with self.db.session_factory() as session:
            query = select(Reklama_filter)
            result = await session.execute(query)
            reklama_filters = result.scalars().all()
            return reklama_filters
    
    async def delete_reklama_filter(self, text: str) -> None:
        """
        Удаляет фильтр рекламы из базы данных по его тексту.

        Args:
            text (str): Текст фильтра рекламы

        Returns:
            None
        """
        async with self.db.session_factory() as session:
            query = select(Reklama_filter).where(Reklama_filter.text == text)
            result = await session.execute(query)
            reklama_filter = result.scalar_one_or_none()

            if reklama_filter:
                await session.delete(reklama_filter)
                await session.commit()
                self.logger.info(f"Фильтр рекламы '{text}' удален из базы данных")
            else:
                self.logger.warning(f"Фильтр рекламы '{text}' не найден в базе данных")


    async def add_ne_relevant_filter(self, text: str) -> NeRelevant_filter:
        """
        Добавляет новый фильтр не релевантных вакансий в базу данных.

        Args:
            text (str): Текст фильтра не релевантных вакансий

        Returns:
            NeRelevant_filter: Созданный объект фильтра не релевантных вакансий
        """
        async with self.db.session_factory() as session:
            ne_relevant_filter = NeRelevant_filter(text=text)
            session.add(ne_relevant_filter)
            await session.commit()
            await session.refresh(ne_relevant_filter)
            self.logger.info(f"Фильтр не релевантных вакансий '{text}' добавлен в базу данных")
    
    async def get_ne_relevant_filters(self) -> list[NeRelevant_filter]:
        """
        Получает все фильтры не релевантных вакансий из базы данных.

        Returns:
            list[NeRelevant_filter]: Список всех фильтров не релевантных вакансий
        """
        async with self.db.session_factory() as session:
            query = select(NeRelevant_filter)
            result = await session.execute(query)
            ne_relevant_filters = result.scalars().all()
            return ne_relevant_filters
        

    async def delete_ne_relevant_filter(self, text: str) -> None:
        """
        Удаляет фильтр не релевантных вакансий из базы данных по его тексту.

        Args:
            text (str): Текст фильтра не релевантных вакансий

        Returns:
            None
        """
        async with self.db.session_factory() as session:
            query = select(NeRelevant_filter).where(NeRelevant_filter.text == text)
            result = await session.execute(query)
            ne_relevant_filter = result.scalar_one_or_none()

            if ne_relevant_filter:
                await session.delete(ne_relevant_filter)
                await session.commit()
                self.logger.info(f"Фильтр не релевантных вакансий '{text}' удален из базы данных")
            else:
                self.logger.warning(f"Фильтр не релевантных вакансий '{text}' не найден в базе данных")