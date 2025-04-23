import inspect
from collections import namedtuple

# Проверяем, есть ли уже функция getargspec в модуле inspect
if not hasattr(inspect, 'getargspec'):
    # Создаем namedtuple для нашего ArgSpec, если его нет
    if not hasattr(inspect, 'ArgSpec'):
        inspect.ArgSpec = namedtuple('ArgSpec', ['args', 'varargs', 'keywords', 'defaults'])
    
    # Если функции нет, добавляем её, используя getfullargspec
    def getargspec(func):
        args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations = inspect.getfullargspec(func)
        return inspect.ArgSpec(args, varargs, varkw, defaults)
    
    # Добавляем функцию в модуль inspect
    inspect.getargspec = getargspec 