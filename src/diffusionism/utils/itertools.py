from itertools import *

class shift:
    class unfill: ...
    
    iterable: object
    index: int
    fillvalue: object
    
    @staticmethod
    def __new__(cls, iterable, index: int = 0, fillvalue: object = unfill):
        instance = super().__new__(cls)
        instance.iterable = iterable
        instance.index = index
        instance.fillvalue = fillvalue
        return instance
    
    def __setattr__(self, name, value):
        try:
            if self.__iterable is not None:
                super().__setattr__('_shift__iterable', None)
        except AttributeError:
            super().__setattr__('_shift__iterable', None)
        return super().__setattr__(name, value)
    
    def move(self, index: int = 0):
        self.index = index
    
    def fill(self, value: object = unfill):
        self.fillvalue = value
    
    def __iter__(self):
        if self.__iterable is not None:
            return self.__iterable
        iterable = self.iterable
        index = self.index
        fillvalue = self.fillvalue
        if hasattr(iterable, '__getitem__'):
            if index > 0:
                iterable = iter(iterable[index:])
                if fillvalue is not shift.unfill:
                    iterable = chain(iterable, repeat(fillvalue, index))
            elif index < 0:
                iterable = iter(iterable[:index])
                if fillvalue is not shift.unfill:
                    iterable = chain(repeat(fillvalue, -index), iterable)
            else:
                iterable = iter(iterable)
        else:
            from typing import Iterator
            if not isinstance(iterable, Iterator):
                iterable = iter(iterable)
            else:
                self.iterable, iterable = tee(iterable, 2)
            if index > 0:
                iterable = islice(iterable, index, None)
                if fillvalue is not shift.unfill:
                    iterable = chain(iterable, repeat(fillvalue, index))
            elif index < 0:
                iterable = islice(iterable, -index)
                if fillvalue is not shift.unfill:
                    iterable = chain(repeat(fillvalue, -index), iterable)
        self.__iterable = iterable
        return self.__iterable
    
    def __next__(self):
        if self.__iterable is None:
            iter(self)
        return next(self.__iterable)