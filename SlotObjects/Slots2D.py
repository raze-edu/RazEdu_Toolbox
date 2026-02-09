from BaseSlot import *
from ToolBox.print_align import *

class Slot2D:
    __name__ = '2DSlot'
    __slots__ = ()

    def __init__(self, *args):
        for i, arg in enumerate(args):
            if i < len(self.__slots__):
                self.__setattr__(self.__slots__[i], self.Slot_(*arg))

    @property
    def Slot_(self):
        slots = tuple([f'{s}' for s in self.__slots__])
        class SB(SlotBase):
            __slots__ = slots
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
        return SB

    @property
    def keys(self):
        keys = []
        [[keys.append(f'{key}:{k}') for k in self.__slots__] for key in self.__slots__]
        return (key for key in keys)
        
    def __iter__(self):
        return (self.__getattribute__(slot) for slot in self.__slots__)

    def __dict__(self):
        return {key: self.__getitem__(key) for key in self.keys}

    def __list__(self):
        return [val for val in self.__dict__().values()]

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return all([a == b for a, b in zip(self.__list__(), other.__list__())])

    def __getitem__(self, item):
        return self.__getattribute__(item.split(':')[0]).__getattribute__(item.split(':')[1])

    def __setitem__(self, item, val):
        self.__getattribute__(item.split(':')[0]).__setattr__(item.split(':')[1], val)

class Math2D(Slot2D):
    __name__ = '2DMathSlot'
    __slots__ = ()

    @classmethod
    def sum_of_objs(cls, *args):
        return cls(*[cls().Slot_.sum_of_objs(*arg).__tuple__() for arg in zip(*args)])

    @classmethod
    def avg_of_objs(cls, *args):
        return cls.sum_of_objs(*args) / len(args)

    @property
    def Slot_(self):
        class SM(MathObj):
            __slots__ = self.__slots__
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
        return SM

    def __repr__(self):
        string = f'{self.__class__.__name__} <\n'
        strings = []
        for slot, item in zip(self.__slots__, self):
            temp = f'{item}'.split('\n')
            temp[0] = f'\t{slot} => {temp[0]} => '
            for part in temp[1:]:
                part = part.replace('\t', ' ')
                temp[0] += f'| {part} '
            #strings += f'{temp[0]}\n'
            strings.append(temp[0])
        return string + repr_helper().align(strings) + '>'


    def math_other(self, obj):
        if isinstance(obj, (MathObj, Math2D)):
            return (item for item in obj)
        elif isinstance(obj, (int, float)):
            size = len(self.__slots__)
            return [[obj]*size]*size

    def __add__(self, other):
        temp = self.math_other(other)
        if not temp is None:
            return self.__class__(*[a.__add__(b).__tuple__() for a, b in zip(self, temp)])
    
    def __iadd__(self, other):
        temp = self + other
        if not temp is None:
            self = temp
        return self

    def __mul__(self, other):
        temp = self.math_other(other)
        if not temp is None:
            return self.__class__(*[a.__mul__(b).__tuple__() for a, b in zip(self, temp)])
    
    def __imul__(self, other):
        temp = self * other
        if not temp is None:
            self = temp
        return self

    def __sub__(self, other):
        temp = self.math_other(other)
        if not temp is None:
            return self.__class__(*[a.__sub__(b).__tuple__() for a, b in zip(self, temp)])
    
    def __isub__(self, other):
        temp = self - other
        if not temp is None:
            self = temp
        return self

    def __truediv__(self, other):
        temp = self.math_other(other)
        if not temp is None:
            return self.__class__(*[a.__truediv__(b).__tuple__() for a, b in zip(self, temp)])
    
    def __itruediv__(self, other):
        temp = self / other
        if not temp is None:
            self = temp
        return self

    def __floordiv__(self, other):
        temp = self.math_other(other)
        if not temp is None:
            return self.__class__(*[a.__floordiv__(b).__tuple__() for a, b in zip(self, temp)])
    
    def __ifloordiv__(self, other):
        temp = self // other
        if not temp is None:
            self = temp
        return self

    def __mod__(self, other):
        temp = self.math_other(other)
        if not temp is None:
            return self.__class__(*[a.__mod__(b).__tuple__() for a, b in zip(self, temp)])
    
    def __imod__(self, other):
        temp = self + other
        if not temp is None:
            self = temp
        return self

    def __pow__(self, other):
        temp = self.math_other(other)
        if not temp is None:
            return self.__class__(*[a.__pow__(b).__tuple__() for a, b in zip(self, temp)])

    def __ipow__(self, other):
        temp = self ** other
        if not temp is None:
            self = temp
        return self

    @property
    def sum(self):
        return sum(self.__list__())
        
    @property
    def avg(self):
        return self.sum / (len(self.__slots__) ** 2)

    def __list__(self):
        arr = []
        for key in self.__slots__:
            for k in self.__slots__:
                arr.append(self.__getattribute__(key).__getattribute__(k))
        return arr

    def __abs__(self):
        return self.__class__(*[[abs(val) for val in obj] for obj in self])

    def __neg__(self):
        return self.__class__(*[[-val for val in obj] for obj in self])

    def max(self):
        '''
        return a self.Slot_ object with the max value for each slot
        '''
        return self.Slot_(*[max(attr) for attr in zip(*[obj for obj in self])])
    
    def min(self):
        '''
        return a self.Slot_ object with the min value for each slot
        '''
        return self.Slot_(*[min(attr) for attr in zip(*[obj for obj in self])])
