from SlotObjects.BaseSlotClass import *

class MathObj(SlotBase):
    __name__ = 'MathSlots'
    __slots__ = ()
    __default__ = ()
    __types__ = ()
    
    def _convert(self, other):
        if isinstance(other, (list, tuple)):
            return self.__class__(*other)
        if isinstance(other, dict):
            return self.__class__(**other)
        return other

    # something to handle div by zero issues
    @staticmethod
    def div_zero(a, b, flag=True):
        a, b = [v if not v is None else v for v in [a, b]]
        if a == 0 or b == 0 or (a == 0 and b == 0):
            return .0 if flag else 0
        return a / b if flag else a // b


    # methods that create an object out of a set of its instances
    @classmethod
    def sum_of_objs(cls, *args):
        temp = cls(*args[0])
        for arg in args[1:]:
            temp += arg
        return temp

    @classmethod
    def avg_of_objs(cls, *args):
        temp = cls.sum_of_objs(*[cls(*arg) for arg in args])
        return temp / len(args)

    @classmethod
    def avg_weigthed_objs(cls, *args):
        total_weight = sum([arg[0] for arg in args])
        temp = cls.sum_of_objs(*[cls(*arg[1]) * arg[0] for arg in args])
        return temp / total_weight

    def make_weighted(self, weight=1.0):
        return (weight, self)

    # propertys
    @property
    def max(self):
        return max([self.__getattribute__(slot) for slot in self.__slots__ if not self.__getattribute__(slot) is None])

    @property
    def min(self):
        return min([self.__getattribute__(slot) for slot in self.__slots__ if not self.__getattribute__(slot) is None])

    @property
    def avg(self):
        return sum([self.__getattribute__(slot) for slot in self.__slots__]) / len(self.__slots__)

    @property
    def relative_max(self):
        return [self.__getattribute__(slot) / self.max for slot in self.__slots__]

    @property
    def relative_sum(self):
        return [self.__getattribute__(slot) / self.sum for slot in self.__slots__]

    # arithmetic

    def __add__(self, other):
        if isinstance(other, MathObj):
            return self.__class__(*[self.__getattribute__(slot) if not slot in other.__slots__ else self.__getattribute__(slot) + other.__getattribute__(slot) for slot in self.__slots__])
        elif isinstance(other, (int, float)):
            return self.__class__(*[self.__getattribute__(slot) + other for slot in self.__slots__])
        elif isinstance(other, (list, tuple, dict)):
            return self + self._convert(other)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, MathObj):
            return self.__class__(*[self.__getattribute__(slot) if not slot in other.__slots__ else self.__getattribute__(slot) - other.__getattribute__(slot) for slot in self.__slots__])
        elif isinstance(other, (int, float)):
            return self.__class__(*[self.__getattribute__(slot) - other for slot in self.__slots__])
        elif isinstance(other, (list, tuple, dict)):
            return self - self._convert(other)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, MathObj):
            return self.__class__(*[self.__getattribute__(slot) if not slot in other.__slots__ else self.__getattribute__(slot) * other.__getattribute__(slot) for slot in self.__slots__])
        elif isinstance(other, (int, float)):
            return self.__class__(*[self.__getattribute__(slot) * other for slot in self.__slots__])
        elif isinstance(other, (list, tuple, dict)):
            return self * self._convert(other)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, MathObj):
            temp = {}
            for slot in self.__slots__:
                if not slot in other.__slots__ or other.__getattribute__(slot) == 0:
                    temp[slot] = self.__getattribute__(slot)
                else:
                    temp[slot] = self.__getattribute__(slot) / other.__getattribute__(slot)
            return self.__class__(**temp)
        elif isinstance(other, (int, float)):
            return self.__class__(*[self.__getattribute__(slot) if other == 0 else self.__getattribute__(slot) / other for slot in self.__slots__])
        elif isinstance(other, (list, tuple, dict)):
            return self // self._convert(other)
        return NotImplemented

    def __floordiv__(self, other):
        if isinstance(other, MathObj):
            temp = {}
            for slot in self.__slots__:
                if not slot in other.__slots__ or other.__getattribute__(slot) == 0:
                    temp[slot] = self.__getattribute__(slot)
                else:
                    temp[slot] = self.__getattribute__(slot) // other.__getattribute__(slot)
            return self.__class__(**temp)
        elif isinstance(other, (int, float)):
            return self.__class__(*[self.__getattribute__(slot) if other == 0 else self.__getattribute__(slot) // other for slot in self.__slots__])
        elif isinstance(other, (list, tuple, dict)):
            return self / self._convert(other)
        return NotImplemented

    def __mod__(self, other):
        if isinstance(other, MathObj):
            temp = {}
            for slot in self.__slots__:
                if not slot in other.__slots__ or other.__getattribute__(slot) == 0:
                    temp[slot] = self.__getattribute__(slot)
                else:
                    temp[slot] = self.__getattribute__(slot) // other.__getattribute__(slot)
            return self.__class__(**temp)
        elif isinstance(other, (int, float)):
            return self.__class__(*[self.__getattribute__(slot) if other == 0 else self.__getattribute__(slot) // other for slot in self.__slots__])
        elif isinstance(other, (list, tuple, dict)):
            return self / self._convert(other)
        return NotImplemented

    def __pow__(self, other):
        if isinstance(other, MathObj):
            return self.__class__(*[self.__getattribute__(slot) if not slot in other.__slots__ else self.__getattribute__(slot) ** other.__getattribute__(slot) for slot in self.__slots__])
        elif isinstance(other, (int, float)):
            return self.__class__(*[self.__getattribute__(slot) ** other for slot in self.__slots__])
        elif isinstance(other, (list, tuple, dict)):
            return self ** self._convert(other)
        return NotImplemented
    # arythmetic inplace

    def __neg__(self):
        return self.__class__(*[-self.__getattribute__(slot) for slot in self.__slots__])

    def __iadd__(self, other):
        self = self + other
        return self

    def __isub__(self, other):
        self = self - other
        return self

    def __imul__(self, other):
        self = self * other
        return self

    def __itruediv__(self, other):
        self = self / other
        return self

    def __idiv__(self, other):
        self = self // other
        return self
