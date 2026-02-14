class Prefix:
    __slots__ = 'prefix', 'base', 'offset'
    def __init__(self, pre_strings, stepsize):
        self.prefix = pre_strings
        self.base = stepsize
        self.offset = self.prefix.index('')
        
    def get_factor(self, prefix):
        return self.base ** (self.offset - self.prefix.index(prefix))

    @property
    def value(self):
        class Value(int):
            __name__ = 'Value'
            _prefix = self.prefix
            _base = self.base
            def __new__(cls, value, **kwargs):
                print(len(cls._prefix)-1, cls._prefix.index(kwargs.get('prefix', '')), kwargs.get('prefix', ''), value)
                if value != 0:
                    return super().__new__(cls, round(value * (cls._base**((len(cls._prefix)-1) - cls._prefix.index(kwargs.get('prefix', '')))), 0))
                else:
                    return super().__new__(cls, 0)

            def __list__(self):
                return [int(self) * self._base ** -i for i in range(0, len(self._prefix))][::-1]

            @property
            def autofix(self):
                for val, name in zip(self.__list__(), self._prefix):
                    if 1000 > abs(val) >= 1:
                        return val, name
                return val, name

            def __str__(self):
                return f"{'{:.3f}'.format(self.autofix[0])} {self.autofix[1]}Unit"

            def get(self, prefix=''):
                return self.__list__()[self._prefix.index(prefix)]

            def class_check(self, other):
                print('classcheck')
                return int(other) if other.__class__.__name__ == self.__class__.__name__ else int(Value(other))

            def __add__(self, other):
                return Value(int(self) + self.class_check(other), prefix=self._prefix[-1])

            def __sub__(self, other):
                return Value(int(self) - self.class_check(other), prefix=self._prefix[-1])

            def __mul__(self, other):
                return Value(int(self) * self.class_check(other), prefix=self._prefix[-1])
            
            def __truediv__(self, other):
                return Value(int(self) / self.class_check(other), prefix=self._prefix[-1])
        return Value


class PrefixLib:
  SI = Prefix(('Peta', 'Tera', 'Giga', 'Mega', 'Kilo', '', 'Milli', 'Micro', 'Nano', 'Pico'), 1000)
  IT = Prefix(('Peta', 'Tera', 'Giga', 'Mega', 'Kilo', ''), 1024)

