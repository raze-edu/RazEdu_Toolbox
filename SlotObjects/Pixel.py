from random import randint as rng
from SlotObjects.MathSlotClass import *

class Pixel3(MathObj):
    __slots__ = 'r', 'g', 'b'
    __name__ = 'Pixel3'
    default = 0, 0, 0
    def __init__(self, *args, **kwargs):
        args = [self.default[i] if i >= len(args) else args[i] for i in range(len(self.__slots__))]
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f'{self.__name__} => ' + '/'.join([str(item) for item in self])

    def __tuple__(self):
        return tuple([int(val) for val in self])
    
    @property
    def normalized(self):
        return self.__class__(*[v / self.max * 255 for v in self])

    @classmethod
    def random(cls):
        return cls(*[rng(0, 256) for _ in cls.__slots__])

class Pixel4(Pixel3):
    __slots__ = 'r', 'g', 'b', 'a'
    __name__ = 'Pixel4'
    default = 0, 0, 0, 255
    
class ColorWheel(object):
    def __init__(self, pos:float=.5, colors=(Pixel3(255, 0, 0), Pixel3(0, 255, 0), Pixel3(0, 0, 255))):
        self._prime_colors = colors
        self.position = pos

    @property
    def prime_colors(self):
        return self._prime_colors
    
    @property
    def prime_distance(self):
        return 1 / len(self.prime_colors)    

    @property
    def pos_between_colors(self):
        a = 0.0 if self.position == 0.0 else self.position // self.prime_distance
        b = (a + 1) % len(self.prime_colors)
        return self.prime_colors[int(a)], self.prime_colors[int(b)]

    def get_pos_color(self, normalized=True):
        if self.position == 0.0:
            return self.pos_between_colors[0]
        elif self.position % self.prime_distance == 0.0:
            return self.pos_between_colors[0]
        else:
            dist = (self.position % self.prime_distance) / self.prime_distance
            temp = self.pos_between_colors[0] * (1 - dist) + self.pos_between_colors[1] * dist
            return temp if not normalized else temp.normalized

    def delta(self, val):
        self.position += val

    def __iadd__(self, val:float):
        self.position = self.position + val
        return self

    def __isub__(self, val:float):
        self.position = self.position - val
        return self

    def __setattr__(self, key, val):
        if key == 'position':
            while val < 0.0:
                val += 1
            while val > 1.0:
                val -= 1
        super().__setattr__(key, val)

if __name__ == '__main__':
    t = Pixel4.random()
    print(t, t.__tuple__())