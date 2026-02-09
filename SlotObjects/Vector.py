from MathSlotClass import MathObj as MAObj

class Vector(MAObj):
    __slots__ = 'x', 'y'
    __name__ = 'Vector'

    def __init__(self, *args, **kwargs):
        args = [0 if i >= len(args) else args[i] for i in range(len(self.__slots__))]
        super().__init__(*args, **kwargs)
        #self.normalize()

    def __repr__(self):
        return f'{self.__name__} => ' + '/'.join([str(item) for item in self])

    def __copy__(self):
        return self.__class__(*[self.__getattribute__(slot) for slot in self.__slots__])

    def __iter__(self):
        return [self.__getattribute__(slot) for slot in self.__slots__].__iter__()

    def __eq__(self, other):
        if not isinstance(other, (self.__class__)):
            return False
        for a, b in zip(self, other):
            if a != b:
                return False
        return True

    @classmethod
    def from_points(cls, a, b):
        return cls(*[b - a for a, b in zip(a, b)])

    def line_intersects(self, axis, point):
        if self.__getattribute__(axis) == 0:
            return None
        return self.in_distance(point / self.__getattribute__(axis), absolute=True)

    def in_distance(self, dist, absolute=False):
        '''
        Caution this isnt accurate for large distances
        '''
        if not absolute:
            dist *= self._mult
        if isinstance(dist, (int, float)):
            return self.equalVert(*[item * dist for item in self])
        elif isinstance(dist, Vertex2D):
            return self.equalVert(*self * dist)

    def rotate(self, degree):
        vect = [self.x, self.y]
        degree = rad(degree)
        vect = (np.array(vect) @ np.array([cos(degree), sin(degree)]), np.array(vect) @ np.array([-sin(degree), cos(degree)]))
        self.x, self.y = vect
        return self

    def normalize(self):
        total = sum([abs(val) for val in self])
        [self.__setattr__(slot, v) for slot, v in zip(self.__slots__, [val / total for val in self])]

# Execution Sandbox
if __name__ == '__main__':
	pass
