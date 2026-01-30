from SlotObjects.Verticies import Vertex

class FetcherObject:
    __slots__ = 'getter', 'keys'
    
    def __init__(self, getter, keys=[]):
        self.getter = getter
        self.keys = keys

    def __getattribute__(self, key):
        if key in super().__getattribute__('keys'):
            return super().__getattribute__('getter').__getattribute__(key)
        else:
            return super().__getattribute__(key)


class FetchRef(FetcherObject):
    def __init__(self, getter, keys=['center', 'size', 'half_size']):
        super().__init__(getter, keys)

    def abs_center(self, center=Vertex(0, 0)):
        return self.getter.center + self.getter.half_size * center

    def abs_size(self, size=Vertex(1, 1)):
        return self.getter.size * size
    


if __name__ == '__main__':
	pass
