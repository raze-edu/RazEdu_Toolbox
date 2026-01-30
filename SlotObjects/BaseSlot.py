class SlotBase:
    __name__ = 'BaseSlot'
    __slots__ = ()
    __types__ = ()
    __default__ = ()

    def __init__(self, *args, **kwargs):
        [self.__setattr__(self.__slots__[i], val) for i, val in enumerate(self.__default__) if i < len(self.__slots__)]
        [self.__setattr__(self.__slots__[i], val) for i, val in enumerate(args) if i < len(self.__slots__)]
        [self.__setattr__(key, val) for key, val in kwargs.items() if key in self.__slots__]

    def __setattr__(self, slot, var):
        if len(self.__types__) != len(self.__slots__):
            super().__setattr__(slot, var)
        else:
            if isinstance(slot, int) and isinstance(var, self.__types__[slot]):
                super().__setattr__(self.__slots__[slot], var)
            elif isinstance(var, self.__types__[self.__slots__.index(slot)]):
                super().__setattr__(slot, var)
                            
    def __getattribute__(self, item):
        if isinstance(item, int) and 0 < item < len(self.__slots__):
            return super().__getattribute__(self.__slots__[item])
        elif isinstance(item, str):
            return super().__getattribute__(item)
        elif isinstance(item, slice):
            return [super().__getattribute__(slot) for slot in self.__slots__][item]
        else:
            return None

    def __eq__(self, other):
        return all([a == b for a, b in zip(self.__iter__(), other.__iter__())])

    def __iter__(self):
        return (self.__getattribute__(slot) for slot in self.__slots__)

    def __list__(self):
        return [self.__getattribute__(slot) for slot in self.__slots__]

    def __tuple__(self):
        return tuple([self.__getattribute__(slot) for slot in self.__slots__])

    def __dict__(self):
        return {slot: self.__getattribute__(slot) for slot in self.__slots__}

    def __repr__(self):
        temp = f'{self.__class__.__name__}:\n'
        for key, val in self.__dict__().items():
            temp += f'\t{key}\t: {val}\n'
        return temp

    def get(self, *args, default=None):
        return [got if not got is None else default for got in [self.__getattribute__(arg) for arg in args]]
