from MathSlotClass import MathObj as MAObj

class Vertex(MAObj):
	__slots__ = 'x', 'y'
	__name__ = 'Vertex2D'
	
	def __init__(self, *args, **kwargs):
		args = [0 if i >= len(args) else args[i] for i in range(len(self.__slots__))]
		super().__init__(*args, **kwargs)
	
	def __repr__(self):
		return f'{self.__name__} => ' + '/'.join([str(item) for item in self])
	
	def __copy__(self):
		return self.__class__(*[self.__getattribute__(slot) for slot in self.__slots__])
	
	def __eq__(self, other):
		if not isinstance(other, (self.__class__)):
			return False
		for a, b in zip(self, other):
			if a != b:
				return False
		return True
	
	def __iter__(self):
		return [self.__getattribute__(slot) for slot in self.__slots__].__iter__()
	
	def __lt__(self, other):
		assert isinstance(other, (self.__class__)), f'other must be of type VirtualVertex, Vertex or Vector but is {type(other)}'
		return all([a < b for a, b in zip(self, other)])
		
	def __gt__(self, other):
		assert isinstance(other, (self.__class__)), f'other must be of type VirtualVertex, Vertex or Vector but is {type(other)}'
		return all([a > b for a, b in zip(self, other)])
		
	def __tuple__(self):
		return tuple([round(self.__getattribute__(slot)) for slot in self.__slots__])
	
	@classmethod
	def center_off(cls, *verticies):
		return cls(*[min(a) + (max(a) - min(a)) / 2 for a in zip(*[vert for vert in verticies])])
	
	@classmethod
	def random(cls, minmax):
		return cls([rng(*minmax) for item in cls.__slots__])
	
	@property
	def equalVect(self):
		return Vector2D
	
	@property
	def max_dist(self):
		return max([abs(self.__getattribute__(slot)) for slot in self.__slots__])
	
	def between(self, other, step=.5):
		assert isinstance(other, self.__class__), f'other must be of type VirtualVertex, Vertex or Vector but is {type(other)}'
		temp = other.relative_to(self) * step
		return self + temp
	
	def rotate(self, degree):
		vect = [self.x, self.y]
		degree = rad(degree)
		vect = (np.array(vect) @ np.array([cos(degree), sin(degree)]), np.array(vect) @ np.array([-sin(degree), cos(degree)]))
		self.x, self.y = vect
		return self
	
	def relative_to(self, other, output=0):
		assert isinstance(other, self.__class__), f'other must be of type VirtualVertex, Vertex or Vector but is {type(other)}'
		if output == 0:
			return self - other
		return self.equalVect(*[val for val in self - other])
	
	def distance(self, other):
		assert isinstance(other, self.__class__), f'other must be of type VirtualVertex, Vertex or Vector but is {type(other)}'
		return sum([axis ** 2 for axis in other - self]) ** .5
	
	def snap(self, other, dist: float = 2.5):
		return self.distance(other) <= dist



			
		
class Area:
	__slots__ = '_childs', '_center', '_size', 'name'
	def __init__(self, name:str, center:Vertex, size:Vertex):
		self.name = name
		self._childs = []
		self._center = center
		self._size = size
		
	@classmethod
	def _create_relative(cls, name:str, parent:Vertex, relation:Vertex):
		temp = parent * relation / 2
		return cls(name, temp, temp)
	
	def appendChild(self, name, relative_pos, relative_size):
		pos = self._center + self._size * relative_pos
		size = self._size * relative_size
		self._childs.append(self.__class__(f'{self.name}.{name}', pos, size))
		
	def move_to(self, pos:Vertex):
		self._center = pos
	@property
	def sides(self):
		tl, br = self._size + self._center, self._center - self._size
		return min([tl.x, br.x]), min([tl.y, br.y]), max([tl.x, br.x]), max([tl.y, br.y])
	@property
	def width(self):
		return self.sides[2] - self.sides[0]
	@width.setter
	def width(self, value):
		self._size = Vertex(value / 2, self.height)
	@property
	def height(self):
		return self.sides[3] - self.sides[1]
	@height.setter
	def height(self, value):
		self._size = Vertex(self.width, value / 2)
	@property
	def to_rect(self):
		return (self.sides[0], self.sides[1], self.width, self.height)
	@property
	def corners(self):
		'''
		Gets the corners as Vertex objects
		:return: top left, top right, bottom right, bottom left
		'''
		return Vertex(*self.sides[:2]), Vertex(self.sides[2], self.sides[1]), Vertex(self.sides[2], self.sides[3]), Vertex(self.sides[0], self.sides[3])
	
	def isInside(self, point:Vertex, offset:Vertex):
		if self.corners[0] + offset < point < self.corners[2] + offset:
			for child in self._childs:
				if child.isInside(point, self.corners[0]):
					return child
			return self
		return False


if __name__ == '__main__':
	test = Area('a', Vertex(50, 50), Vertex(50, 50))
	test.appendChild('b', Vertex(-.5, -.5), Vertex(.5, .5))
	test.appendChild('c', Vertex(.5, .5), Vertex(.5, .5))
	print(test.corners)
	print([child._center for child in test._childs])
	print(test.isInside(Vertex(51, 51), Vertex(0, 0)).name)
