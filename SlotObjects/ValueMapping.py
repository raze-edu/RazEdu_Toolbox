from SlotObjects.Verticies import Vertex

class LinearMapping:
	__slots__ = 'a_start', 'a_end', 'b_start', 'b_end', 'relative_value'
	def __init__(self, a_start, a_end, b_start, b_end, rel_pos):
		self.a_start, self.b_start = a_start, b_start
		self.a_end, self.b_end = a_end, b_end
		self.relative_value = rel_pos
		
	@property
	def a_dist(self):
		return self.a_end - self.a_start
	
	@property
	def b_dist(self):
		return self.b_end - self.b_start
	
	@property
	def abs_a(self):
		return self.a_dist * self.relative_value + self.a_start

	@property
	def abs_b(self):
		return self.b_dist * self.relative_value + self.b_start

	def get_relative_value(self, related_to:int, value):
		self.relative_value = (value - [self.a_start, self.b_start][related_to]) / [self.a_dist, self.b_dist][related_to]


class ValueRange:
	__slots__ = 'start', 'end', 'value', 'iter'
	def __init__(self, start=0, end=100, value=0, iter=1):
		kwargs = {'start': start, 'end': end, 'value': value, 'iter': iter}
		#assert (end - start) % iter == 0, f'end {end} needs to be divisable by {iter}'
		[self.__setattribute__(slot, kwargs.get(slot)) for slot in self.__slots__]

	def __setattribute__(self, slot, val):
		if slot == 'value':
			val = self.start if val < self.start else self.end if val > self.end else val
			val = 0 if val == 0 else val // self.iter * self.iter
			if not (self.start <= val < self.end):
				val = self.start if self.start > val else self.end - 1
		super().__setattr__(slot, val)

	@property
	def val_pos(self):
		if self.value == self.start:
			value = self.start
		else:
			value = self.value / (self.end - self.start)
		return Vertex(value, 1)
	
	def __len__(self):
		return self.distance // self.iter

	@property
	def distance(self):
		return self.end - self.start
	@property
	def step_n(self):
		return 0 if self.value == 0 else self.value // self.step_size

	@property
	def rel_pos(self):
		print(f'rel pos = {self.value / self.distance}')
		return self.value / self.distance

	@rel_pos.setter
	def rel_pos(self, val):
		'''
  		uses the relative position to get the value
    		if in some gui a slider would be used 
      		along its line from x0 to x1, x1 - x0 divided by the X position of the slider
		the alignment to the stepsize is handled by the __setattr__  
  		'''
		self.value = self.distance * val
