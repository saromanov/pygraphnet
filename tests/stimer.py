import time

#http://www.huyng.com/posts/python-performance-analysis/
class Timer:
	def __init__(self):
		self.t = None
	def __start__(self):
		self.start = time.time()
	def __exit__(self):
		self.end = time.time()
		self.result = (self.end - self.start) * 1000
