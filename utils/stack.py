#!/usr/bin/python
#-*- coding:utf-8 -*-

class Stack(object):
	def __init__(self):
		super(Stack,self).__init__()
		self.items = list()

	def isEmpty(self):
		flag = False
		if len(self.items)==0:
			flag = True

		return flag

	def push(self,item):
		self.items.append(item)
		
	def pop(self):
		self.items.pop()			
	
	def peek(self):
		if len(self.items) != 0:
			return self.items[-1]

	def size(self):
		return len(self.itmes)

