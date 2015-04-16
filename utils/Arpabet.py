#!/usr/bin/python
#-*- coding:utf-8 -*-

import pickle
import re
import sys
import os
import configure
from stemming.porter2 import stem
from nltk.corpus import cmudict as cmu


vowels = ['AO','AA','IY','UW','EH','IH','UH','AH','AX','AE','EY','AY','OW','AW','OY','ER','AXR']
"""

说明：如果每次判断需要加载一边所有词对应的字典，时间将会大大的增加，不能将字典作为一个参数传递
"""
def isAorAn(word,wordsPro):
	#wordsPro = cmu.dict()  # 获取所有的单词和发音
	reval = wordsPro.get(word.lower())
	# Determine the first phoneme is vowels
	if reval:
		global vowels
		val = re.sub(r'\d','',reval[0][0])
		if val in vowels:
			return True
		else:
			return False
	
	return False


if __name__ == '__main__':
	print isAorAn('play')
	
