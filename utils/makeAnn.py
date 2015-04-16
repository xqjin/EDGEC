#!/usr/bin/python

import re
import sys

def makeAnn(input1,output1,tag="ArtOrDet"):
	with open(input1,'r') as fr:
		clines = fr.readlines()
		fr.close()
	
	# is a new line?
	NL = False
	reval = list()
	for line in clines:
		if line.startswith("S") or (line.__contains__(tag) and line.startswith("A")) :
			reval.append(line)
			NL = False

		if line == "\n" and not NL:
			reval.append(line)
			NL = True
	else:
		with open(output1,'w') as fw:
			fw.writelines(reval)
			fw.close()

if __name__ == '__main__':
	pass