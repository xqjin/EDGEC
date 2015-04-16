#!/usr/bin/python
#-*- coding:utf-8 -*-

from stack import Stack

"""
	文件的作用，解析成分分析树，根据查找的成分，解析出成分的开头和结尾；
	findpre:  -1 for not found
	findnext: -2 for not found
"""

def findPreNext(tree,constituent="NP"):
	#tree="(ROOT(S(PP(0)(NP(1)(2)(3)))(4)(NP(5)(6)(7))(VP(8)(NP(NP(9)(10))(PP(11)(NP(ADJP(12)(13))(14)(15)(16))))(ADVP(17))(PP(18)(NP(NP(19)(20)(21))(22)(NP(23)(24)(25))(26)(NP(27)(28)(29)))))(30)))"
	#res = [('0', '4'), ('4', '8'), ('8', '17'), ('8', '11'), ('11', '17'), ('18', '30'), ('18', '22'), ('22', '26'), ('26', '30')]
	#The result of the example, I have found that the example is parsed correctly .....
	reval = set()	
	
	index = 0  # search from the beginning ....
	while index != -1: 
		index = tree.find(constituent,index)
		if index ==-1:
			break
		index_next = index+len(str(constituent))
		index_pre = index
	

		tpre  = findPre(tree,index_pre)
		tnext = findNext(tree,index_next)
		
		if tpre!=-1024 and tpre !=-1 and tnext!=len(tree)+1024 and tnext != -2:
			tri = (tpre,tnext)
			reval.add(tri)

		if index != -1:  # if any,find from the next position ....
			index += 1
	reval = sorted(reval,key=lambda k:int(k[0]))
	return reval

def findPre(tree,index):
	if index<0 or index > len(tree)-1:
		return -1024  # the word atfer the NP is not found !.....
	
	reval = 0
	numstr = "0123456789"
	while tree[index] not in numstr:
		if index > 0:
			index -= 1
		else:
			return -1
	
	else:
		nl = list()  # short num list contain the num 
		while tree[index] in numstr:
			nl.append(tree[index])	
			if index > 0:
				index -=1
		nl.reverse()
		for n in nl:
			reval = 10*reval + int(n)
		reval = str(reval)
		
	return reval

def findNext(tree,index):
	#max num
	max_num = len(tree)+1024
	tstack = Stack()
	if index>len(tree)-1 or index < 0:
		return max_num    # the word atfer the NP is not found !
	
	reval = 0
	while True:
		if tree[index] == "(":
			tstack.push("(")
		elif tree[index] == ")":
			if tstack.isEmpty():
				break   # find the number after tree[index]
			else:
				tstack.pop()
		else:
			pass

	
		if index < len(tree)-1: 
			index += 1
		else:
			return max_num 
	
	
	numstr = "0123456789"
	while tree[index] not in numstr:
		if index < len(tree)-1:
			index += 1
		else:
			return  -2 # denote the next word not found !	
	else:
		while tree[index] in numstr:
			reval = 10*reval + int(tree[index])
			if index < len(tree)-1:
				index += 1
	
	return str(reval)

if __name__ == "__main__":
	#ss = "(ROOT(S(NP(NP(0)(1)(2))(PP(3)(NP(4)(5)(6))))(VP(7)(PP(8)(NP(9)))(SBAR(10)(S(NP(11)(12))(VP(13)(VP(14)(PRT(15)))))))(16)))"
	#print findPreNext(ss)
	#(ROOT(NP(0)(1)(2)(3)))
	ss = "(ROOT(S(NP(0))(VP(1)(ADVP(2))(NP(NP(3)(4))(SBAR(WHNP(5))(S(VP(6)(ADJP(ADVP(7)(8))(9)(PP(10)(NP(NP(11)(12)(13))(PP(14)(NP(15)(16)(17)(18)))))))))))(19)))"
	#print findPreNext(ss)

