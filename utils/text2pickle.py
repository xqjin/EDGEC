#!/usr/bin/python
#-*- coding:utf-8 -*-
import pickle
from lxml import etree
import re
import sys
import os
import configure


def text2pickle_conll(input1,output1,output2,output3):
	"""
		turn text of *.conll to pickle object

		将训练语料和测试语料中解析树的形式变成字符串的形式,方便后续程序的处理;

		(ROOT(S(PP(0)(NP(1)(2)(3)))(4)(NP(5)(6)(7))(VP(8)(NP(NP(9)(10))(PP(11)(NP(ADJP(12)(13))(14)(15)(16))))(ADVP(17))(PP(18)(NP(NP(19)(20)(21))(22)(NP(23)(24)(25))(26)(NP(27)(28)(29)))))(30)))

		the configure file contain the path of the file

	"""
	with open(input1,'r') as fr:
		content = fr.read()
		fr.close()

	pickle_save_dict = dict()

	#print content
	content_lines = content.split("\n")

	for line in content_lines:
		if line:
			tokens = line.split("\t")
			#为了检查异常，每行分割后含有九个元素，如果没有九个元素，显示异常
			if len(tokens) !=9:
				print "Error,please check it carefully!!!"
			
			#set NID
			NID = pickle_save_dict.setdefault(tokens[0],dict())
			#set PID
			PID = NID.setdefault(tokens[1],dict())
			#set SID
			SID = PID.setdefault(tokens[2],dict())
			#set TokenID
			TOKENID = SID.setdefault(tokens[3],dict())
			#set Token,pos,DPHead,DPREL,SYNT
			TOKENID['TOKEN'] = tokens[4]
			TOKENID['POS'] = tokens[5]
			TOKENID['DPHEAD'] = tokens[6]
			TOKENID['DPREL'] = tokens[7]
			TOKENID['SYNT'] = tokens[8]

	else:
		with open(output1,'wb') as fw:
			pickle.dump(pickle_save_dict,fw)
			fw.close()
		
		# process the number sentence !
		pickle_parse_tree = dict()
		for nid,nid_data in pickle_save_dict.items():
			pickle_parse_tree.setdefault(nid,dict())
			for pid,pid_data in pickle_save_dict[nid].items():
				pickle_parse_tree[nid].setdefault(pid,dict())
				for sid,sid_data in pickle_save_dict[nid][pid].items():
					parse = str()
					sid_data = sorted(sid_data.items(),key=lambda k:eval(k[0]))
					for tid,tid_data in sid_data:
						temp_str = tid_data['SYNT']
						#print sid,temp_str,tid
						temp_str = temp_str.replace("*","("+tid+")")
						#print sid,temp_str,tid
						parse+=temp_str
					else:
						pickle_parse_tree[nid][pid][sid]=parse
						#print "The whole parse tree is:"
						#print parse
		with open(output2,'wb') as fw:
			pickle.dump(pickle_parse_tree,fw)
			fw.close()



		# process the token sentence !
		pickle_sentence = dict()
		for nid,nid_data in pickle_save_dict.items():
			pickle_sentence.setdefault(nid,dict())
			for pid,pid_data in pickle_save_dict[nid].items():
				pickle_sentence[nid].setdefault(pid,dict())
				for sid,sid_data in pickle_save_dict[nid][pid].items():
					temp_sent = list()
					sid_data = sorted(sid_data.items(),key=lambda k:eval(k[0]))
					for tid,tid_data in sid_data:
						temp_token = tid_data['TOKEN']
						temp_sent.append(temp_token)
					else:
						pickle_sentence[nid][pid][sid]=temp_sent

		with open(output3,'wb') as fw:
			pickle.dump(pickle_sentence,fw)
			fw.close()

def text2pickle_ann(ifile,ofile):
	"""
	文件说明：这个文件的主要作用是将用单词作为索引的标注数据，转换成为比较好处理的方式；内存对象，可以直接判断错误的类别，纠正的文本；
	内存对象：nid，pid，sid，type  可以得到一个list，而每个list中，包含着错误的起始位置，错误如何纠正；

	<XML> 数据的处理

	ifile :  需要处理的文件
	ofile :  处理完成后的文件所在位置
	"""
	# read the annotation file from the file
	with open(ifile,'r') as fr:
		ann = fr.read()
		fr.close()
	
	re_val = dict()

	#ann  = ann.split(r"^$\n")	
	ann = re.split(r'\n\n',ann)
	#print len(ann)
	
	for subann in ann:		
		#tree = etree.parse(ifile)

		# we find when the text contain '&' symbol , it occcur an error !
		if not subann:
			continue

		(subann,num) = re.subn(r"&","",subann)
		root = etree.fromstring(subann)
		for item in root:
			if item.tag == "MISTAKE":
				nid=item.get("nid")
				pid=item.get("pid")
				sid=item.get("sid")
				start_token=item.get("start_token")
				end_token=item.get("end_token")
	
				v_type = str()
				v_cor = str()

				for subitem in item:
					if subitem.tag == "TYPE":
						v_type = subitem.text		 
					if subitem.tag == "CORRECTION":
						v_cor = subitem.text
						if not v_cor:
							v_cor = str()
				#print nid,pid,sid,start_token,end_token,v_cor,v_type 
				detail = re_val.setdefault(nid,dict()).setdefault(pid,dict()).setdefault(sid,dict()).setdefault(v_type,list())
				de = dict()
				de["start_token"] = start_token
				de["end_token"] = end_token
				de["cor"] = v_cor
				detail.append(de)
			else:
				continue
	else:
		with open(ofile,'wb') as fw:
			pickle.dump(re_val,fw)
			fw.close()
	#print "The file is saved correction !"

if __name__ == "__main__":
	input1 = configure.fCorpusTrainConll
	output1 = configure.fCorpusPickleTrainConll 
	output2 = configure.fCorpusPickleTrainSentTree
	text2pickle_conll(input1,output1,output2)

	ifile =  configure.fCorpusTrainAnn
	ofile =  configure.fCorpusPickleTrainAnn 
	text2pickle_ann(ifile,ofile)


	input1 = configure.fCorpusTestConll
	output1 = configure.fCorpusPickleTestConll 
	output2 = configure.fCorpusPickleTestSentTree
	text2pickle_conll(input1,output1,output2)

	ifile =  configure.fCorpusTestAnn
	ofile =  configure.fCorpusPickleTestAnn 
	text2pickle_ann(ifile,ofile)


	input1 = configure.fCorpusTrialConll
	output1 = configure.fCorpusPickleTrialConll 
	output2 = configure.fCorpusPickleTrialSentTree
	text2pickle_conll(input1,output1,output2)

	ifile =  configure.fCorpusTrialAnn
	ofile =  configure.fCorpusPickleTrialAnn 
	text2pickle_ann(ifile,ofile)
