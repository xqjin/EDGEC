#!/usr/bin/python
#-*- coding:utf-8 -*-
import random
import pickle
import re
import sys
import os
import configure
import parseNumTree as pnt
from stemming.porter2 import stem
from nltk.corpus import cmudict as cmu
from Key2Value import PrepKV as PKV
from Key2Value import ArtKV as AKV
from Arpabet import isAorAn
import nltk


"""
    这个文件的主要作用是：提取特征
    为了方便特征的拓展，特征采用 fnum 的形式    
    f1:  source word the word use
"""



def TokenFeature(input1,input2,input3,output1,istrain,POSTAG,ET):
    """
        input1: numsentence
        input2: processed
        input3: annotation file
        output1: the output of the token feature file
        istrain: trian file or not

    """ 

    #load the parse constituent  tree
    with open(input1,'rb') as fr:
        numsent = pickle.load(fr)
        fr.close()

    #load the processed  file 
    with open(input2,'rb') as fr:
        processed = pickle.load(fr)
        fr.close()

    #load the ann  file
    with open(input3,'rb') as fr:
        ann = pickle.load(fr)
        fr.close()

    fw = open(output1,'w')

    wordsPro = cmu.dict()


    for nid,cnid in sorted(numsent.items(),key = lambda val:int(val[0])):   
        for pid,cpid in sorted(cnid.items(),key = lambda val:int(val[0])):
            for sid,csid in sorted(cpid.items(),key = lambda val:int(val[0])):

                sent = processed[nid][pid][sid]
                
                #POSP = getPOSPositionInSent(sent,["IN","TO","VB"])

                # 得到含有标注数据的错误的位置
                if ET=="Prep":
                    WordPOS = isPREP
                    # 得到所有候选错误的位置
                    POSP = getPOSPositionInSent(sent,POSTAG,ET)
                elif ET=="ArtOrDet":
                    WordPOS = isDET
                    POSP = getParseTreePositionInSent(csid)
                else:  # Nn
                    WordPOS = isNN
                    POSP = getPOSPositionInSent(sent,POSTAG,ET)


                ASW = getAnnotationSourceWord(ann,sent,nid,pid,sid,WordPOS,ET)

                if istrain:
                    Positions = set(POSP + ASW.keys())
                    if ET=="Nn":
                        Instances = TrainTestFeatureNumInstanceNn(sent,Positions,ASW,NUM=4)
                    else:
                        Instances = TrainTestFeatureNUMInstancePrepArt(sent,Positions,ASW,ET,istrain,wordsPro)
                else:
                    Positions = set(POSP)
                    ASW = dict()
                    if ET=="Nn":
                        Instances = TrainTestFeatureNumInstanceNn(sent,Positions,ASW,NUM=4)
                    else:
                        Instances = TrainTestFeatureNUMInstancePrepArt(sent,Positions,ASW,ET,istrain,wordsPro)

                preIns = [nid,pid,sid]

                for Ins in Instances:
                    insstr = preIns + Ins
                    fw.write("\t".join(insstr)+"\n")
                    

def getParseTreePositionInSent(csid,constituent="NP"):
    """
    返回冠词应该出现的位置
    """
    parse_result = pnt.findPreNext(csid,constituent)
    reval = [int(m)+1 for m,n in parse_result]
    return reval


def TrainTestFeatureNumInstanceNn(sent,positions,ASW,NUM=4):
    reval = list()
    positions = [ p for p in positions if p<len(sent) and p>=0]

    # Waring this is different from the prep and artordet,since noun carry much information!
    tindex = dict()
    tindex[0] = [-4,-3,-2,-1,0,1,2,3,4]
    tindex[1] = [-4,-3,-2,-1,1,2,3,4]

    for posi in positions:
        temp = list()
        ti = 1

        centerWord = "itnlpsin"

        if posi in ASW.keys():
            temp.append( getWordPosTag( ASW[posi]["SW"] ) )
            temp.append( getWordPosTag( ASW[posi]["AW"] ) )
        else:

            temp.append(sent[str(posi)]["POS"])
            temp.append(sent[str(posi)]["POS"])
            assert sent[str(posi)]["POS"].startswith("NN")

        for i in tindex[ti]:
            token = "NOTWORD"
            index = posi+i
            if index>=0 and index<len(sent):
                index = str(index)
                token = sent[index]['TOKEN']

                if i==0: #process the center word
                    if isPlural(token):
                        centerWord = "itnlpplu"
                    token = centerWord

            temp.append(token)
        else:
            temp = [str(posi),str(ti)]+temp
            reval.append(temp)
    return reval


def TrainTestFeatureNUMInstancePrepArt(sent,positions,ASW,ET,istrain,wordsPro,NUM=4):
    assert ET=="Prep" or ET=="ArtOrDet"
    reval = list()

    # to avoid the index exceed the bound of the sent; for some wrong pos tagger !
    positions = [ p for p in positions if p<len(sent) and p >=0]

    tindex = dict()

    tindex[0] = [-4,-3,-2,-1,0,1,2,3]
    tindex[1] = [-4,-3,-2,-1,1,2,3,4]
    
    for posi in positions:

        temp = list()
        ti = 0

        if posi in ASW:
            temp.append(ASW[posi]["SW"])
            temp.append(ASW[posi]["AW"])

            if ASW[posi]["AW"] != "NULL":
                ti = 1

        else:

            flag0,flag1,flag2 = judgeError(sent,posi,ET)

            if (flag0 and flag1) or (not flag0 and flag2):
                temp.append(sent[str(posi)]["TOKEN"])
                temp.append(sent[str(posi)]["TOKEN"])
                ti = 1
            else:
                temp.append("NULL")
                temp.append("NULL")

        pretoken = "NULL"
        for i in tindex[ti]:
            token = "NOTWORD"
            posttoken = "NOTWORD"
            index = posi+i
            if index>=0 and index<len(sent):
                index = str(index)
                token = sent[index]['TOKEN']
                
                if pretoken.strip().lower() in ('a','an','the'):
                    posttoken = pretoken + "_" + token
                else:
                    posttoken = "null_" + token

                if ti==1 and i==1:
                    #pretoken = ASW[posi]["AW"]
                    pretoken = temp[2]
                else:
                    pretoken = token

            temp.append(posttoken)
        else:
            if istrain:
                temp = [str(posi),str(ti),str('aanthe')]+temp
                reval.append(temp)
            else:
                last4word = temp[-4]
                last4tag,last4word = last4word.split("_")

                ####To be finish ###
                dets = getDets(last4word,wordsPro)
                for det in dets:
                    temp[-4] = det + "_" + last4word
                    tempsave = [str(posi),str(ti),det]+temp
                    reval.append(tempsave)

    return ProcessRes(sent,reval,ET,True)

# get the possible det before the word !
def getDets(word,wordsPro):
    re = ['null','the']
    if not isPlural(word):
        if isAorAn(word,wordsPro):
            re.append('an')
        else:
            re.append('a')
    return re


def ProcessRes(sent,val,ET,istrain):
    # 针对训练样本采样，对训练样本不采样
    assert ET=="Prep" or ET=="ArtOrDet"
    ErrorsType = list()
    if ET == "Prep":
        ErrorsType = PKV.keys()
        WordPOS = isPREP
    else:
        ErrorsType = AKV.keys()
        WordPOS = isDET

    reval = list()
    for v in val:
        # v:  posi,ti,det,sw,aw,t1,t2,t3,t4,t5,t6,t7,t8
        posi = v[0]
        sw = v[3].lower()
        aw = v[4].lower()

        # delete all the instance we do not consider!
        if aw not in ErrorsType or sw not in ErrorsType:
            continue

        # delete the instance we do not consider !
        et = sent[posi]["TOKEN"].lower()
        if et not in ErrorsType and WordPOS(et):
            continue
        
        # rand delete a half instance
        #if aw==sw and sw=="null" and random.randint(0,1) and istrain:
        #    continue

        reval.append(v)

    return reval

def judgeError(sent,posi,ET):
    assert ET=="Prep" or ET=="ArtOrDet"
    flag0 = ET == "Prep"

    flag10 = sent[str(posi)]["POS"].startswith("IN") or sent[str(posi)]["POS"].startswith("TO")
    flag11 = getWordPosTag(sent[str(posi)]["TOKEN"]).startswith("IN") or getWordPosTag(sent[str(posi)]["TOKEN"]).startswith("TO")
    flag1 = flag10 and flag11

    flag20 = sent[str(posi)]["POS"].startswith("DT")
    flag21 = getWordPosTag(sent[str(posi)]["TOKEN"]).startswith("DT")
    flag2 = flag20 and flag21

    return flag0,flag1,flag2


def getPOSPositionInSent(sent,poses,ET):
    """
    得到候选错误的位置
    针对介词 ： 包含介词的位置 TO IN  以及动词 VB 的后面
    针对冠词 ： 包含冠词的位置  DT   名词 NN 前面

    返回可能出错的位置
    """
    reIndex = list()
    #all the possible values is : TO IN:VB   DET:NN
    for index in range(len(sent)):
        for pos in poses:
            # in case the corpus annotation is wrong ！
            if sent[str(index)]["POS"].startswith(pos) and getWordPosTag(sent[str(index)]["TOKEN"]).startswith(pos):
                val = index
                if pos=="VB" and ET=="Prep":val += 1
                if pos=="NN" and ET=="ArtOrDet":val -= 1
                # 确定val的值合法
                if val<0 and val>=len(sent):continue
                reIndex.append(val)

    return reIndex

def getWordPosTag(word):
    """
    返回word的词性
    """
    word = word.decode(encoding='ascii',errors='ignore')
    if not word:
        return "UNKNOWN"

    text = nltk.word_tokenize(word)
    text = nltk.pos_tag(text)

    return text[0][1]


def isNN(word):
    word = word.decode(encoding='ascii',errors='ignore')
    text = nltk.word_tokenize(word)
    text = nltk.pos_tag(text)

    reval = False
    for m,n in text:
        if n.startswith("NN"):
            reval = True
    
    return reval



def isPREP(word):
    word = word.decode(encoding='ascii',errors='ignore')
    text = nltk.word_tokenize(word)
    text = nltk.pos_tag(text)

    reval = False
    for m,n in text:
        if n=="IN" or n=="TO":
            reval = True
    
    return reval

def isDET(word):
    word = word.decode(encoding='ascii',errors='ignore')
    text = nltk.word_tokenize(word)
    text = nltk.pos_tag(text)

    reval = False
    for m,n in text:
        if n=="DT":
            reval = True
    return reval



def isPlural(word):
    word = word.decode(encoding='ascii',errors='ignore')
    text = nltk.word_tokenize(word)
    text = nltk.pos_tag(text)

    reval = False
    for m,n in text:
        if n=="NNPS" or n=="NNS":
            reval = True
    return reval

def getAnnotationSourceWord(ann,sent,nid,pid,sid,WordPOS,ET):
    """
    如果这个句子中没有介词错误直接放回,否则返回一个字典
    """
    cor_list = ann.get(nid,dict()).get(pid,dict()).get(sid,dict()).get(ET,list())
    if not cor_list: return dict()  # not found the error in the sentence !

    SAW = dict()
    """
    SAW[index][SW]
    SAW[index][AW]
    index: position in the correct text !!
    index: 这个位置应该是冠词或者介词，否则需要将当前位置的单词后移动；
    """

    for cor_dict in cor_list:
        start_token = int(cor_dict["start_token"])
        end_token   = int(cor_dict["end_token"])

        if end_token-start_token<=0:continue

        cor = cor_dict["cor"]

        if not cor.strip(): 
            #del case
            for index0 in range(start_token,end_token):
                if WordPOS(sent[str(index0)]["TOKEN"]):
                    break
                    
            SAW.setdefault(index0,dict())
            SAW[index0]["AW"] = "NULL"
            SAW[index0]["SW"] = sent[str(index0)]["TOKEN"]

        else:
            #process the correct text
            text = nltk.word_tokenize(cor)  
            text = nltk.pos_tag(text)

            hasInSent = False
            for index1 in range(start_token,end_token):
                if WordPOS(sent[str(index1)]["TOKEN"]):
                    hasInSent = True
                    break
                
            
            hasInCor = False
            for index2 in range(len(text)):
                if WordPOS(text[index2][0]):
                    hasInCor = True
                    break

            index3 = start_token + index2
            # modify
            if hasInSent and hasInCor:
                SAW.setdefault(index1,dict())
                SAW[index1]["AW"] = text[index2][0]
                SAW[index1]["SW"] = sent[str(index1)]["TOKEN"]
                
                
            # del
            if not hasInCor and hasInSent:
                SAW.setdefault(index1,dict())
                SAW[index1]["AW"] = "NULL"
                SAW[index1]["SW"] = sent[str(index1)]["TOKEN"]

            # add
            if hasInCor and not hasInSent:
                SAW.setdefault(index3,dict())
                SAW[index3]["AW"] = text[index2][0]
                SAW[index3]["SW"] = "NULL"


    return SAW

if __name__ == "__main__":
    pass
