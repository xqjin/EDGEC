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
from Key2Value import PrepKV as PKV
from Key2Value import ArtKV as AKV
import nltk


default = "</s>"
CV = 5

def isword(word):
    word = word.strip().lower()
    for a in word:
        if  a not in 'abcdefghijklmnopqrstuvwxyz_':
            return False

    else:
        return True

def getToken(ifile,wordset):
    begin = 8

    with open(ifile) as fr:
        while True:
            line = fr.readline()[:-1]
            if not line:
                break

            line = line.strip().split()
            #['829', '1', '0', '17', '0', 'det', 'NULL', 'NULL', 'environment', 'that', 'can', 'sustain', 'their', 'lives', '.', 'NOTWORD']
            words = [ w.lower() for w in line[begin:] if w!="NOTWORD" and isword(w)]

            wordset |= set(words)


    return wordset

def getTrainTestToken(ifile1="train",ifile2="test"):
    wordset = set()
    getToken(ifile1,wordset)
    getToken(ifile2,wordset)
    global default
    wordset.add(default)
    return wordset

def getIndex2Word(ifile1="train",ifile2="test",ifile3="WI",ofile="traintestIW"):
    """
    ifile:the path of the word2vec's index file
    """
    wordset = getTrainTestToken(ifile1,ifile2)

    # load the word2index file
    with open(ifile3,'rb') as fr:
        WI = pickle.load(fr)
        fr.close()

    IW = dict()
    for word in wordset:
        index = WI.get(word,-1)
        IW.setdefault(index,set())
        IW[index].add(word)
    else:
        with open(ofile,'wb') as fw:
            pickle.dump(IW,fw)
            fw.close()

def getTrainTestVec(ifile1="traintestIW",ifile2="p2vec",ofile="traintestvec"):

    TT2V = dict()  # short for train-test-to-vec
    with open(ifile1,'rb') as fr:
        IW = pickle.load(fr)
        fr.close()

    for index,words in IW.items():
        if index == -1:continue

        with open(ifile2 %index,'rb') as fr:
            W2V = pickle.load(fr)
            fr.close()

        for word in words:
            TT2V[word] = W2V[word]

    else:
        with open(ofile,'wb') as fw:
            pickle.dump(TT2V,fw)
            fw.close()

def VectorFeature(ifile1="traintestvec",ifile2="train or test token file",ifile3="validation",ifile4="vectrain",istrain="False",isArt=True):
    # We only process this error this time !!!
    #[nid,pid,sid,posi,ti,sw,aw,words]

    # record the number of each catalogy!
    KVNum = dict()

    # load the train and test vector !
    W2V = pickle.load(open(ifile1,'rb'))

    global CV,default
    defaultVec = W2V[default]

    num = 1  # To record the number of instance in the train or file !!

    fr   = open(ifile2,'r')
    fw_t = open(ifile4,'w')
    if istrain:fw_v = open(ifile3,'w')

    # load the proper key-value dict from the KV 
    if isArt:
        KV = AKV
    else:
        KV = PKV
    
    while True:
        line = fr.readline()[:-1]
        if not line: break

        line = line.lower().split()

        temp = list()
        pre = line[:5]

        # get the source word and annotation word!
        SW  = KV.get(line[5],-1) 
        AW  = KV.get(line[6],-1)

        if SW==-1 or AW == -1 or isSkip(SW,AW):
            continue

        pre.append(SW)
        pre.append(AW)

        notWord = 0

        for l in line[7:]:
            tempVec= W2V.get(l,"NOTFOUND")
            if tempVec == "NOTFOUND":
                notWord += 1

            temp.append(W2V.get(l,defaultVec))      
        else:
            if notWord >=4:continue
            if (istrain and num%CV!=0) or (not istrain):
                # record the catalogy num!
                KVNum.setdefault(AW,0)
                KVNum[AW] += 1

                fw_t.write("\t".join(pre)+"\n")
                for t in temp:
                    fw_t.write("\t".join(t)+"\n")
            else:
                fw_v.write("\t".join(pre)+"\n")
                for t in temp:
                    fw_v.write("\t".join(t)+"\n")

            num += 1

    # output the information about the train or test file!!
    total = 0
    if istrain:
        print "Train File!"
    else:
        print "Test File!"
    for k,v in KVNum.items():
        print k,v
        total += v
    else:
        print "total Num is ",total

def isSkip(SW,AW):
    reval = False
    if SW == AW and AW=="0":
        temp = random.randint(1,8)
        if temp != 2:
            reval = True

    return reval
