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
from Key2Value import NnKV as NKV
from nltk.corpus import cmudict as cmu
from nltk.corpus import wordnet as wn
from Arpabet import isAorAn
import nltk


def Result2Dict(ifile):
    TestRes = dict()
    fr = open(ifile,'r')
    begin = 7

    # read the result file to the dict object !
    """
    nid , pid , sid , posi , ti : 这个位置是否是介词  
    """
    while True:
        line = fr.readline()[:-1]
        if not line:
            break

        line = line.split()
        nid,pid,sid,posi,ti,sw,aw = line[:begin]
        prob = line[begin:]

        TestRes.setdefault(nid,dict())
        TestRes[nid].setdefault(pid,dict())
        TestRes[nid][pid].setdefault(sid,dict())
        TestRes[nid][pid][sid].setdefault(posi,dict())
        TestRes[nid][pid][sid][posi]["ti"]=ti
        TestRes[nid][pid][sid][posi]["sw"]=sw
        TestRes[nid][pid][sid][posi]["aw"]=aw
        TestRes[nid][pid][sid][posi]["prob"]=prob

    return TestRes


def sin2plu(word):
    vowle = ['a','e','i','o','u']
    consonant = ['b','c','d','f','g','h','j','k','l','m','n','p','q','r','s','t','v','w','x','y','z'] 

    if len(word)>=2 and (word[-2] in vowle and word[-1]=='y'):
        word = word[:-1] + "ies"

    elif len(word)>=1 and word[-1]=='s':
        word += "es"

    else:
        word += "s"

    return word

def plu2sin(word):
    sin = wn.morphy(word)
    if not sin:
        sin = word
    return sin



def ProcessTest(ifile1,ifile2,ofile1,ET):
    if ET=="ArtOrDet" or ET == "Prep":
        ProcessTestArtPrep(ifile1,ifile2,ofile1,ET)
    else:
        ProcessTestNn(ifile1,ifile2,ofile1)

def ProcessTestNn(ifile1,ifile2,ofile1):

    """
    ifile1 : result file
    ifile2 : test sentence in a list 
    ofile1 : output of the correct file
    """

    TestRes = Result2Dict(ifile1)

    # load the sentence!
    TestSent = pickle.load(open(ifile2,'rb'))

    # write the result 
    fw = open(ofile1,'wb')

    # process the sentence!
    for nid,cnid in sorted(TestSent.items(),key=lambda v:int(v[0]) ):
        for pid,cpid in sorted(cnid.items(),key=lambda v:int(v[0]) ):
            for sid,csid in sorted(cpid.items(),key=lambda v:int(v[0]) ):
                fout = list()
                res = TestRes.get(nid,dict()).get(pid,dict()).get(sid,dict())
                reskeys = sorted(res.keys(),key=lambda val:int(val))
                reskeys = [int(v) for v in reskeys]

                for index in range(len(csid)):
                    if index not in reskeys:
                        fout.append(csid[index])
                    else:
                        aw = res[str(index)]["aw"]
                        sw = res[str(index)]["sw"]
                        ti = res[str(index)]["ti"]
                        prob = res[str(index)]["prob"]

                        if aw == sw:
                            fout.append(csid[index])
                        else:
                            if not isAccept(sw,aw,prob):
                                fout.append(csid[index])
                                continue

                            # singular to plural !
                            if sw=="0" and aw!="0":
                                word = sin2plu(csid[index])
                                if index==0:
                                    word = word[:1].upper() + word[1:]

                                fout.append(word)
                            # plural to singular
                            elif sw!="0" and aw=="0":
                                word = plu2sin(csid[index])
                                fout.append(word)

                else:
                    fw.write(" ".join(fout)+"\n")


def ProcessTestArtPrep(ifile1,ifile2,ofile1,ET):
    """
    ifile1 : result file
    ifile2 : test sentence in a list 
    ofile1 : output of the correct file
    """

    TestRes = Result2Dict(ifile1)

    # load the sentence!
    TestSent = pickle.load(open(ifile2,'rb'))

    # write the result 
    fw = open(ofile1,'wb')

    wordsPro = cmu.dict()
    
    # process the sentence!
    for nid,cnid in sorted(TestSent.items(),key=lambda v:int(v[0]) ):
        for pid,cpid in sorted(cnid.items(),key=lambda v:int(v[0]) ):
            for sid,csid in sorted(cpid.items(),key=lambda v:int(v[0]) ):
                fout = list()
                res = TestRes.get(nid,dict()).get(pid,dict()).get(sid,dict())
                reskeys = sorted(res.keys(),key=lambda val:int(val))
                reskeys = [int(v) for v in reskeys]

                for index in range(len(csid)):
                    if index not in reskeys:
                        fout.append(csid[index])
                    else:
                        aw = res[str(index)]["aw"]
                        sw = res[str(index)]["sw"]
                        ti = res[str(index)]["ti"]
                        prob = res[str(index)]["prob"]

                        if aw == sw:
                            fout.append(csid[index])
                        else:
                            if not isAccept(sw,aw,prob):
                                fout.append(csid[index])
                                continue

                            # add prep!
                            if sw=="0" and aw!="0":
                                word = Tag2WordArtPrep(ET,aw,csid[index],wordsPro)
                                if index==0:
                                    word = word[:1].upper() + word[1:]

                                #if the positon should not place a prep or art
                                if word!="NULL":
                                    fout.append(word)
                                fout.append(csid[index])
                            # del
                            elif sw!="0" and aw=="0":
                                assert(ti=="1")
                            # modify
                            else:
                                assert(ti=="1")
                                #word = VK[aw]
                                if index+1 >= len(csid): continue
                                word = Tag2WordArtPrep(ET,aw,csid[index+1],wordsPro)
                                if index==0:
                                    word = word[:1].upper() + word[1:]
                                if word!='NULL':
                                    fout.append(word)
                                else:
                                    fout.append(csid[index])

                else:
                    fw.write(" ".join(fout)+"\n")

def isAccept(sw,aw,prob):
    sw = int(sw)
    aw = int(aw)

    #return True 

    if(eval(prob[aw]) >=eval(prob[sw])):
        return True
    else:
        return False

    
def Tag2WordArtPrep(ET,tag,nextword,wordsPro):
    if ET=="ArtOrDet":
        KV = AKV
    elif ET=="Prep":
        KV = PKV
    else:
        KV = NKV

    # from tag to prep
    VK = dict()
    for k,v in KV.items():
        VK[v] = k

    word = VK[tag]

    if ET=="ArtOrDet" and (word in ('a','an') ):
        if not isAorAn(nextword,wordsPro):
            word = 'a'
        else:
            word = 'an'

        if isSpecifyPos(nextword,"NNS"):
            word = "NULL"
            
    return word

def isSpecifyPos(word,pos):
    word = word.decode(encoding='ascii',errors='ignore')
    text = nltk.word_tokenize(word)
    text = nltk.pos_tag(text)

    reval = False
    for m,n in text:
        if n==pos:
            reval = True

    return reval
