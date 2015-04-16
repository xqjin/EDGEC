#!/usr/bin/python
#-*- coding:utf-8 -*-
import random
import pickle
import re
import sys
import math
import os
import configure
import parseNumTree as pnt
from stemming.porter2 import stem
import nltk

"""
对数据进行归一化操作：
    归一化的结果为：均值为零，方差为1

输入数据的格式为：

infor
wordvec1
wordvec2
wordvec3
...

wordvecL

.
.
.


"""
def uniform(input1,output1,wordL=8,vectorL=50):
    mean = calculateMean(input1,wordL,vectorL)
    variance = calculateVariance(input1,mean,wordL,vectorL)


    fr = open(input1)
    fw = open(output1,"w")

    while True:
        info = fr.readline()

        if not info:
            break

        fw.write(info)

        for i in range(wordL):
            line = fr.readline()[:-1]
            line = line.split()

            fout =[0]*vectorL 

            for j in range(vectorL):
                fout[j] = (eval(line[j])-mean[i*vectorL+j])/variance[i*vectorL+j]   

            else:
                fout = [str(v) for v in fout]
                fw.write("\t".join(fout)+"\n")
    
    fr.close()
    fw.close()


def calculateMean(input1,wordL,vectorL):
    
    fr = open(input1)

    mean = [0.0]*wordL*vectorL

    number = 0

    while True:
        info = fr.readline()[:-1]

        if not info:
            fr.close()
            break

        for i in range(wordL):
            line = fr.readline()[:-1]
            line = line.split()

            for j in range(vectorL):
                mean[i*vectorL+j] += eval(line[j])
        else:
            number +=1

    mean = [v/number for v in mean]
    return mean 


def calculateVariance(input1,mean,wordL,vectorL):
    
    fr = open(input1)

    variance = [0.0]*wordL*vectorL

    number = 0

    while True:
        info = fr.readline()[:-1]

        if not info:
            fr.close()
            break

        for i in range(wordL):
            line = fr.readline()[:-1]
            line = line.split()

            for j in range(vectorL):
                variance[i*vectorL+j] += math.pow(eval(line[j])-mean[i*vectorL+j],2)
        else:
            number +=1

    variance = [math.sqrt(v/number) for v in variance]
    return variance 
