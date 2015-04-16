#!/usr/bin/python
#-*- coding:utf-8 -*-

import configure
from utils.text2pickle import *
from utils.tokenFeature import *
from utils.word2vecFeature import *
from utils.ProcessTest import *
from utils.makeAnn import *
from utils.uniform import *
from scripts.m2scorer import evaluateIt

class ProcessPrep(object):
	"""docstring for ProcessPrep"""
	def __init__(self):
		super(ProcessPrep,self).__init__()

		self.trainConll = configure.fCorpusTrainConll
		self.ptrainConll = configure.fCorpusPickleTrainConll
		self.ptrainSentTree = configure.fCorpusPickleTrainSentTree
		self.ptrainSentence = configure.fCorpusPickleTrainSentence

		self.trainAnn = configure.fCorpusTrainAnn
		self.ptrainAnn = configure.fCorpusPickleTrainAnn

		self.testConll = configure.fCorpusTestConll
		self.ptestConll = configure.fCorpusPickleTestConll
		self.ptestSentTree = configure.fCorpusPickleTestSentTree
		self.ptestSentence = configure.fCorpusPickleTestSentence

		self.testAnn = configure.fCorpusTestAnn
		self.ptestAnn = configure.fCorpusPickleTestAnn

		self.word2vecWI = configure.fword2vecWI
		self.word2vecUWI = configure.fword2vecUWI
		self.word2vecVec = configure.fword2vecVec

		self.testM2 = configure.fCorpusTestM2
		self.perpM2 = configure.fCorpusTestPrepM2

		#From Here is different!    -----------------------   From Here is different! #

		self.trainToken = configure.fTrainTokenPrep
		self.testToken = configure.fTestTokenPrep

		self.traintestIW = configure.fTrainTestIWPrep
		self.traintestVec = configure.fTrainTestVecPrep

		self.validateVec =configure.fValidateVecPrep
		self.trainVec =configure.fTrainVecPrep
		self.testVec =configure.fTestVecPrep

		self.validateUVec =configure.fValidateUVecPrep
		self.trainUVec =configure.fTrainUVecPrep
		self.testUVec =configure.fTestUVecPrep

		self.CNNResult = configure.fCNNResultPrep
		self.CNNCorrectRes = configure.fCNNCorrectResPrep

		self.POSTAG = ["IN","TO"]
		self.ET = "Prep"

	def preprocessTrainTest(self):

		text2pickle_conll(self.trainConll,self.ptrainConll,self.ptrainSentTree,self.ptrainSentence)
		text2pickle_ann(self.trainAnn,self.ptrainAnn)

		text2pickle_conll(self.testConll,self.ptestConll,self.ptestSentTree,self.ptestSentence)
		text2pickle_ann(self.testAnn,self.ptestAnn)

	def getTokenFeature(self):
		TokenFeature(self.ptrainSentTree,self.ptrainConll,self.ptrainAnn,self.trainToken,True,self.POSTAG,self.ET)
		TokenFeature(self.ptestSentTree,self.ptestConll,self.ptestAnn,self.testToken,False,self.POSTAG,self.ET)

	def getVectorFeature(self):

		#execute only when the token is change!
		getIndex2Word(self.trainToken,self.testToken,self.word2vecWI,self.traintestIW)
		getTrainTestVec(self.traintestIW,self.word2vecVec,self.traintestVec)

		# 提取特征
		#VectorFeature(self.traintestVec,self.trainToken,self.validateVec,self.trainVec,True,False)
		#VectorFeature(self.traintestVec,self.testToken,self.validateVec,self.testVec,False,False)
		# 归一化操作
		#uniform(self.trainVec,self.trainUVec)
		#uniform(self.testVec,self.testUVec)
		#uniform(self.validateVec,self.validateUVec)

	def makeOutput(self):
		#ProcessTest("123",self.ptestSentence,self.DNNCorrectRes)
		ProcessTest(self.CNNResult,self.ptestSentence,self.CNNCorrectRes,self.ET)


	def makePrepM2(self):
		makeAnn(self.testM2,self.perpM2,tag="Prep")


	def evaluateRes(self):
		verbose = False 
		p,r,f1 = evaluateIt(self.CNNCorrectRes,self.perpM2,verbose)
		if not verbose:
			print "p:\t",p
			print "r:\t",r
			print "f:\t",f1

if __name__ == "__main__":
	Prep = ProcessPrep()
	#Prep.preprocessTrainTest()
	Prep.getTokenFeature()
	#Prep.getVectorFeature()
	#Prep.makeOutput()
	#Prep.makePrepM2()  # need to run one time
	#Prep.evaluateRes()
