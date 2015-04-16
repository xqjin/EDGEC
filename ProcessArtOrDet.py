#!/usr/bin/python
#-*- coding:utf-8 -*-

import configure
from utils.text2pickle import *
from utils.tokenFeature import *
from utils.word2vecFeature import *
from utils.ProcessTest import *
from utils.makeAnn import makeAnn
from utils.uniform import *
from scripts.m2scorer import evaluateIt



class ProcessArtOrDet(object):
	"""docstring for ProcessArtOrDet"""
	def __init__(self):
		super(ProcessArtOrDet,self).__init__()

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

		# For word2vec the same to article and prep;
		self.word2vecWI = configure.fword2vecEWI
		self.word2vecVec = configure.fword2vecEVec


		#the annotation file is the same for article and prep
		self.testM2 = configure.fCorpusTestM2
		self.perpM2 = configure.fCorpusTestPrepM2
		self.artM2 = configure.fCorpusTestArtOrDetM2



		#From Here is different!    -----------------------   From Here is different! #
		self.trainToken = configure.fTrainTokenArt
		self.testToken = configure.fTestTokenArt

		self.traintestIW = configure.fTrainTestIWArt
		self.traintestVec = configure.fTrainTestVecArt

		self.validateVec =configure.fValidateVecArt
		self.trainVec =configure.fTrainVecArt
		self.testVec =configure.fTestVecArt

		self.validateUVec =configure.fValidateUVecArt
		self.trainUVec =configure.fTrainUVecArt
		self.testUVec =configure.fTestUVecArt

		self.CNNResult = configure.fCNNResultArt
		self.CNNCorrectRes = configure.fCNNCorrectResArt

		self.POSTAG = ["DT","NN"]
		self.ET = "ArtOrDet"


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
		#VectorFeature(self.traintestVec,self.trainToken,self.validateVec,self.trainVec,True,True)
		#VectorFeature(self.traintestVec,self.testToken,self.validateVec,self.testVec,False,True)
		
		# 归一化操作
		#uniform(self.trainVec,self.trainUVec)
		#uniform(self.testVec,self.testUVec)
		#uniform(self.validateVec,self.validateUVec)

	def makeOutput(self):
		#ProcessTest("123",self.ptestSentence,self.DNNCorrectRes)
		ProcessTest(self.CNNResult,self.ptestSentence,self.CNNCorrectRes,self.ET)


	def makeM2(self):
		print "Dot need to execute everytime!"
		makeAnn(self.testM2,self.artM2,tag="ArtOrDet")


	def evaluateRes(self):
		verbose = False 
		p,r,f1 = evaluateIt(self.CNNCorrectRes,self.artM2,verbose)
		if not verbose:
			print "p:\t",p
			print "r:\t",r
			print "f:\t",f1


if __name__ == "__main__":
	ArtOrDet = ProcessArtOrDet()
	#ArtOrDet.getTokenFeature()
	ArtOrDet.getVectorFeature()
	#ArtOrDet.makeOutput()
	#ArtOrDet.makeM2()  # need to run one time
	#ArtOrDet.evaluateRes()
