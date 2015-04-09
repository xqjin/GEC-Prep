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

		self.trainToken = configure.fTrainToken
		self.testToken = configure.fTestToken

		self.word2vecWI = configure.fword2vecWI
		self.word2vecUWI = configure.fword2vecUWI
		self.word2vecVec = configure.fword2vecVec

		self.traintestIW = configure.fTrainTestIW
		self.traintestVec = configure.fTrainTestVec


		self.validateVec =configure.fValidateVec
		self.trainVec =configure.fTrainVec
		self.testVec =configure.fTestVec

		self.validateUVec =configure.fValidateUVec
		self.trainUVec =configure.fTrainUVec
		self.testUVec =configure.fTestUVec

		self.DNNResult = configure.fDNNResult
		self.DNNCorrectRes = configure.fDNNCorrectRes

		self.testM2 = configure.fCorpusTestM2
		self.perpM2 = configure.fCorpusTestPrepM2


	def preprocessTrainTest(self):

		text2pickle_conll(self.trainConll,self.ptrainConll,self.ptrainSentTree,self.ptrainSentence)
		text2pickle_ann(self.trainAnn,self.ptrainAnn)

		text2pickle_conll(self.testConll,self.ptestConll,self.ptestSentTree,self.ptestSentence)
		text2pickle_ann(self.testAnn,self.ptestAnn)

	def getTokenFeature(self):
		Feature(self.ptrainSentTree,self.ptrainConll,self.ptrainAnn,self.trainToken,True)
		Feature(self.ptestSentTree,self.ptestConll,self.ptestAnn,self.testToken,False)

	def getVectorFeature(self):

		#execute only when the token is change!
		#getIndex2Word(self.trainToken,self.testToken,self.word2vecWI,self.traintestIW)
		#getTrainTestVec(self.traintestIW,self.word2vecVec,self.traintestVec)

		# 提取特征
		VectorFeature(self.traintestVec,self.trainToken,self.validateVec,self.trainVec,True)
		VectorFeature(self.traintestVec,self.testToken,self.validateVec,self.testVec,False)
		# 归一化操作
		#uniform(self.trainVec,self.trainUVec)
		#uniform(self.testVec,self.testUVec)
		#uniform(self.validateVec,self.validateUVec)

	def makeOutput(self):
		ProcessTest(self.DNNResult,self.ptestSentence,self.DNNCorrectRes)


	def makePrepM2(self):
		makeAnn(self.testM2,self.perpM2,tag="Prep")


	def evaluateRes(self):
		verbose = True 
		p,r,f1 = evaluateIt(self.DNNCorrectRes,self.perpM2,verbose)
		if not verbose:
			print "p:\t",p
			print "r:\t",r
			print "f:\t",f1

if __name__ == "__main__":
	Prep = ProcessPrep()
	#Prep.preprocessTrainTest()
	#Prep.getTokenFeature()
	Prep.getVectorFeature()
	#Prep.makeOutput()
	#Prep.makePrepM2()  # need to run one time
	#Prep.evaluateRes()
