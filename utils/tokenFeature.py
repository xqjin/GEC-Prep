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
import nltk


"""
	这个文件的主要作用是：提取特征
	为了方便特征的拓展，特征采用 fnum 的形式	
	f1:  source word the word use
"""

def Feature(input1,input2,input3,output1,istrain):
	"""
		input1: numsentence
		input2: processed
		input3: annotation file
		output1: the output of the token feature file
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


	for nid,cnid in sorted(numsent.items(),key = lambda val:int(val[0])):	
		for pid,cpid in sorted(cnid.items(),key = lambda val:int(val[0])):
			for sid,csid in sorted(cpid.items(),key = lambda val:int(val[0])):

				sent = processed[nid][pid][sid]

				POSP = getPOSPositionInSent(sent,["IN","TO","VB"])

				ASW = getAnnotationSourceWord(ann,sent,nid,pid,sid,ET="Prep")

				if istrain:
					Positions = set(POSP + ASW.keys())
					Instances = TrainFeatureNUMInstance(sent,Positions,ASW)
				else:
					Positions = set(POSP)
					Instances = TestFeatureNUMInstance(sent,Positions)

				preIns = [nid,pid,sid]

				for Ins in Instances:
					insstr = preIns + Ins
					fw.write("\t".join(insstr)+"\n")
					

def TestFeatureNUMInstance(sent,positions,NUM=4):

	reval = list()

	positions = [ p for p in positions if p<len(sent)]

	tindex = dict()
	tindex[0] = [-4,-3,-2,-1,0,1,2,3]
	tindex[1] = [-4,-3,-2,-1,1,2,3,4]


	for posi in positions:
		temp = list()
		ti = 0

		if sent[str(posi)]["POS"].startswith("IN") or sent[str(posi)]["POS"].startswith("TO"):
			temp.append(sent[str(posi)]["TOKEN"])
			temp.append("NOTPREDICT")
			ti = 1
		else:
			temp.append("NULL")
			temp.append("NOTPREDICT")
		
		for i in tindex[ti]:
			token = "NOTWORD"
			index =	posi+i
			if index>=0 and index<len(sent):
				index = str(index)
				token = sent[index]['TOKEN']

			temp.append(token)
		else:
			temp = [str(posi),str(ti)]+temp
			reval.append(temp)

	return reval

def TrainFeatureNUMInstance(sent,positions,ASW,NUM=4):
	reval = list()

	# to avoid the index exceed the bound of the sent; for some wrong pos tagger !
	positions = [ p for p in positions if p<len(sent)]

	tindex = dict()
	tindex[0] = [-4,-3,-2,-1,0,1,2,3]
	tindex[1] = [-4,-3,-2,-1,1,2,3,4]

	for posi in positions:
		temp = list()
		ti = 0

		if posi in ASW:
			temp.append(ASW[posi]["SW"])
			temp.append(ASW[posi]["AW"])

			if ASW[posi]["SW"] != "NULL":
				ti = 1

		elif sent[str(posi)]["POS"].startswith("IN") or sent[str(posi)]["POS"].startswith("TO"):
			temp.append(sent[str(posi)]["TOKEN"])
			temp.append(sent[str(posi)]["TOKEN"])
			ti = 1
		else:
			temp.append("NULL")
			temp.append("NULL")



		for i in tindex[ti]:
			token = "NOTWORD"
			index =	posi+i
			if index>=0 and index<len(sent):
				index = str(index)
				token = sent[index]['TOKEN']

			temp.append(token)
		else:
			temp = [str(posi),str(ti)]+temp
			reval.append(temp)

	return reval


def getPOSPositionInSent(sent,poses):
	reIndex = list()

	for index in range(len(sent)):
		for pos in poses:
			if sent[str(index)]["POS"].startswith(pos):
				val = index
				if pos=="VB":val += 1
				reIndex.append(val)

	return reIndex


def isPREP(word):
	text = nltk.word_tokenize(word)
	text = nltk.pos_tag(text)

	reval = False
	for m,n in text:
		if n=="IN" or n=="TO":
			reval = True
	
	return reval


def getAnnotationSourceWord(ann,sent,nid,pid,sid,ET="Prep"):
	"""
	如果这个句子中没有介词错误直接放回,否则返回一个字典
	"""
	cor_list = ann.get(nid,dict()).get(pid,dict()).get(sid,dict()).get(ET,list())
	if not cor_list: return dict()  # not found the prep error in the sentence !

	SAW = dict()
	"""
	SAW[index][SW]
	SAW[index][AW]
	index: prep's position in the correct text !!
	"""

	for cor_dict in cor_list:
		start_token = int(cor_dict["start_token"])
		end_token   = int(cor_dict["end_token"])
		cor = cor_dict["cor"]

		if not cor: 
			#del case
			for index0 in range(start_token,end_token):
				if isPREP(sent[str(index0)]["TOKEN"]):
					break
					
			SAW.setdefault(index0,dict())
			SAW[index0]["AW"] = "NULL"
			SAW[index0]["SW"] = sent[str(index0)]["TOKEN"]

		else:
			text = nltk.word_tokenize(cor)	
			text = nltk.pos_tag(text)

			hasPrepInSent = False
			for index1 in range(start_token,end_token):
				if isPREP(sent[str(index1)]["TOKEN"]):
					hasPrepInSent = True
					break
				
			
			hasPrepInCor = False
			for index2 in range(len(text)):
				if isPREP(text[index2][0]):
					hasPrepInCor = True
					break

			index3 = start_token + index2
			# modify
			if hasPrepInSent and hasPrepInCor:
				SAW.setdefault(index1,dict())
				SAW[index1]["AW"] = text[index2][0]
				SAW[index1]["SW"] = sent[str(index1)]["TOKEN"]
				
				
			# del
			if not hasPrepInCor and hasPrepInSent:
				SAW.setdefault(index1,dict())
				SAW[index1]["AW"] = "NULL"
				SAW[index1]["SW"] = sent[str(index1)]["TOKEN"]

			# add
			if hasPrepInCor and not hasPrepInSent:
				SAW.setdefault(index3,dict())
				SAW[index3]["AW"] = text[index2][0]
				SAW[index3]["SW"] = "NULL"


	return SAW

if __name__ == "__main__":
	pass
