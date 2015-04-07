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
from Key2Value import KV
import nltk


def Result2Dict(ifile):
	TestRes = dict()
	fr = open(ifile,'r')

	# read the result file to the dict object !
	"""
	nid , pid , sid , posi , ti : 这个位置是否是介词  
	"""
	while True:
		line = fr.readline()[:-1]
		if not line:
			break
		nid,pid,sid,posi,ti,sw,aw = line.split()

		TestRes.setdefault(nid,dict())
		TestRes[nid].setdefault(pid,dict())
		TestRes[nid][pid].setdefault(sid,dict())
		TestRes[nid][pid][sid].setdefault(posi,dict())
		TestRes[nid][pid][sid][posi]["ti"]=ti
		TestRes[nid][pid][sid][posi]["sw"]=sw
		TestRes[nid][pid][sid][posi]["aw"]=aw

	return TestRes

def ProcessTest(ifile1,ifile2,ofile1):
	"""
	ifile1 : result file
	ifile2 : test sentence in a list 
	ofile1 : output of the correct file
	"""

	TestRes = Result2Dict(ifile1)

	# load the sentence!
	fr = open(ifile2,'rb')
	TestSent = pickle.load(fr)
	fr.close()

	# write the result 
	fw = open(ofile1,'wb')

	# from tag to prep
	VK = dict()
	for k,v in KV.items():
		VK[v] = k

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

						if aw == sw:
							fout.append(csid[index])
						else:
							# add prep!
							if sw=="0" and aw!="0":
								assert(ti=="0")
								word = VK[aw]
								if index==0:
									word = word[:1].upper() + word[1:]
								fout.append(word)
								fout.append(csid[index])
							# del
							elif sw!="0" and aw=="0":
								assert(ti=="1")
							# modify
							else:
								assert(ti=="1")
								word = VK[aw]
								if index==0:
									word = word[:1].upper() + word[1:]
								fout.append(word)

				else:
					fw.write(" ".join(fout)+"\n")
