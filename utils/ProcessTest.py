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
from nltk.corpus import cmudict as cmu
from Arpabet import isAorAn
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

def ProcessTest(ifile1,ifile2,ofile1,isArt):
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

                        if aw == sw:
                            fout.append(csid[index])
                        else:
                            # add prep!
                            if sw=="0" and aw!="0":
                                #assert(ti=="0")
                                #word = VK[aw]
                                word = Tag2Word(isArt,aw,csid[index],wordsPro)
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
                                word = Tag2Word(isArt,aw,csid[index+1],wordsPro)
                                if index==0:
                                    word = word[:1].upper() + word[1:]
                                if word!='NULL':
                                    fout.append(word)
                                else:
                                    fout.append(csid[index])

                else:
                    fw.write(" ".join(fout)+"\n")



def Tag2Word(isArt,tag,nextword,wordsPro):
    if isArt:
        KV = AKV
    else:
        KV = PKV

    # from tag to prep
    VK = dict()
    for k,v in KV.items():
        VK[v] = k

    word = VK[tag]

    if isArt and (word in ('a','an') ):
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
