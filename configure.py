#!/usr/bin/python
#-*- coding:utf-8 -*-

# 设置工程的目录
phome = '/home/xqjin/'

# 设置语料库的路径:训练语料,测试语料以及检验语料
pCorpusTrain = phome + 'Graduate/Corpus/NUCLE/train/'
pCorpusTest  = phome + 'Graduate/Corpus/NUCLE/test/'

pCorpusPickle = phome + 'Graduate/Corpus/NUCLE/pickle/'

pTokenFile = phome + 'Graduate/Corpus/Prep/tokenFeature/'
pVectorFile = phome + 'Graduate/Corpus/Prep/vectorFeature/'

pCorpusWord2vec = phome + "Graduate/Corpus/word2vec/"

pOutputPrep = phome + "Graduate/Output/Prep/"


# train *.conll 文件的位置,已经处理pickle结果存放的位置;
fCorpusTrainConll = pCorpusTrain + 'conll13st-preprocessed.conll'
fCorpusPickleTrainConll = pCorpusPickle + 'train.conll'
fCorpusPickleTrainSentTree = pCorpusPickle + 'train.senttree'
fCorpusPickleTrainSentence = pCorpusPickle + 'train.sentence'

# test *.conll 文件的位置,已经处理pickle结果存放的位置;
fCorpusTestConll = pCorpusTest + 'official-preprocessed.conll'
fCorpusPickleTestConll = pCorpusPickle + 'test.conll'
fCorpusPickleTestSentTree = pCorpusPickle + 'test.senttree'
fCorpusPickleTestSentence = pCorpusPickle + 'test.sentence'

# modify file of the prep!
fCorpusTestM2 = pCorpusTest + "official-preprocessed.m2"
fCorpusTestPrepM2 = pCorpusTest + "Perp.m2"

# train *.ann position
fCorpusTrainAnn = pCorpusTrain + 'conll13st-preprocessed.conll.ann'
fCorpusPickleTrainAnn = pCorpusPickle + 'train.ann'

# test *.ann position
fCorpusTestAnn = pCorpusTest + 'official-preprocessed.conll.ann'
fCorpusPickleTestAnn = pCorpusPickle + 'test.ann'


fTrainToken = pTokenFile + "trainTokenFeature.prep"
fTestToken  = pTokenFile + "testTokenFeature.prep"


fword2vecWI  = pCorpusWord2vec + "word2vec/index/word_index.dict"
fword2vecUWI = pCorpusWord2vec + "word2vec/uindex/word_index.dict"
fword2vecVec = pCorpusWord2vec + "word2vec/vector/word2vec%s.dict"
fword2vecUVec = pCorpusWord2vec + "word2vec/uvector/word2vec%s.dict"

fTrainTestIW = pVectorFile + "temp/traintestIW.dict"
fTrainTestVec = pVectorFile + "temp/traintestVec.dict"

fTrainVec = pVectorFile + "train.vec"
fTrainUVec = pVectorFile + "train.uvec"

fValidateVec = pVectorFile + "validate.vec"
fValidateUVec = pVectorFile + "validate.uvec"

fTestVec = pVectorFile + "test.vec"
fTestUVec = pVectorFile + "test.uvec"

fDNNModel = pOutputPrep + "/DNN/model/DNN-";
fDNNResult =pOutputPrep + "/DNN/Result/DNN.result";
fDNNCorrectRes =pOutputPrep + "/DNN/Result/DNNCorrect.result";














"""


PCorpusWord2vec = phome + "Graduate/Corpus/word2vec/"
# 设置程序交叉验证使用相应的训练和测试数据，未经处理过的
pCorpusCV = phome + 'Graduate/Corpus/CV/'
# 设置程序处理语料的结果
pCorpusText = phome + 'Graduate/Corpus/text/'
pOutputResult = phome + 'Graduate/Output/result/'
pOutputModel = phome + 'Graduate/Output/model/'


pCVTokenFeature = phome + 'Graduate/Output/CVFeature/token/'
pCVNumFeature = phome + 'Graduate/Output/CVFeature/num/'

pTokenFeature = phome + 'Graduate/Output/Feature/token/'
pNumFeature = phome + 'Graduate/Output/Feature/num/'





# 保存提取出的特征，特征用token的形式表示，没有经过编码
pOutputTrainToken = pTokenFeature + 'train/'
pOutputTestToken = pTokenFeature + 'test/'

pOutputTrainNum = pNumFeature + 'train/'
pOutputTestNum = pNumFeature + 'test/'

# save the token and token's index file 
fCorpusPickleWordDictSet = pCorpusPickle + "word.set"
fCorpusPickleWordIndexDict = pCorpusPickle + "wordindex"

# word2vec 聚类结果文件
fCorpusTextClassSorted = pCorpusText + 'classes.sorted.txt'

# 将word2vec 聚类结果处理成： word：cluster_label 的形式的文件所存放的位置；
fCorpusPickleWord2Clu = pCorpusPickle + 'word2clu'

fCorpusNumInfo = pCorpusPickle + "NumInfo"

fOutputResultCorrect = pOutputResult + 'train_res'

fOutputModelArt = pOutputModel + 'train.model'

fCorpusTestM2 = pCorpusTest + "official-preprocessed.m2"
fOutputResultDet = pOutputResult + 'test.det'




pCorpusText = "/home/xqjin/Code/CodeBlocks/NN/model/"

pCorpusTextModel = pCorpusText + "dbn.model"

pCorpusTextParaA = pCorpusText + "modelA"
pCorpusTextParaB = pCorpusText + "modelB"
pCorpusTextParaC = pCorpusText + "modelC"
pCorpusTextParaD = pCorpusText + "modelD"
pCorpusTextParaP = pCorpusText + "modelP"

pCorpusTextParaVBE = pCorpusText + "modelVBE"

fCorpusWord2vecAcc = PCorpusWord2vec + "acc_vec"





#Can del del  del del del del del del  #Can del del  del del del del del del 

pOutputTrainNumWord2vec = phome + 'Graduate/Output/train/num/word2vec/'
pOutputTestNumWord2vec = phome + 'Graduate/Output/test/num/word2vec/'

# 存放编码后的文件的位置；
fOutputTrainNum = pOutputTrainNum + "train"
fOutputTestNum = pOutputTestNum + "test"

fOutputTrainNumWord2vec = pOutputTrainNumWord2vec + "train"
fOutputTestNumWord2vec = pOutputTestNumWord2vec + "test"

fCorpusWord2vecWordIndex = PCorpusWord2vec + "word_index.dict"
fCorpusWord2vecWordVec = PCorpusWord2vec + "word2vec"
"""