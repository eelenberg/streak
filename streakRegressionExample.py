import numpy as np
import pandas as pd
import sys
import csv
from scipy import io as sio
from scipy import stats
from sklearn import svm, preprocessing, base, datasets, feature_selection, linear_model, decomposition
from os import listdir
from os.path import isfile, join, exists
import argparse
from itertools import combinations,product,chain,combinations_with_replacement
from functools import reduce
import random
from streakFunctions import *
from joblib import Parallel, delayed
from joblib import load, dump
from time import time
import load_data
import streakFunctions as streak


def runStreamingLogisticMain(numTrain=800,K=80,numIt=10,regParam=0,streamTolerance=0.5,logFlag=1):
    
    if logFlag:
        resultsFilename = 'resultsReg.txt'
        if not exists(resultsFilename):
            #initialize results file with table
            fptr = open(resultsFilename,'w')
            fptr.write('data' + '\t' + 'Ntr' + '\t' + 'p' + '\t' + 'K' '\t' + 'alg' + '\t' + 'tol' '\t' + 
                'llhd' + '\t' + 'genScore' + '\t' + 'time' + '\t' + 'fevals' + '\n')
            fptr.close()


    # #select a dataset
    # saveString = 'phis'
    # dataString = 'phishL'
    # p = 68
    # pdim = p
    # # m = regParam
    # runVerbose = True
    # Ntotal = 2*numTrain
    # saveString = saveString + "_Ntr" + str(numTrain) + "_tol" + str(streamTolerance) + "_K" + str(K) + "_numIt" + str(numIt)
    
    saveString = 'phispair'
    dataString = 'phishP'
    p = 68
    pdim = (p**2 + 3*p)/2
    # m = regParam
    runVerbose = True
    Ntotal = 2*numTrain
    saveString = saveString + "_Ntr" + str(numTrain) + "_tol" + str(streamTolerance) + "_K" + str(K) + "_numIt" + str(numIt)
    
    
    if saveString[0:4] in ['rcv1','leuk','phis']:
        if runVerbose:
            llStrNorms = np.zeros((pdim+1,numIt))
        else:
            llStrNorms = np.zeros((K,numIt))
        scoreStrs = np.zeros_like(llStrNorms)
        timeStrs = np.zeros(numIt)
        evalStrs = np.zeros_like(timeStrs)

    #initialize arrays, print or store?
    if runVerbose:
        llRandNorms = np.zeros((pdim+1,numIt))
    else:
        llRandNorms = np.zeros((K,numIt))
    llLocalNorms = np.zeros_like(llRandNorms)
    scoreRands = np.zeros_like(llRandNorms)
    scoreLocals = np.zeros_like(llRandNorms)
    timeLocals = np.zeros(numIt)
    timeRands = np.zeros_like(timeLocals)
    evalLocals = np.zeros_like(timeLocals)
    evalRands = np.zeros_like(timeLocals)

    for it in np.arange(numIt):
        print "ITERATION: %d, N_training=%d, dim=%d, K=%d, tolerance=%f, regularizer=%f" % (it,numTrain,p,K,streamTolerance,regParam)
            
        #read dataset (downloaded from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html)            
        if saveString[0:4] == 'phis':
            X_tot,y_tot = load_data.loadFromLibsvm('phishing.txt',Ntotal,p)
            # random permutation of data samples
            nperm = np.random.permutation(Ntotal)
            X_train = X_tot[nperm[:numTrain]]
            y_train = y_tot[nperm[:numTrain]]
            X_test = X_tot[nperm[numTrain:]]
            y_test = y_tot[nperm[numTrain:]]
            print "loaded train and test sets (each size %d) from phishing dataset" % numTrain
            # X_train = X_train - np.mean(X_train,0)
            # X_test = X_test - np.mean(X_test,0)
            # y_train = 0.5*(y_train + 1)
            # y_test = 0.5*(y_test + 1)

        elif saveString[0:4] == 'leuk':
            X_train,y_train = load_data.loadFromLibsvm('leu',numTrain,p)
            X_test,y_test = load_data.loadFromLibsvm('leu.t',numTrain,p)
            print "loaded train and test sets (each size %d) from leukemia dataset" % numTrain
            X_train = X_train - np.mean(X_train,0)
            X_test = X_test - np.mean(X_test,0)
            y_train = 0.5*(y_train + 1)
            y_test = 0.5*(y_test + 1)

        elif saveString[0:4] == 'rcv1':
            X_train,y_train = load_data.loadFromLibsvm('rcv1_train.binary',numTrain,p)
            X_test,y_test = load_data.loadFromLibsvm('rcv1_test.binary',numTrain,p)
            print "loaded train and test sets (each size %d) from rcv1 binary dataset" % numTrain
            X_train = X_train - np.mean(X_train,0)
            X_test = X_test - np.mean(X_test,0)
            #map the y's that have from -1 to 0
            #map from 1 -1 to 1 0      1/2 (x+1)
            #random permutation every time? no, this is taken elsewhere
            # pperm = np.random.permutation(p)
            # X_train = X_train[:,pperm]
            # X_test = X_test[:,pperm]
            y_train = 0.5*(y_train + 1)
            y_test = 0.5*(y_test + 1)
            
        
        llNull = streak.getLogLikelihood(np.zeros(p),np.zeros(p),y_train,regParam)        
        
        if regParam == 0:
            regVal = 1.0/Cinf
        else:
            regVal = regParam
        
        if saveString[0:4] in ['rcv1','leuk','gise','sona','phis']:
            if saveString[0:7] in ['rcv1pai','leukpai','gisepai','sonapai','phispai']:
                print "Streaming Pairs"
                streamIter = chain(range(p),combinations_with_replacement(range(p),2))
                randomStreamIter = streak.random_permutation(streamIter)
            else:
                print "Streaming"
                randomStreamIter = streak.random_permutation(range(p))

            t1 = time()
            bsupS,llStr,scoreStrs[:,it],evalStrs[it] = streak.solveStreamingRegression(K,streamTolerance,
                X_train,y_train,X_test,y_test,llNull,regVal,randomStreamIter,'logistic') 
            t2 = time()
            timeStrs[it] = t2-t1
            print "streaming finished in %f" % (t2-t1)
            if logFlag:
                with open(resultsFilename,'a') as lptr:
                    lptr.write('\t'.join([dataString,str(numTrain),str(p),str(K),'stream',str(streamTolerance),
                        str(llStr[-1]),str(scoreStrs[-1,it]),str(timeStrs[it]),str(int(evalStrs[it]))]) + '\n')
            
            print "Random"
            t1 = time()
            if runVerbose:
                bsupR,llRand,scoreRands[1:(K+1),it],evalRands[it] = streak.solveRandomLogistic(K,X_train,y_train,X_test,y_test,
                    llNull,regVal,randomStreamIter)
                #TODO: pad with zeros?
                llRand = np.concatenate(( np.array([0.]),llRand,llRand[K-1]*np.ones((pdim-K)) ))
                scoreRands[(K+1):,it] = scoreRands[K,it]
            else:
                bsupR,llRand,scoreRands[:,it],evalRands[it] = streak.solveRandomLogistic(K,X_train,y_train,X_test,y_test,
                    llNull,regVal,randomStreamIter)
            t2 = time()
            timeRands[it] = t2-t1
            # print llRand,scoreRands
            print "random finished in %f" % (t2-t1)
            if logFlag:
                with open(resultsFilename,'a') as lptr:
                    #give random subset a tolerance of '2' so we can sort by this field later
                    lptr.write('\t'.join([dataString,str(numTrain),str(p),str(K),'random','2',str(llRand[-1]),
                        str(scoreRands[-1,it]),str(timeRands[it]),str(int(evalRands[it]))]) + '\n')
            
            if saveString[0:7] in ['rcv1pai','leukpai','gisepai','sonapai','phispai']:
                print "Local Search (and generating all pairwise features)"
            else:
                print "Local Search"
            t1 = time()
            bsupL,llLocal,scoreLocals[:,it],evalLocals[it] = streak.solveLocalSearchLogistic(K,X_train,y_train,X_test,y_test,
                llNull,regVal,randomStreamIter).getStats()
            t2 = time()
            timeLocals[it] = t2-t1
            if logFlag:
                with open(resultsFilename,'a') as lptr:
                    #give local search a tolerance of '-1' so we can sort by this field later
                    lptr.write('\t'.join([dataString,str(numTrain),str(p),str(K),'localsearch','-1',str(llLocal[-1]),
                        str(scoreLocals[-1,it]),str(timeLocals[it]),str(int(evalLocals[it]))]) + '\n')
            print "localsearch finished in %f" % (t2-t1)
        else: 
            raise ValueError("unsupported dataset")
        
        llStrNorms[:,it] = llStr
        llLocalNorms[:,it] = llLocal
        llRandNorms[:,it] = llRand
        


if __name__ == '__main__':
    argsList = []
    loggingFlag = 1
    # nVec = [50]
    nVec = [2000]
    tolVec = [0.75,0.1]
    kVec = [20]
    # kVec = [20,40,80]
    regVec = [0.0]
    for n in nVec:
        for reg in regVec:
            for k in kVec:
                for tol in tolVec:
                    argsList.append({
                         'numTrain':        n,
                         'K':               k,
                         'regParam':        reg,
                         'streamTolerance': tol,
                         'logFlag':         loggingFlag,
                         'numIt':           2})
    #run in parallel
    njobs = 1 #only 1 core
    # njobs = -3 #use all but 2 cores
    r = Parallel(n_jobs=njobs)(delayed(runStreamingLogisticMain)(**args) for args in argsList)
        
