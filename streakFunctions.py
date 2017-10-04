import numpy as np
import pandas as pd
import sys
import csv
from scipy import io as sio
from scipy import stats
from sklearn import svm, preprocessing, base, datasets, feature_selection, linear_model, decomposition
from sklearn.metrics import roc_curve,auc,classification_report,f1_score,accuracy_score,roc_auc_score
from os import listdir
from os.path import isfile, join, exists
import argparse
from itertools import combinations,product,chain,combinations_with_replacement
from functools import reduce
import random

'''
#functions to implement STREAK for regression and interpretability experiments
'''

#1/C is the regularizer, so make C large
Cinf = 1e10

def getLogLikelihood(X,b,y,C):
    #not rescaled by n = X.shape[0]
    dp = X.dot(b)
    return float(np.sum(np.multiply(y[:,np.newaxis],dp) - np.logaddexp(0,dp),0) - float(C)/2.0*b.T.dot(b))
    
def getLogLikelihoodLinear(X,b,y,alp):
    #not rescaled by n = X.shape[0]
    dp = X.dot(b)
    # return -1*np.sum(np.log(dp - y)) - float(alp)*b.T.dot(b)
    # return -1*np.sum((dp - y)**2) - float(alp)*b.T.dot(b)
    return np.array([[-1*np.sum((dp - y)**2) - float(alp)*b.T.dot(b)]])

def getLogGradient(X,b,y,C):
    #wrt points, not the support beta
    edp = np.exp(X.dot(b))
    return y - np.divide(edp,1 + edp)[:,0]

def getLogGradientLinear(X,b,y,alp):
    #wrt points, not support beta
    dp = X.dot(b)
    return y - dp

def random_permutation(iterable, r=None):
    #Random selection from itertools.permutations(iterable, r)
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(random.sample(pool, r))

def getFeature(X,inds):
    if X is None:
        return None
    #return either a feature (column) or a product of 2 features
    if isinstance(inds, int):
        return X[:,inds]
    elif isinstance(inds,tuple):
        return X[:,inds[0]]*X[:,inds[1]]

def padWithValueAt(l,vind):
    '''
    leave l unchanged up to index vind, then complete the list by padding with the value at l[vind]
    '''
    lSize = len(l)
    return l[:vind]+[l[vind]]*(lSize-vind)

def solveExhaustiveLogistic(F,y,Ftest,ytest,K,ridgeReg):
    '''
    return the best K-subset by exhaustive search
    '''
    sel = np.zeros(F.shape[1])
    selML = -np.inf
    regr = linear_model.LogisticRegression(penalty='l2',solver='lbfgs',fit_intercept=False,C=1.0/ridgeReg)
    for ksub in combinations(np.arange(F.shape[1]),K):
        Fslice = F[:,ksub]
        regr.fit(Fslice, y)
        tmp1 = getLogLikelihood(Fslice,regr.coef_.T,y,ridgeReg)
        if tmp1 > selML:
            selML = tmp1
            kbest = ksub
    Fslice = F[:,kbest]
    regr.fit(Fslice, y)
    selS = regr.score(Ftest[:,kbest],ytest)
    return np.array(kbest),selML,selS
    
    
class StreamingGreedyVerbose(object):
    '''
    Implementation of the STREAK algorithm
    Output is verbose, meaning that getStats() returns stats from the entire duration of the algorithm
    '''
    def __init__(self,K,ytrain,ytest,thresh,fnorm,ridgeReg,regressionType='logistic',p=0):
        #initialize parameters
        self.K = K
        self.ytrain = ytrain
        self.ytest = ytest
        self.thresh = thresh
        self.fnorm=fnorm
        self.ridgeReg = ridgeReg
        self.regressionType = regressionType
        #initialize state
        self.XcurrentTrain = np.empty((ytrain.shape[0],0))
        if ytest is not None:
            self.XcurrentTest = np.empty((ytest.shape[0],0))
        self.numEvaluations = 0
        self.llPrev = 0
        self.ksize = 0
        self.bestfeats = [0]*K
        if self.regressionType in ['logistic']:
            self.regr = linear_model.LogisticRegression(penalty='l2',solver='lbfgs', \
            fit_intercept=False,C=1.0/self.ridgeReg)
            self.getLikelihood = lambda (X,b): getLogLikelihood(X, b, self.ytrain, self.ridgeReg) - self.fnorm
        elif self.regressionType in ['linear','ridge','Ridge']:
            self.regr = linear_model.Ridge(alpha=0, fit_intercept=True)
            self.getLikelihood = lambda (X,b): getLogLikelihoodLinear(X, b, self.ytrain, self.ridgeReg) - self.fnorm
        else:
            raise ValueError('Regression Type Currently Unsupported')
        if p!= 0:
            statsDimension = p+1
        else:
            statsDimension = K
        self.numSeen = 0
        self.score = [0]*statsDimension
        self.llBest = [0]*statsDimension
        self.numAddLocal = K
        self.localsize = 0

    def update(self,xtrain,xtest=None,ind=-1):
        #input a single feature (training and test) and its corresponding index (not entire data matrix)
        self.numSeen += 1
        if self.ksize >= self.K:
            #if already selected K then just use previous scores
            self.score[self.numSeen] = self.score[self.numSeen-1]
            self.llBest[self.numSeen] = self.llBest[self.numSeen-1]
            return
        Fslice = np.hstack((self.XcurrentTrain,xtrain.reshape((self.ytrain.shape[0],1))))
        self.regr.fit(Fslice, self.ytrain)
        self.numEvaluations += 1
        tmpBest = self.getLikelihood((Fslice,self.regr.coef_.T))
        if (tmpBest - self.llPrev >= self.thresh/self.K) and (self.ksize<self.K): 
            #if marginal gain greater than tau/K,
            #add this feature and update scores
            self.bestfeats[self.ksize] = ind
            self.llPrev = tmpBest
            self.llBest[self.numSeen] = tmpBest
            self.XcurrentTrain = Fslice
            if xtest is not None:
                FsliceTest = np.hstack((self.XcurrentTest,xtest.reshape((self.ytest.shape[0],1))))
                self.score[self.numSeen] = self.regr.score(FsliceTest,self.ytest)
                self.XcurrentTest = FsliceTest
            self.ksize += 1
            self.numAddLocal -= 1
        else:
            #don't add this feature, use previous scores
            self.score[self.numSeen] = self.score[self.numSeen-1]
            self.llBest[self.numSeen] = self.llBest[self.numSeen-1]
        return

    def updatePastK(self,xtrain,xtest=None,ind=-1):
        #if already went through this instance, add features from other instances via local search
        #increment localsize when a new feature is added
        self.numSeen += 1
        kindex = self.ksize + self.localsize
        if kindex < self.K:
            #add current feature and return
            Fslice = np.hstack((self.XcurrentTrain,xtrain.reshape((self.ytrain.shape[0],1))))
            self.regr.fit(Fslice, self.ytrain)
            self.numEvaluations += 1
            self.llPrev = self.getLikelihood((Fslice,self.regr.coef_.T))
            self.bestfeats[kindex] = ind
            self.llBest[self.numSeen] = self.llPrev
            self.XcurrentTrain = Fslice
            if xtest is not None:
                FsliceTest = np.hstack((self.XcurrentTest,xtest.reshape((self.ytest.shape[0],1))))
                self.score[self.numSeen] = self.regr.score(FsliceTest,self.ytest)
                self.XcurrentTest = FsliceTest
            self.localsize += 1
            return
        else:
            #check all possibilities of replacing with current element and add if reaplacing an element
            #will increase over llPrev and it is the best replacement
            tmpFits = [0]*self.numAddLocal
            tmpScores = [0]*self.numAddLocal
            Fslices = [np.zeros_like(self.XcurrentTrain)]*self.numAddLocal
            if xtest is not None:
                FslicesTest = [np.zeros_like(self.XcurrentTest)]*self.numAddLocal
            for ktest in range(self.numAddLocal):
                fkeep = range(self.ksize+ktest) + range(self.ksize+ktest+1,self.K)
                Fslices[ktest] = np.hstack((self.XcurrentTrain[:,fkeep],xtrain.reshape((self.ytrain.shape[0],1))))
                self.regr.fit(Fslices[ktest], self.ytrain)
                self.numEvaluations += 1
                tmpFits[ktest] = self.getLikelihood((Fslices[ktest],self.regr.coef_.T))
                if xtest is not None:
                    FslicesTest[ktest] = np.hstack((self.XcurrentTest[:,fkeep],xtest.reshape((self.ytest.shape[0],1))))
                    tmpScores[ktest] = self.regr.score(FslicesTest[ktest],self.ytest)
            #find the best element to replace and see if removing it improves over llPrev
            bestFitIndex = np.argmax(tmpFits)
            if (tmpFits[bestFitIndex] - self.llPrev >= 0):
                self.bestfeats[self.ksize+bestFitIndex] = ind
                self.llPrev = tmpFits[bestFitIndex]
                self.llBest[self.numSeen] = tmpFits[bestFitIndex]
                self.XcurrentTrain = Fslices[bestFitIndex]
                if xtest is not None:
                    self.XcurrentTest = FslicesTest[bestFitIndex]
                    self.score[self.numSeen] = tmpScores[bestFitIndex]
            else:
                self.llBest[self.numSeen] = self.llBest[self.numSeen-1]
                if xtest is not None:
                    self.score[self.numSeen] = self.score[self.numSeen-1]
        return

    def getStats(self):
        #pad from numSeen to pdim with the value at numSeen
        return self.bestfeats,padWithValueAt(self.llBest,self.numSeen),padWithValueAt(self.score,self.numSeen),self.numEvaluations

    def isFull(self):
        return (self.ksize >= self.K)

class StreamingLocalSearchVerbose(object):
    '''
    Implementation of the local search algorithm
    Output is verbose, meaning that getStats() returns stats from the entire duration of the algorithm
    '''
    def __init__(self,K,ytrain,ytest,fnorm,ridgeReg,regressionType='logistic',p=0):
        #initialize parameters
        self.K = K
        self.ytrain = ytrain
        self.ytest = ytest
        self.fnorm=fnorm
        self.ridgeReg = ridgeReg
        self.regressionType = regressionType
        #initialize state
        self.XcurrentTrain = np.empty((ytrain.shape[0],0))
        self.XcurrentTest = np.empty((ytest.shape[0],0))
        self.numEvaluations = 0
        self.llPrev = 0
        self.ksize = 0
        self.bestfeats = [0]*K
        if self.regressionType in ['logistic']:
            self.regr = linear_model.LogisticRegression(penalty='l2',solver='lbfgs', \
            fit_intercept=False,C=1.0/self.ridgeReg)
            self.getLikelihood = lambda (X,b): getLogLikelihood(X, b, self.ytrain, self.ridgeReg) - self.fnorm
        elif self.regressionType in ['linear','ridge','Ridge']:
            self.regr = linear_model.Ridge(alpha=0, fit_intercept=True)
            self.getLikelihood = lambda (X,b): getLogLikelihoodLinear(X, b, self.ytrain, self.ridgeReg) - self.fnorm
        if p!= 0:
            statsDimension = p+1
        else:
            statsDimension = K
        self.numSeen = 0
        self.score = [0]*statsDimension
        self.llBest = [0]*statsDimension

    def update(self,xtrain,xtest,ind=-1):
        #pass features
        self.numSeen += 1
        if self.ksize < self.K:
            #add it and return
            # print "ksize=%d, adding index %d" % (self.ksize,ind)
            Fslice = np.hstack((self.XcurrentTrain,xtrain.reshape((self.ytrain.shape[0],1))))
            FsliceTest = np.hstack((self.XcurrentTest,xtest.reshape((self.ytest.shape[0],1))))
            self.regr.fit(Fslice, self.ytrain)
            self.numEvaluations += 1
            self.llPrev = self.getLikelihood((Fslice,self.regr.coef_.T))
            self.bestfeats[self.ksize] = ind
            self.score[self.numSeen] = self.regr.score(FsliceTest,self.ytest)
            self.llBest[self.numSeen] = self.llPrev
            self.XcurrentTrain = Fslice
            self.XcurrentTest = FsliceTest
            self.ksize += 1
            return
        else:
            #check all possibilities and add if removing an element
            #and replacing will increase over llPrev and it is the largest
            tmpFits = [0]*self.K
            tmpScores = [0]*self.K
            Fslices = [np.zeros_like(self.XcurrentTrain)]*self.K
            FslicesTest = [np.zeros_like(self.XcurrentTest)]*self.K
            for ktest in range(self.K):
                # fkeep = self.bestfeats[:ktest,(ktest+1):self.K]
                fkeep = range(ktest) + range(ktest+1,self.K)
                # print fkeep
                Fslices[ktest] = np.hstack((self.XcurrentTrain[:,fkeep],xtrain.reshape((self.ytrain.shape[0],1))))
                self.regr.fit(Fslices[ktest], self.ytrain)
                self.numEvaluations += 1
                tmpFits[ktest] = self.getLikelihood((Fslices[ktest],self.regr.coef_.T))
                FslicesTest[ktest] = np.hstack((self.XcurrentTest[:,fkeep],xtest.reshape((self.ytest.shape[0],1))))
                tmpScores[ktest] = self.regr.score(FslicesTest[ktest],self.ytest)
            #find the best element to replace and see if removing it improves the model
            bestFitIndex = np.argmax(tmpFits)
            if (tmpFits[bestFitIndex] - self.llPrev >= 0):
                self.bestfeats[bestFitIndex] = ind
                self.llPrev = tmpFits[bestFitIndex]
                self.score[self.numSeen] = tmpScores[bestFitIndex]
                self.llBest[self.numSeen] = tmpFits[bestFitIndex]
                self.XcurrentTrain = Fslices[bestFitIndex]
                self.XcurrentTest = FslicesTest[bestFitIndex]
            else:
                self.score[self.numSeen] = self.score[self.numSeen-1]
                self.llBest[self.numSeen] = self.llBest[self.numSeen-1]
                
        return

    def getStats(self):
        return self.bestfeats,padWithValueAt(self.llBest,self.numSeen),padWithValueAt(self.score,self.numSeen),self.numEvaluations


class StreamingGreedyML(object):
    '''
    #lightweight implementation of STREAK, specifically for maximizing classification likelihood 
    for a particular label of a black box model
    optimizing over a sparse interpretation (e.g. sparse set of segments/superpixels)
    no evaluation on test set, only output final stats
    '''
    def __init__(self,K,indexed_image,thresh,classifier_fn,ytrain=1,classInd=0,fnorm=0,p=0):
        #initialize parameters
        self.K = K
        self.image = indexed_image
        self.ytrain = ytrain
        self.thresh = thresh
        self.classInd = classInd
        self.fnorm=fnorm
        self.classifier_fn = classifier_fn
        #initialize state
        self.XcurrentTrain = indexed_image.ref_image
        self.numEvaluations = 0
        self.llPrev = 0
        self.ksize = 0
        self.bestfeats = [0]*K
        self.numAddLocal = K
        self.localsize = 0
    
    def getLikelihood(self,x):
        return self.classifier_fn(np.expand_dims(x,axis=0))[:,self.classInd] - self.fnorm

    def update(self,ind):
        if self.ksize >= self.K:
            return
        Fslice = self.image.add_segments(self.bestfeats[:self.ksize] + [ind])
        # print self.llPrev,self.bestfeats[:self.ksize]
        tmpBest = self.getLikelihood(Fslice)
        self.numEvaluations += 1
        if (tmpBest - self.llPrev >= self.thresh/self.K) and (self.ksize<self.K): #if marginal gain greater than tau/K
            self.bestfeats[self.ksize] = ind
            self.llPrev = tmpBest
            self.XcurrentTrain = Fslice
            # print "size %d, new ll: %f" % (self.ksize,tmpBest)
            self.ksize += 1
            self.numAddLocal -= 1
        return

    def updatePastK(self,ind):
        kindex = self.ksize + self.localsize
        if kindex < self.K:
            #add new index and return
            Fslice = self.image.add_segments(self.bestfeats[:kindex]+[ind])
            self.llPrev = self.getLikelihood(Fslice)
            self.bestfeats[kindex] = ind
            self.XcurrentTrain = Fslice
            self.numEvaluations += 1
            self.localsize += 1
            return
        else:
            #check all possibilities and add if removing an element
            #and replacing will increase over llPrev and it is the largest
            tmpFits = [0]*self.numAddLocal
            Fslices = [np.zeros_like(self.XcurrentTrain)]*self.numAddLocal
            for ktest in range(self.numAddLocal):
                fkeep = range(self.ksize+ktest) + range(self.ksize+ktest+1,self.K)
                # print fkeep
                Fslices[ktest] = self.image.add_segments([self.bestfeats[f] for f in fkeep] + [ind])
                self.numEvaluations += 1
                tmpFits[ktest] = self.getLikelihood(Fslices[ktest])
            #find the best element to replace and see if removing it improves the model
            bestFitIndex = np.argmax(tmpFits)
            if (tmpFits[bestFitIndex] - self.llPrev >= 0):
                self.bestfeats[self.ksize+bestFitIndex] = ind
                self.llPrev = tmpFits[bestFitIndex]
                # print "size %d, new ll: %f" % (self.ksize,self.llPrev)
        return

    def getStats(self):
        return self.bestfeats[:(self.ksize+self.localsize)],self.llPrev,self.numEvaluations
        
    def isFull(self):
        return (self.ksize >= self.K)


def updateInstances(I,m,k,ep,newItem):    
    #perform geometric search between lowerLimit and upperLimit
    #if m=0 then lowerLimit=upperLimit and no instance created
    lowerLimit = (1.-ep)*m/(9.*k**2) + 1./Cinf
    upperLimit = m*k + 1./Cinf
    #take log base 1-ep, check for sign flips by ensuring min and max
    logLowerLimit = np.minimum(np.log(lowerLimit)/np.log(1.-ep),np.log(upperLimit)/np.log(1.-ep))
    logUpperLimit = np.maximum(np.log(lowerLimit)/np.log(1.-ep),np.log(upperLimit)/np.log(1.-ep))
    # print logLowerLimit,logUpperLimit,lowerLimit,upperLimit
    newInts = np.arange(np.ceil(logLowerLimit), np.trunc(logUpperLimit)+1,1)
    newIset = set((1.-ep)**newInts)
    Ikeys = I.viewkeys() #dynamic view of the keys in I
    #erase all keys that are not in new I 
    for toRemove in (Ikeys - newIset):
        del I[toRemove]
    for toUpdate in Ikeys:
        #update object here
        I[toUpdate].append(newItem)
    for toAdd in (newIset - Ikeys):
        #create new object here
        I[toAdd] = [newItem]
    return I


def solveRandomLogistic(K,F,y,Ftest,ytest,fnorm,ridgeReg,Uiter):
    numEvaluations = 0
    Kiter = Uiter[:K]
    llBest = -np.inf*np.ones(K)
    scoreBest = np.zeros(K)
    bBest = np.zeros((F.shape[1],K))
    #choose a random subset and then evaluate nested models
    regr = linear_model.LogisticRegression(penalty='l2',solver='lbfgs',fit_intercept=False,C=1.0/ridgeReg)
    bestfeats = Uiter[:K]
    Fcurrent = np.empty((y.shape[0],0))
    FcurrentTest = np.empty((ytest.shape[0],0))
    for k,u in enumerate(Kiter):
        # print k
        #refit new coeffs and compute new likelihoods
        Fcurrent = np.hstack((Fcurrent,getFeature(F,u).reshape((y.shape[0],1))))
        regr.fit(Fcurrent, y)    
        numEvaluations += 1
        llBest[k] = getLogLikelihood(Fcurrent,regr.coef_.T,y,ridgeReg) - fnorm
        FcurrentTest = np.hstack((FcurrentTest,getFeature(Ftest,u).reshape((ytest.shape[0],1))))
        scoreBest[k] = regr.score(FcurrentTest,ytest)
    return bestfeats,llBest,scoreBest,numEvaluations

def solveLocalSearchLogistic(K,F,y,Ftest,ytest,fnorm,ridgeReg,Uiter=-1):
    if Uiter==-1:
        numIt = F.shape[1]
        Uiter = np.arange(numIt)
    localInstance = StreamingLocalSearchVerbose(K,y,ytest,fnorm,ridgeReg,regressionType='logistic',p=len(Uiter))
    for u in Uiter:
        localInstance.update(getFeature(F,u), getFeature(Ftest,u), u)
    return localInstance


def solveStreamingRegression(K,tol,F,y,Ftest=None,ytest=None,fnorm=0,ridgeReg=0,Uiter=None,regressionType='logistic',fillBestInstance=True):
    if regressionType in ['logistic']:
        regr = linear_model.LogisticRegression(penalty='l2',solver='lbfgs',fit_intercept=False,C=1.0/ridgeReg)
        likelihood_fn = getLogLikelihood
    elif regressionType in ['ridge']:
        regr = linear_model.Ridge(alpha=0, fit_intercept=True)
        likelihood_fn = getLogLikelihoodLinear
    m = 0
    uBest = 0
    totalEvaluations = 0
    I = dict()
    #add elements to instances
    Iview = I.viewitems()
    for u in Uiter:
        Fslice = getFeature(F,u).reshape((F.shape[0],1))
        regr.fit(Fslice, y)
        totalEvaluations += 1
        uEval = likelihood_fn(Fslice,regr.coef_.T,y,ridgeReg) - fnorm
        if uEval > m:
            m = uEval
            uBest = u
        I = updateInstances(I, m, K, tol, u)
        
    if len(I) == 0:
        print "no instances, exiting..."
        return [],0,0,len(Uiter)
    print "running %d final instances..." % len(I)
    outputs = []*len(I)
    for t,elements in Iview:
        completedInstances = solveThresholdedGreedy(K,t,F,y,Ftest,ytest,regressionType,fnorm,ridgeReg,elements,psize=len(Uiter))
        outputs.append(completedInstances)

    lls,numEvaluations = zip(*[(out.getStats()[1][-1],out.getStats()[3]) for out in outputs])
    # lls,numEvaluations = zip(*[(lambda x: (x[1][-1],x[3]))(out.getStats()) for out in outputs])
    bestInstanceIdx = np.argmax(lls)
    totalEvaluations += sum(numEvaluations)
    bestInstance = outputs[bestInstanceIdx]
    bestNumEvaluations = numEvaluations[bestInstanceIdx]

    if fillBestInstance:
        # if output size ksize is less than k, then add K-ksize from the other buckets via local search
        #add these extra evaluations to totalEvaluations with updated getStats
        if not bestInstance.isFull():
            unseenFeatures = set().union(*[I[t] for t in I if t != bestInstance.thresh]) - set(I[bestInstance.thresh])
            # print "%d unseen features" % len(unseenFeatures)
            #add to the set greedily
            for u in unseenFeatures:
                # print "Adding %s to best instance" % str(u)
                bestInstance.updatePastK(getFeature(F,u),getFeature(Ftest,u),u)
        
        totalEvaluations = totalEvaluations + bestInstance.getStats()[3] - bestNumEvaluations
    
    #compare to the singleton uBest
    if m > bestInstance.getStats()[1][-1]:
        Fslice = getFeature(F,u).reshape((F.shape[0],1))
        regr.fit(Fslice, y)
        pdim = len(Uiter)+1
        if Ftest:
            score = regr.score(getFeature(Ftest,u).reshape((Ftest.shape[0],1)),ytest)
        else:
            score = None
        return [uBest]*pdim,[m]*pdim,[score]*pdim,totalEvaluations
    else:
        #replace number of evaluations in bestInstance with totalEvaluations
        return bestInstance.getStats()[:3] + (totalEvaluations,)

def solveThresholdedGreedy(K,thresh,F,y,Ftest,ytest,regressionType,fnorm,ridgeReg,Uiter=-1,psize=0):
    if Uiter==-1:
        numIt = F.shape[1]
        Uiter = np.arange(numIt)
    greedyInstance = StreamingGreedyVerbose(K,y,ytest,thresh,fnorm,ridgeReg,regressionType=regressionType,p=psize)
    for u in Uiter:
        if not greedyInstance.isFull():
            greedyInstance.update(getFeature(F,u), getFeature(Ftest,u), u)
    return greedyInstance

