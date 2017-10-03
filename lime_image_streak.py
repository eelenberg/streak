"""
Functions for explaining classifiers that use Image data, supports STREAK algorithm.
"""
from __future__ import unicode_literals

import itertools,re,copy,types,os
import numpy as np
import scipy as sp
import sklearn
import skimage as si
from skimage import io,segmentation,util

import streakFunctions as streak
from time import time
from lime import explanation
from lime import lime_base
from lime import lime_image

class MyLimeBase(lime_base.LimeBase):
    '''
    Extend LimeBase class with a new feature selection method that supports streaming_greedy feature selection
    This is described in Section A.8 of the paper
    '''
    def feature_selection(self, data, labels, weights, num_features, method):
        """Selects features for the model. see explain_instance_with_data to
           understand the parameters."""
        tstart = time()
        # print "method is %s" % method
        if method in ['thresholded_greedy','streaming_greedy','streaming_regression','streaming','STREAK','streak','streak_lime']:
            out = self.thresholded_greedy(data,labels,weights,num_features)
            tend = time()
            self.selection_time = tend - tstart
            return out
        else:
            # print "Using LimeBase.feature_selection"
            out = super(MyLimeBase,self).feature_selection(data, labels, weights, num_features, method)
            tend = time()
            self.selection_time = tend - tstart
            return out
    
    @staticmethod
    def thresholded_greedy(data,labels,weights,num_features,streamTolerance=0.5):
        """Threshold features entering as a stream"""
        p = data.shape[1]
        randomStreamIter = streak.random_permutation(range(p))
        llNull = streak.getLogLikelihoodLinear(np.zeros(p),np.zeros(p),labels.ravel(),0)
        used_features,_,_,_ = streak.solveStreamingRegression(num_features,streamTolerance,data,labels,None,None,llNull,0,randomStreamIter,'ridge') 
        
        # #when used_features is empty list, return an array of False
        # #will this throw an error anyway?
        # if not used_features:
        #     return np.array([False]*p)

        return np.array(used_features)


class SegmentedImage(object):
    """
    Lightweight Image class that allows for accessing segments of an image 
    with respect to a reference (e.g. all grey).
    add_segments adds the indexed image segments to the reference
    subtract_segments adds the indexed reference segments to the image
    """

    def __init__(self, raw_image, segments, ref_image):
        """Initializer.
            Collect images, segments, and compute the number of segments`
        Args:
            raw_image: 3d np array image that has been segmented
            segments: 2d np array, same size as 1 channel of raw_image in which each pixel is mapped 
                    to a superpixel, computed by a segmentation algorithn
            ref_image: reference image e.g. all grey, for evaluating a sparse explanation
        """
        
        self.raw_image = raw_image
        self.segments = segments
        self.ref_image = ref_image
        self.num_segments = np.unique(segments).shape[0]
        
    def add_segments(self, segments_to_add):
        return self.add_image_segments(self.ref_image, self.raw_image, self.segments, segments_to_add)

    def subtract_segments(self, segments_to_subtract):
        return self.add_image_segments(self.raw_image, self.ref_image, self.segments, segments_to_subtract)

    @staticmethod
    def add_image_segments(base_image, segments_image, segments, segments_to_add):
        output = copy.deepcopy(base_image)
        mask = np.zeros(segments.shape).astype(bool)
        for z in segments_to_add:
            mask[segments == z] = True
        output[mask] = segments_image[mask]
        return output


class ImageExplanation(object):
    def __init__(self, image, segments):
        """Init function.

        Args:
            image: 3d numpy array
            segments: 2d numpy array, with the output from skimage.segmentation
        """
        self.image = image
        self.segments = segments
        self.intercept = {}
        self.local_exp = {}
        self.num_evals = 0

    def get_image_and_mask(self, label, positive_only=True, hide_rest=False,
                           num_features=5, min_weight=0.):
        """Init function.

        Args:
            label: label to explain
            positive_only: if True, only take superpixels that contribute to
                the prediction of the label. Otherwise, use the top
                num_features superpixels, which can be positive or negative
                towards the label
            hide_rest: if True, make the non-explanation part of the return
                image gray
            num_features: number of superpixels to include in explanation

        Returns:
            (image, mask), where image is a 3d numpy array and mask is a 2d
            numpy array that can be used with
            skimage.segmentation.mark_boundaries
        """
        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        segments = self.segments
        image = self.image
        exp = self.local_exp[label]
        mask = np.zeros(segments.shape, segments.dtype)
        if hide_rest:
            temp = np.zeros(self.image.shape)
        else:
            temp = self.image.copy()
        if positive_only:
            fs = [x[0] for x in exp
                  if x[1] > 0 and x[1] > min_weight][:num_features]
            for f in fs:
                temp[segments == f] = image[segments == f].copy()
                mask[segments == f] = 1
            return temp, mask
        else:
            for f, w in exp[:num_features]:
                if np.abs(w) < min_weight:
                    continue
                c = 0 if w < 0 else 1
                mask[segments == f] = 1 if w < 0 else 2
                temp[segments == f] = image[segments == f].copy()
                temp[segments == f, c] = np.max(image)
                for cp in [0, 1, 2]:
                    if c == cp:
                        continue
                    # temp[segments == f, cp] *= 0.5
            return temp, mask



class LimeImageExplainer(object):
    #todo: refactor to extend class instead of rewriting it?
    """Explains predictions on Image (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self, kernel_width=.25, verbose=False,
                 feature_selection='auto'):
        """Init function.

        Args:
            training_data: numpy 2d array
            training_labels: labels for training data. Not required, but may be
                used by discretizer.
            feature_names: list of names (strings) corresponding to the columns
                in the training data.
            categorical_features: list of indices (ints) corresponding to the
                categorical columns. Everything else will be considered
                continuous. Values in these columns MUST be integers.
            categorical_names: map from int to list of names, where
                categorical_names[x][y] represents the name of the yth value of
                column x.
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
                Support added for
                'greedy_likelihood' greedy maximization of likelihood in Section 6.2 of paper
                'thresholded_greedy' LIME + STREAK in Section A.8 of paper
            discretize_continuous: if True, all non-categorical features will
                be discretized into quartiles.
            discretizer: only matters if discretize_continuous is True. Options
                are 'quartile', 'decile' or 'entropy'
        """
        kernel_width = float(kernel_width)

        def kernel(d):
            return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        self.feature_selection = feature_selection
        
        # self.base = lime_base.LimeBase(kernel, verbose)
        #call new child class as lime_base
        self.base = MyLimeBase(kernel, verbose)

    def explain_instance(self, image, classifier_fn, labels=(1,),
                         hide_color=None,
                         top_labels=5, num_features=100000, num_samples=1000,
                         batch_size=10,
                         qs_kernel_size=4,
                         distance_metric='cosine', model_regressor=None):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            data_row: 1d numpy array, corresponding to a row
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
            qs_kernel_size: the size of the kernal to use for the quickshift
                segmentation

        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """
        from skimage.segmentation import quickshift
        segments = quickshift(image, kernel_size=qs_kernel_size,
                              max_dist=200, ratio=0.2)
        # print "Number of (quickshift) segments: %d" % np.unique(segments).shape[0]
        fudged_image = image.copy()
        if hide_color is None:
            for x in np.unique(segments):
                fudged_image[segments == x] = (
                    np.mean(image[segments == x][:, 0]),
                    np.mean(image[segments == x][:, 1]),
                    np.mean(image[segments == x][:, 2]))
        else:
            fudged_image[:] = hide_color

        top = labels
        
        if self.feature_selection in ['greedy_likelihood','streak_likelihood']:
            #Method described in Section 6.2 of the paper
            # print "Greedy Sparse Likelihood (no perturbed images)"
            
            _, imageLabel = self.data_labels(image, fudged_image, segments,
                                            classifier_fn, num_samples=1,
                                            batch_size=1)

            ret_exp = ImageExplanation(image, segments)
            if top_labels:
                top = np.argsort(imageLabel[0])[-top_labels:]
                ret_exp.top_labels = list(top)
                ret_exp.top_labels.reverse()

            tol = 0.7
            y = 1 #this will maximize likelihood of the given label, instead of matching the model's likelihood
            Uiter = streak.random_permutation(range(np.unique(segments).shape[0]))
            yss = classifier_fn(np.expand_dims(fudged_image,axis=0))
            indexed_image = SegmentedImage(image,segments,fudged_image)

            for label in top:
                fnorm = yss[0,label]
                gl_superpixels,gl_score,num_evaluations = solveStreamingML(num_features,tol,indexed_image,y,label,fnorm,classifier_fn,Uiter)
                ret_exp.intercept[label] = -1.
                #assign every segment in the set with score 1, in the future maybe rescale?
                ret_exp.local_exp[label] = zip(gl_superpixels,[1.]*len(gl_superpixels))
                ret_exp.score = gl_score
                ret_exp.num_evals = num_evaluations
            return ret_exp

        else:
            #run same ImageExplanation code as before, now supporting thresholded_greedy
            data, labels = self.data_labels(image, fudged_image, segments,
                                            classifier_fn, num_samples,
                                            batch_size=batch_size)

            distances = sklearn.metrics.pairwise_distances(
                data,
                data[0].reshape(1, -1),
                metric=distance_metric
            ).ravel()

            ret_exp = ImageExplanation(image, segments)
            if top_labels:
                top = np.argsort(labels[0])[-top_labels:]
                ret_exp.top_labels = list(top)
                ret_exp.top_labels.reverse()

            for label in top:
                (ret_exp.intercept[label],
                 ret_exp.local_exp[label],
                 ret_exp.score) = self.base.explain_instance_with_data(
                    data, labels, distances, label, num_features,
                    model_regressor=model_regressor,
                    feature_selection=self.feature_selection)
                ret_exp.num_evals = num_samples
            return ret_exp

    def data_labels(self,
                    image,
                    fudged_image,
                    segments,
                    classifier_fn,
                    num_samples,
                    batch_size=10):
        """Generates images and predictions in the neighborhood of this image.

        Args:
            image: 3d numpy array, the image
            fudged_image: 3d numpy array, image to replace original image when
                superpixel is turned off
            segments: segmentation of the image
            classifier_fn: function that takes a list of images and returns a
                matrix of prediction probabilities
            num_samples: size of the neighborhood to learn the linear model
            batch_size: classifier_fn will be called on batches of this size.

        Returns:
            A tuple (data, labels), where:
                data: dense num_samples * num_superpixels
                labels: prediction probabilities matrix
        """
        n_features = np.unique(segments).shape[0]
        data = np.random.randint(0, 2, num_samples * n_features).reshape(
                (num_samples, n_features))
        labels = []
        data[0, :] = 1
        imgs = []
        for row in data:
            temp = copy.deepcopy(image)
            zeros = np.where(row == 0)[0]
            mask = np.zeros(segments.shape).astype(bool)
            for z in zeros:
                mask[segments == z] = True
            temp[mask] = fudged_image[mask]
            imgs.append(temp)
            if len(imgs) == batch_size:
                preds = classifier_fn(np.array(imgs))
                labels.extend(preds)
                imgs = []
        if len(imgs) > 0:
            preds = classifier_fn(np.array(imgs))
            labels.extend(preds)
        return data, np.array(labels)


def solveStreamingML(K,tol,indexed_image,y,classInd,fnorm,classifier_fn,Uiter=None,fillBestInstance=True):
    #implemented as multiple passes and running several instances of Thresholded Greedy with different thresholds
    #todo: support parallel instances
    m = 0
    uBest = 0
    totalEvaluations = 0
    I = dict()
    #add elements to instances, first evaluate all singletons
    Iview = I.viewitems()
    uSingletons = classifier_fn(np.array([indexed_image.add_segments([u]) for u in Uiter]))[:,classInd] - fnorm
    for uIndex,u in enumerate(Uiter):
        Fslice = indexed_image.add_segments([u])
        totalEvaluations += 1
        uEval = uSingletons[uIndex]
        if uEval > m:
            m = uEval
            uBest = u
        I = streak.updateInstances(I, m, K, tol, u)
        
    if len(I) == 0:
        print "no instances, exiting..."
        return [],0,len(Uiter)
    print "running %d final instances..." % len(I)
    outputs = []*len(I)
    for t,elements in Iview:
        completedInstances = solveThresholdedGreedyML(K,t,indexed_image,y,classInd,fnorm,classifier_fn,elements,psize=len(Uiter))
        outputs.append(completedInstances)
    
    lls,numEvaluations = zip(*[tuple(out.getStats()[1:3]) for out in outputs])
    bestInstanceIdx = np.argmax(lls)
    #get number of evaluations across all instances and add to totalEvaluations
    totalEvaluations += sum(numEvaluations)
    bestInstance = outputs[bestInstanceIdx]
    bestNumEvaluations = numEvaluations[bestInstanceIdx]

    if fillBestInstance:        
        # if output size ksize is less than k, then add K-ksize elements from the other buckets via local search
        #add these extra evaluations to totalEvaluations with updated getStats
        if not bestInstance.isFull():
            # print bestInstance.ksize,bestInstance.localsize
            unseenFeatures = set().union(*[I[t] for t in I if t != bestInstance.thresh]) - set(I[bestInstance.thresh])
            # print "%d unseen features" % len(unseenFeatures)
            #add to the set greedily
            for u in unseenFeatures:
                bestInstance.updatePastK(u)
        
        if not bestInstance.isFull():
            #if still not full, also add features already seen and ignored by bestInstance (even if below threshold)
            seenFeatures = set(I[bestInstance.thresh]) - set(bestInstance.bestfeats[:(bestInstance.ksize+bestInstance.localsize)])
            for u in seenFeatures:
                # print "Adding %s (already seen) to best instance" % str(u)
                bestInstance.updatePastK(u)
        
        totalEvaluations = totalEvaluations + bestInstance.getStats()[2] - bestNumEvaluations
    
    #compare to the singleton uBest
    if m > bestInstance.getStats()[1]:
        # print "Singleton is best"
        return [uBest],m,totalEvaluations
    else:
        #replace number of evaluations in bestInstance with totalEvaluations
        return bestInstance.getStats()[:2] + (totalEvaluations,)

def solveThresholdedGreedyML(K,thresh,F,y,classInd,fnorm,classifier_fn,Uiter=None,psize=0):
    if Uiter is None:
        numIt = F.shape[1]
        Uiter = np.arange(numIt)
    greedyInstance = streak.StreamingGreedyML(K,F,thresh,classifier_fn,y,classInd,fnorm,p=psize)
    for u in Uiter:
        if not greedyInstance.isFull():
            greedyInstance.update(u)
    return greedyInstance

