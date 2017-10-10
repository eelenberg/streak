import os
import matplotlib.pyplot as plt
import keras
from keras.applications import inception_v3 as inc_net
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from skimage.io import imread
from lime.explanation import id_generator
import numpy as np
from time import time
from skimage import io
import argparse
import load_networks
import lime
from lime import lime_image
import lime_image_streak
from skimage.segmentation import mark_boundaries
from tf_predict import *
# from tensorflow.examples import label_image
# import label_image #local version

# #tensorflow preprocessing
def import_tf_imgs(path_list):
    out = []
    for img_path in path_list:
        # tmp  = label_image.read_tensor_from_image_file(img_path, input_height=299, input_width=299)
        tmp = io.imread(img_path)
        out.append(tmp)
    # return np.vstack(out)
    return out

def import_tf_img(img_path):
    # return label_image.read_tensor_from_image_file(img_path, input_height=299, input_width=299)
    return np.expand_dims(io.imread(img_path),axis=0)

  
#keras preprocessing
def transform_img_fn(path_list):
    out = []
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = inc_net.preprocess_input(x)
        out.append(x)
    return np.vstack(out)


def streakInterpretationExample(testImageName,model_classifier_fn,label_idx=None,
    K=5,feature_selection='greedy_likelihood',kernel_size=6,num_samples=20,
    plotFlag=0,writeFlag=0,logFlag=0):
    
    testImage = transform_img_fn([testImageName])[0]
    #new explainer class that can support streak feature_selection types
    explainer = lime_image_streak.LimeImageExplainer(feature_selection=feature_selection)
    texplain_start = time()
    if label_idx is None:
        num_top_labels = 1
    else:
        #if interpreting a particular label_idx, increase num_top_labels so it is included
        num_top_labels = 5
    explanation = explainer.explain_instance(testImage, model_classifier_fn, top_labels=num_top_labels, 
        num_features=K, qs_kernel_size=kernel_size, hide_color=0, num_samples=num_samples)
    texplain_end = time()
    t = texplain_end - texplain_start
    if label_idx is None:
        label_idx = explanation.top_labels[0]
    numSelected = len(explanation.local_exp[label_idx])

    tempImage, mask = explanation.get_image_and_mask(label_idx, positive_only=True, num_features=K, hide_rest=True)

    if plotFlag:
        fig = plt.imshow(tempImage/2+0.5) 
        plt.show()

    if writeFlag:
        outputName = 'outputs/' + id_generator() + '.jpg'
        #converts from float64 to uint8
        # I'm dividing by 2 and adding 0.5 because of how this (keras) Inception represents images
        io.imsave(outputName,tempImage / 2 + 0.5)
        # #surrounds with whitespace
        # fig = plt.figure()
        # ax = plt.axes([0,0,1,1])
        # # I'm dividing by 2 and adding 0.5 because of how this (keras) Inception represents images
        # plt.imshow(tempImage / 2 + 0.5)
        # plt.axis('off')
        # # plt.tight_layout()
        # plt.savefig(outputName, format='jpg', bbox_inches=extent,dpi=300)

    if logFlag:
        resultsFilename = 'resultsInterp.txt'
        if not os.path.exists(resultsFilename):
            #initialize results file with table
            fptr = open(resultsFilename,'w')
            fptr.write('testImage' + '\t' + 'labelIdx' + '\t' + 'K' + '\t' + 'segmentation_kernel_size' '\t' + 
                'feature_selection_type' + '\t' + 'num_samples' + '\t' + 'outputName' + '\t' + 'sparsity' + '\t' + 
                'time' + '\t' + 'evals' + '\n')
            fptr.close()
        #write output to log
        with open(resultsFilename,'a') as lptr:
            lptr.write('\t'.join([testImageName,str(label_idx),str(K),str(kernel_size),feature_selection,str(num_samples),
                outputName,str(numSelected),str(t),str(explanation.num_evals)]) + '\n') 

    return t,explanation.score,tempImage,testImage,numSelected,explanation.num_evals



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='read image inputs.')
    parser.add_argument('filenames', metavar='-f', type=str, nargs='*',
            default=['flowers_etsy.jpg'],
            help='list of filenames to interpret, separated by spaces')
    args = parser.parse_args()
    image_names = args.filenames
    
    print('using keras:', keras.__version__)
    
    if 0: #keras inception network
        inet_model = load_networks.load_keras_inception_imagenet()
        #kernel size should be 3-6, smaller means larger number of segments        
        # K,kernel_size,num_samples,predict,writeFlag,logFlag = 5,6,1000,inet_model.predict,1,1 
        K,kernel_size,num_samples,predict,writeFlag,logFlag = 1,8,5,inet_model.predict,0,0  #devel
        
    elif 1: #keras transfer learning
        inet_model = load_networks.load_keras_inception_transfer()
        #kernel size should be 3-6, smaller means larger number of segments        
        #todo: compare streaming_greedy, highest_weights, and forward_selection with larger K
        # K,kernel_size,num_samples,predict,writeFlag,logFlag = 30,3,1000,inet_model.predict,1,1  
        K,kernel_size,num_samples,predict,writeFlag,logFlag = 5,6,1000,inet_model.predict,1,1  
        # K,kernel_size,num_samples,predict,writeFlag,logFlag = 1,8,5,inet_model.predict,0,0  #devel
        
    else: #tensorflow transfer learning
        g = load_networks.load_tf_transfer()
        #kernel size should be 3-6, smaller means larger number of segments
        #todo: compare streaming_greedy, highest_weights, and forward_selection with larger K
        # K,kernel_size,num_samples,predict,writeFlag,logFlag = 30,3,1000,lambda x: tf_predict(x,0,5,g),1,1 
        K,kernel_size,num_samples,predict,writeFlag,logFlag = 5,6,1000,lambda x: tf_predict(x,0,5,g),1,1 
        # K,kernel_size,num_samples,predict,writeFlag,logFlag = 1,8,5,lambda x: tf_predict(x,0,5,g),0,0  #devel
                    

    fs_list = ['highest_weights','forward_selection','lasso_path','greedy_likelihood','streaming_greedy']
    # fs_list = ['greedy_likelihood','forward_selection','streaming_greedy']
    
    for testImageName in image_names:
        print testImageName
          
        testImageSize = io.imread(testImageName).shape
        num_fs = len(fs_list)
        t = [0]*num_fs
        s = [0]*num_fs
        numSelected = [0]*num_fs
        numEvaluations = [0]*num_fs
        I_exp = [np.zeros(testImageSize)]*num_fs
            

        for i,fs in enumerate(fs_list):
            t[i],s[i],I_exp[i],_,numSelected[i],numEvaluations[i] = streakInterpretationExample(testImageName,predict,
                None,K,fs,kernel_size,num_samples,0,writeFlag,logFlag)
            print "%s, time: %e, evaluations: %d, final fit score: %e, sparsity: %d" % (fs,t[i],numEvaluations[i],
                s[i],numSelected[i])

        