import sys
import subprocess
import os
import tensorflow as tf
from skimage import io
import numpy as np
import label_image
import load_networks
import argparse

'''
Adapted from tensorflow source code: tensorflow/examples/label_image/label_image.py
example usage
python tf_predict.py image1.jpg image2.jpg
'''

def tf_predict(imageList,verbose=0,numPred=5,graphParams=None):
    #graphParams is a dictionary with imported graph object, filename for labels, and names of the input and output layers
    if not graphParams:
        graphParams = load_networks.load_tf_transfer()

    graph = graphParams['graph']
    labels = label_image.load_labels(graphParams['label_path'])
    input_operation = graph.get_operation_by_name(graphParams['input_name'])
    output_operation = graph.get_operation_by_name(graphParams['output_name'])
    d = len(imageList)
    outputProbs = np.zeros((d,len(labels)))
    
    for image in range(d):
        
        with tf.Session(graph=graph) as sess:
            results = sess.run(output_operation.outputs[0],
                              {input_operation.outputs[0]: imageList[image]})
            results = np.squeeze(results)
        outputProbs[image,:] = results

        if verbose:
            top_k = results.argsort()[-numPred:][::-1]
            for i in top_k:
                print('%s (score = %.5f)') % (labels[i], results[i])
    
    #return full num_images x num_classes matrix of probabilities
    return outputProbs


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='read image inputs.')
    parser.add_argument('filenames', metavar='-f', type=str, nargs='*',
        default=['flowers_etsy.jpg'],
        help='list of filenames to predict, separated by spaces')
    args = parser.parse_args()
    img_list = args.filenames

    tf_predict([io.imread(img_path) for img_path in img_list],1) 

    # tf_predict([io.imread(img_path)],1) 

