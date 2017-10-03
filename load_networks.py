import os
from keras.applications import inception_v3 as inc_net
# from tensorflow.examples import label_image
import label_image #local version



def load_tf_transfer():
    model_dir = os.path.join(os.getcwd(),'retrained','classify_image_graph_def.pb')
    graph = label_image.load_graph(model_dir)
    out = {
        'graph': graph,
        'label_path': os.path.join(os.getcwd(),'retrained','output_labels.txt'),
        'input_name': 'import/DecodeJpeg',
        'output_name': 'import/final_result'}
    return out

def load_keras_inception_transfer():
    from keras.models import Sequential, Model 
    from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Input
    img_width, img_height = 299, 299
    classes = ['daisy','dandelion','roses','sunflowers','tulips']
    weights_file = os.path.join(os.getcwd(),'retrained','bottleneck_fc_model.h5')
    print 'loading inception'
    first_model = inc_net.InceptionV3(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3), pooling=None)
    print 'loading top model'
    top_model = Sequential()
    top_model.add(Flatten(input_shape=first_model.output_shape[1:]))
    top_model.add(Dense(len(classes), activation='sigmoid'))
    top_model.load_weights(weights_file)
    print 'combining...'
    inet_pred = first_model.output
    inet_pred = top_model(inet_pred)
    model = Model(inputs=first_model.input, outputs=inet_pred)
    print 'finished'
    return model
    

def load_keras_inception_imagenet():
    model = inc_net.InceptionV3()
    return model

