import os

import tensorflow as tf
from keras import applications
from keras.layers import Input, Dense, Activation, Bidirectional, Dropout
from keras.layers import Reshape, Lambda, BatchNormalization,LSTM                                                    
import argparse
from keras import backend as K
from loader import TextImageGenerator, MAX_LEN, CHAR_DICT, SIZE, VizCallback, ctc_lambda_func,decode_batch
import glob        
from keras.models import Model                                                      
import argparse
from PIL import Image

def get_model(input_shape, training, finetune):
    inputs = Input(name='the_inputs', shape=input_shape, dtype='float32')
    base_model = applications.VGG16(weights='imagenet', include_top=False)
    inner = base_model(inputs)
    inner = Reshape(target_shape=(int(inner.shape[1]), -1), name='reshape')(inner)
    inner = Dense(256, kernel_initializer='lecun_normal', name='dense1',use_bias=False)(inner) 
    inner = BatchNormalization()(inner)
    inner =  Activation("relu")(inner)
    inner = Dense(256, kernel_initializer='lecun_normal', name='dense2',use_bias=False)(inner) 
    inner = BatchNormalization()(inner)
    inner =  Activation("relu")(inner)
    inner = Dropout(0.6)(inner)  
    lstm = Bidirectional(LSTM(512, return_sequences=True, kernel_initializer='he_normal', name='lstm1', dropout=0.25, recurrent_dropout=0.25))(inner) 

    y_pred = Dense(CHAR_DICT, activation='softmax', kernel_initializer='he_normal',name='dense3')(lstm)
    
    labels = Input(name='the_labels', shape=[MAX_LEN], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    for layer in base_model.layers:
        layer.trainable = finetune
    
    y_func = K.function([inputs], [y_pred])
    
    if training:
        Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out).summary()
        return Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out), y_func
    else:
        return Model(inputs=[inputs], outputs=y_pred)
def loadmodel(weight_path):
    model = get_model((*SIZE, 3), training=False, finetune=0)
    model.load_weights(weight_path)
    return model

def predict(data):
    sess = tf.compat.v1.Session()
    K.set_session(sess)

    batch_size = 3
   
    test_generator  = TextImageGenerator(data)
    test_generator.build_data()


    model = loadmodel("model/best_0.h5")
    X_test = test_generator.imgs.transpose((0, 2, 1, 3))
    y_pred = model.predict(X_test, batch_size=3)
    decoded_res = decode_batch(y_pred)
    return '{}'.format(decoded_res[0])

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='../data/ocr/model/', type=str)
    parser.add_argument('--data', default='../data/ocr/preprocess/test/', type=str)
    parser.add_argument('--device', default=2, type=int)
    args = parser.parse_args()
    
    
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)

    predict(args.model, args.data)

