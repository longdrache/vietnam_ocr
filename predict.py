import os

import tensorflow as tf

from crnn import get_model
from loader import SIZE, MAX_LEN, TextImageGenerator, decode_batch
from keras import backend as K
import glob                                                                 
import argparse
from PIL import Image

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

