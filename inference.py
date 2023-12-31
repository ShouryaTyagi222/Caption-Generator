from PIL import Image
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import pickle as pkl
from tensorflow.keras.preprocessing.sequence import pad_sequences

#generate caption for the images
def idk_to_words(integer,tokenizer):
    for word, index in tokenizer.word_index.items():
        if index==integer:
            return word
    return None

#generate captions of an image
def predict_caption(model,image,tokenizer,max_length):
    in_text='startseq'
    for i in range(max_length):
        sequence=tokenizer.texts_to_sequences([in_text])[0]
        sequence=pad_sequences([sequence],max_length)
        yhat=model.predict([image,sequence],verbose=0)
        yhat=np.argmax(yhat)
        word=idk_to_words(yhat,tokenizer)
        if word is None:
            break
        in_text+=' '+word
        if word=='end':
            break
    return in_text

def model_load(model_path):
    with open(model_path, 'rb') as f:
        mp = pkl.load(f)
    model=mp['model']
    epoch=mp['epoch']
    features=mp['features']
    tokenizer=mp['tokenizer']
    max_length=mp['max_length']
    mapping=mp['mapping']
    return model,epoch,features,tokenizer,max_length,mapping

def main(args):

    image_id=os.path.basename(args.img_path).split('.')[0]
    image=Image.open(args.img_path)

    model,_,features,tokenizer,max_length,mapping=model_load(args.model_path)

    captions=mapping[image_id]
    print('-------------------Acutal-----------------')
    for caption in captions:
        print(caption)
    y_pred=predict_caption(model,features[image_id],tokenizer,max_length)
    print('----------------------predicted-------------------')
    print(y_pred)
    plt.imshow(image)
    
def parse_args():
    parser = argparse.ArgumentParser(description="Train Generator", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--img_path", type=str, default=None, help="Path to the input image")
    parser.add_argument("-m", "--model_path", type=str, default=None, help="To Resume the Training")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)