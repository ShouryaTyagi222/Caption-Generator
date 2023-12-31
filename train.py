import argparse
import os
from tqdm.notebook import tqdm

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.models import Model

from src.model import *
from src.utils import *
            
def main(args):
    WORKING_DIR = '/'
    BASE_DIR=args.input_path
    
    print('LOADING THE REQUIREMENTS...')
    if args.resume_training==None:

        model=VGG16()
        model=Model(inputs=model.inputs,outputs=model.layers[-2].output)

        features={}
        directory=os.path.join(BASE_DIR,'Images')
        for img_name in tqdm(os.listdir(directory)):
            img_path=directory+'/'+img_name
            image=load_img(img_path,target_size=(224,224))
            image=img_to_array(image)
            image=image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
            image=preprocess_input(image)
            feature=model.predict(image,verbose=0)
            image_id=img_name.split('.')[0]
            features[image_id]=feature
    
        #load the caption data
        with open(os.path.join(BASE_DIR,'captions.txt'),'r') as f:
            next(f)
            captions_doc=f.read()

        mapping={}
        for line in tqdm(captions_doc.split('\n')):
            tokens=line.split(',')
            if len(line)<2:
                continue
            image_id,caption=tokens[0],tokens[1:]
            image_id=image_id.split('.')[0]
            caption=' '.join(caption)
            if image_id not in mapping:
                mapping[image_id]=[]
            mapping[image_id].append(caption)

        clean(mapping)

        all_captions=[]
        for key in mapping:
            for caption in mapping[key]:
                all_captions.append(caption)
            
        tokenizer=Tokenizer()
        tokenizer.fit_on_texts(all_captions)

        max_length=max(len(caption.split()) for caption in all_captions)

        model=model_func(max_length,vocab_size)
        epoch=0

    else:
        model,epoch,features,tokenizer,max_length,mapping=model_load(args.model_path)

    vocab_size=len(tokenizer.word_index)+1

    train=list(mapping.keys())
    
    n_epochs=int(args.epochs)
    batch_size=int(args.batch_size)
    steps=len(train)//batch_size

    print('STARTING TRAINING...')
    for i in range(n_epochs):
        generator=data_generator(train,mapping,features,tokenizer,max_length,vocab_size,batch_size)
        model.fit(generator,epochs=1,steps_per_epoch=steps,verbose=1)

        if (epoch+i+1)%5==0:
            model_save(model,epoch+i+1,features,tokenizer,max_length,mapping)
            print('MODEL SAVED AT EPOCH :',epoch+i+1)
    
    model_save(model,epoch+i+1,features,tokenizer,max_length,mapping)
    print('MODEL SAVED AT EPOCH :',epoch+i+1)




def parse_args():
    parser = argparse.ArgumentParser(description="Train Caption Generator", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--input_path", type=str, default=None, help="path to the input folder")
    parser.add_argument("-e", "--epochs", type=str, default="10", help="Number of Epochs")
    parser.add_argument("-b", "--batch_size", type=str, default="64", help="Batch size")
    parser.add_argument("-r", "--resume_training", type=str, default=None, help="To Resume the Training, give the model path")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)