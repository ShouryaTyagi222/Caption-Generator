import os
import pickle as pkl
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense, LSTM,Embedding,Dropout,add

def model_func(max_length,vocab_size):
    inputs1=Input(shape=(4096,))
    fe1=Dropout(0.4)(inputs1)
    fe2=Dense(256,activation='relu')(fe1)
    #sequence feature layer
    inputs2=Input(shape=(max_length,))
    se1=Embedding(vocab_size,256,mask_zero=True)(inputs2)
    se2=Dropout(0.4)(se1)
    se3=LSTM(256)(se2)

    #decoder
    decoder1=add([fe2,se3])
    decoder2=Dense(256,activation='relu')(decoder1)
    outputs=Dense(vocab_size,activation='softmax')(decoder2)

    model=Model(inputs=[inputs1,inputs2],outputs=outputs)
    model.compile(loss='categorical_crossentropy',optimizer='adam')

    return model

def model_save(model,itr,features,tokenizer,max_length):
    model_path=os.path.join(f'checkpoint','caption_generator_model_epoch_{itr}.pkl')
    with open(model_path, 'wb') as f:
        pkl.dump({"model":model,"epoch":itr,"features":features,"tokenizer":tokenizer,"max_length":max_length}, f, protocol=pkl.HIGHEST_PROTOCOL)

def model_load(model_path):
    with open(model_path, 'rb') as f:
        mp = pkl.load(f)
    model=mp['model']
    epoch=mp['epoch']
    features=mp['features']
    tokenizer=mp['tokenizer']
    max_length=mp['max_length']
    return model,epoch,features,tokenizer,max_length