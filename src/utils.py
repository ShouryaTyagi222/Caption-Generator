import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption=captions[i]
            
            caption=caption.lower()
            
            caption=caption.replace('[^A-Za-z]','')
            caption=caption.replace('\s+',' ')
            caption='startseq '+' '.join([word for word in caption.split() if len(word)>1])+' end'
            captions[i]=caption

#create a data generator to get the data in batch to avoid session crash
def data_generator(data_keys,mapping,features,tokenizer,max_length,vocab_size,batch_size):
    X1,X2,y=list(),list(),list()
    n=0
    while 1:
        for key in data_keys:
            n+=1
            captions=mapping[key]
            for caption in captions:
                #encoder the sequece
                seq=tokenizer.texts_to_sequences([caption])[0]
                #split the sequence into x, y part
                for i in range(1,len(seq)):
                    in_seq,out_seq=seq[:i],seq[i]
                    in_seq=pad_sequences([in_seq],maxlen=max_length)[0]
                    out_seq=to_categorical([out_seq],num_classes=vocab_size)[0]
                    
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
                    
            if n==batch_size:
                X1,X2,y=np.array(X1),np.array(X2),np.array(y)
                yield[X1,X2],y
                X1,X2,y=list(),list(),list()
                n=0