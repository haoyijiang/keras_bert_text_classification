import numpy as np
import pandas as pd
from keras_bert import load_trained_model_from_checkpoint,Tokenizer
from keras.layers import *
from keras.callbacks import *
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.models import Model,Sequential
from keras.activations import softmax,sigmoid
from keras.optimizers import Adam
from keras import losses
import codecs
import yaml
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras_self_attention import SeqSelfAttention,SeqWeightedAttention

## 参数
maxlen = 100
Batch_size=16
Epoch =1
config_path="/home/thefair/haoyi/ad_classification/uncased_L-12_H-768_A-12/bert_config.json"
checkpoint_path="/home/thefair/haoyi/ad_classification/uncased_L-12_H-768_A-12/bert_model.ckpt"
dict_path="/home/thefair/haoyi/ad_classification/uncased_L-12_H-768_A-12/vocab.txt"
def get_token_dict(dict_path):
    """

    :param dict_path:bert模型的vocab.txt文件
    :return: 将文件中字进行编码
    """
    token_dict={}
    with codecs.open(dict_path,"r","utf-8") as reader:
        for line in reader:
            token=line.strip()
            token_dict[token]=len(token_dict)
    return token_dict

def data_training_generator():
    """
    读取正负样本，并生成正负样本
    :return: training dataset validation dataset
    """
    neg = []
    pos = []
    with codecs.open("positive.txt","r",'utf-8') as reader:
        for line in reader:
            pos.append(line.strip())

    with codecs.open("negative.txt","r","utf-8") as reader:
        for line in reader:
            neg.append(line.strip())

    return pos[:9000],neg[:9000]

def get_encode(pos,neg,token_dict):
    """

    :param pos:正样本
    :param neg:负样本
    :param token_dict:相关词典
    :return:
    """
    all_data=pos+neg
    X1=[]
    X2=[]
    tokenizer=Tokenizer(token_dict)
    for line in all_data:
        x1,x2=tokenizer.encode(first=line)
        X1.append(x1)
        X2.append(x2)
    X1=sequence.pad_sequences(X1,maxlen=maxlen,padding='post',truncating='post')
    X2=sequence.pad_sequences(X2,maxlen=maxlen,padding="post",truncating='post')
    return [X1,X2]

def build_bert_model(X1,X2):
    """
    :param X1,X2:编码后的结果
    :return: 构建bert第一种模型
    """
    bert_model=load_trained_model_from_checkpoint(config_file=config_path,checkpoint_file=checkpoint_path,seq_len=None)

    wordvec=bert_model.predict([X1,X2])
    return wordvec

#def build_xlnet_modle(

def build_model_attention():
    model=Sequential()
    model.add(Bidirectional(LSTM(units=128,dropout=0.5,recurrent_dropout=0.5,return_sequences=True)))
    #model.add(Bidirectional(LSTM(128,recurrent_dropout=0.5)))
    model.add(SeqWeightedAttention())
    model.add(Dense(1,activation=sigmoid))
    model.compile(loss=losses.binary_crossentropy,optimizer=Adam(1e-5),metrics=['accuracy'])
    return model

def build_model():
    model=Sequential()
    model.add(Bidirectional(LSTM(128,return_sequences=True)))
    model.add(Dense(1,activation=sigmoid))
    model.compile(loss=losses.binary_crossentropy,optimizer=Adam(1e-5),metrics=['categorical_accuracy'])
    return model

def train(wordvec,y):
    model=build_model_attention()
    early_stopping=EarlyStopping(monitor='val_loss',patience=20,verbose=2)
    #filepath="weights-title-attention2_best.hdf5"
    #checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,mode='max')
    model.fit(wordvec,y,batch_size=32,epochs=100,validation_split=0.2,callbacks=[early_stopping])
    #yaml_string=model.to_yaml()
    #with open("test_keras_bert3.yml","w") as f:
    #    f.write(yaml.dump(yaml_string,default_flow_style=True))
    model.save('test_keras_bert4.h5')


if __name__=="__main__":
    print("开始")
    pos,neg=data_training_generator()
    token_dict=get_token_dict(dict_path)
    print("开始 encode")
    [X1,X2]=get_encode(pos,neg,token_dict)
    print("开始 bert")
    wordvec=build_bert_model(X1,X2)
    y=np.concatenate((np.ones(9000,dtype=int),np.zeros(9000,dtype=int)))
    print("开始训练")
    train(wordvec,y)