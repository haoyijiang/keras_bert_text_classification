from ad_classification import get_token_dict,dict_path,build_bert_model
from keras.models import model_from_yaml
from keras_bert import Tokenizer
import yaml
from keras.preprocessing import sequence
from keras_self_attention import SeqSelfAttention,SeqWeightedAttention
import keras
def get_encode(text_list,token_dict):
    """

    :param text_list:
    :param token_dict:
    :return:
    """
    X1 = []
    X2 = []
    tokenizer = Tokenizer(token_dict)
    for line in text_list:
        x1, x2 = tokenizer.encode(first=line)
        X1.append(x1)
        X2.append(x2)
    X1 = sequence.pad_sequences(X1, maxlen=maxlen, padding='post', truncating='post')
    X2 = sequence.pad_sequences(X2, maxlen=maxlen, padding="post", truncating='post')
    return [X1, X2]
if __name__=="__main__":
    maxlen=100
    text_list=["TW 0:02 / 41:54 Mind Your Language Season 3 Episode 2 Who Loves Ya Baby? | Funny TV Show (GM)","I have a dream"]
    token_dict=get_token_dict(dict_path)
    [X1,X2]=get_encode(text_list,token_dict)
    print(X1)
    wordvec=build_bert_model(X1,X2)
    print(wordvec)
    #with open("test_keras_bert2.yml","r") as f:
    #    yaml_string=yaml.load(f)
    #model=keras.models.load_model(yaml_string,custom_objects=SeqSelfAttention.get_custom_objects())
    print("loading weights")
    model=keras.models.load_model("test_keras_bert4.h5",custom_objects=SeqWeightedAttention.get_custom_objects())
    result=model.predict(wordvec)
    print(result)
    del model