# preprocessing for ml model 
'''
1. clean the data (optional)
2. use tfiffvectorizer 
'''
# preprocessing for dl model
'''
1. clean the data (optional)
2. use the tokenizer to convert to sequences 
3. pad the sequences
''' 


# Trying out preprocessing  using the same pipeline as our project
#helper functions for lemmatizations

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import pickle as pk 
from nltk import pos_tag
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


list_classes = ["toxic", "severe toxic", "obscene", "threat", "insult", "identity hate"]
embed_size = 300 # how big is each word vector
max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of

wnl = WordNetLemmatizer()
stop_dict = stopwords.words('english')
tokenizer = pk.load(open('saved_models/tfidf','rb'))
model = pk.load(open('saved_models/model','rb'))

#loading bi-lstm model 
dnn_tokenizer = pk.load(open('saved_models/lstm_glove_tok','rb'))
dnn_model = tf.keras.models.load_model('saved_models/lstm_glove.h5')


# Trying out preprocessing  using the same pipeline as our project
#helper functions for lemmatizations
def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN':'n', 'JJ':'a',
              'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n' 

def lemmatize_sent(text): 
    wnl = WordNetLemmatizer()
    # Text input is string, returns lowercased strings.
    ls = list(wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) for word, tag in pos_tag(word_tokenize(text)))

    str = ''

    for i in  ls:
          str += i.lower() + ' '

    return str  

def clean_text(x):
    
    # remove html tags
    regex = re.compile('<.*?>')
    input =  re.sub(regex, '', x)

    #remove punctuations, numbers.
    input = re.sub('[!@#$%^&*()\n_:><?\-.{}|+-,;""``~`—]|[0-9]|/|=|\[\]|\[\[\]\]',' ',input)
    input = re.sub('[“’\']','',input)   
        
    #remove stopwords

    tmp_str  = ''
    for i in word_tokenize(input):

       if i  not in stop_dict and len(set(i)) > 2:
            tmp_str += i + ' '

    
    #lemmatize the text.
    

    return lemmatize_sent(tmp_str)


def ml_preprocess(text):
    toxcity = True
    text_ = clean_text(text)
    text_ = tokenizer.transform([text_])
    classes = model.predict(text_)
    predict_probas  = model.predict_proba(text_)

    if sum(classes[0]) == 0:
        toxcity = False
 
    

    return {'classes':list_classes, 
            'proba':(predict_probas * 100)[0],
            'toxic':toxcity,
            'message':text,
            'model':'SGDClassifier'
            }


 
def dl_preporcess(text):
    toxcity = True
    text_ = clean_text(text)
    seq = dnn_tokenizer.texts_to_sequences([text])
    # print(seq)

    padded_text_ = pad_sequences(seq,maxlen =maxlen)
    # print(padded_text_.shape)


    preds = dnn_model.predict(padded_text_)

    if sum(preds[0] > 0.5) == 0:
        toxcity = False


    return {'classes':list_classes, 
            'proba':(preds * 100)[0],
            'toxic':toxcity,
            'message':text,
            'model':'BILSTM+glove embeddings'
            }


def proprocess_input(text,mode = 'ml'):

    if mode == 'ml':
        return  ml_preprocess(text)
    else:
        return dl_preporcess(text)




#in the method we are going to convert the numeric output of class labels.
def postprocess_output(mode = 'ml'):

    if mode:
            pass


if __name__ == "__main__":
    
    input_ = "im sorry I screwed around with someones talk page.  It was very bad to do.  I know how having the templates on their talk page helps you assert your dominance over them.  I know I should bow down to the almighty administrators.  But then again, I'm going to go play outside....with your mom.   76.122.79.82"
    # input_ = 'COCKSUCKER BEFORE YOU PISS AROUND ON MY WORK'
    print(proprocess_input(input_,mode = 'dnn'))