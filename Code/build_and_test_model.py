import nltk 
import pandas as pd 
import numpy as np 
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn import SVC
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import json 
import PyPDF2
import pickle
from sklearn import metrics



def convert_to_text_and_get_features(data,risk_level,domain):
    names = []
    full_text_lst = []
    labels = []
    for name in data: 
        data_info = data[name]
        pdf_path = data_info[1]
        pdf_file_obj = open(pdf_path,'rb')
        pdf_reader=PyPDF2.PdfFileReader(pdf_file_obj)
        full_text = ''
        total_num_of_pages = pdf_reader.numPages
        for i in range(total_num_of_pages): 
            page = pdf_reader.getPage(i)
            full_text += page.extractText()
        risk_levels_curr = data[name][2]
        risk_data_level =  risk_levels_curr.get(domain)
        if risk_data_level != None: 
            full_text_lst.append(full_text)
            names.append(name)
            if risk_data_level != risk_level:
                labels.append(0)
            else: 
                labels.append(1)
        df = pd.DataFrame(list(zip(names,full_text_lst,labels)),columns=["name","full_text","labels"])
        stop_words = set(stopwords.words('english'))
        ps = PorterStemmer()
        df["full_text"] = df["full_text"].apply(lambda text : text.lower())
        tokenizer = nltk.RegexpTokenizer(r"[a-z|!]+")
        # Tokenize the words 
        df["full_text"] = df["full_text"].apply(lambda text : tokenizer.tokenize(text))
        # Remove words that are in the stop list
        df["full_text"] = df["full_text"].apply(lambda text : [word for word in text if word not in stop_words])
        ps = PorterStemmer()
        # Stem the df 
        df["full_text"] = df["full_text"].apply(lambda text: [ps.stem(word) for word in text])
        df["full_text"] =  df["full_text"].apply(lambda text: " ".join(text))
    return  df

def pre_process(type_of_pre_process,data): 
    if type_of_pre_process == 'bog': 
        n_gram = (1,1)
    else: 
        n_gram = (2,2)
    
    matrix = CountVectorizer(max_features=10000,ngram_range=n_gram)
    X = matrix.fit_transform(data).toarray()
    return X 


def train_model(pre): 
    f = open("train_data.json")
    train_data = json.load(f)
    trained_models = dict()
    domains = ["Random sequence generation","Allocation concealment","Blinding of Participants and Personnel","Incomplete outcome data","Selective reporting"]
    risk_levels = ["low risk","unclear risk","high risk"]
    for dom in domains:
        for risk in risk_levels:
            train_data_df = convert_to_text_and_get_features(train_data,risk,dom)
            X_train = pre_process(pre,train_data_df["full_text"])
            y_train = train_data_df["labels"]
            param_grid = {'C': [0.1,1, 10, 100,1000,5000,7500,10000,100000,100000], 'gamma': [1,0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001,0.00000001,0.000000001],'kernel': ['rbf', 'poly', 'sigmoid','linear']}
            grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
            grid.fit(X_train,y_train)
            scores = cross_val_score(grid, X_train, y_train, cv=5,scoring='f1_macro')
            #print(scores)
            trained_models[dom + " " + risk] = grid
    filename = open('train_models.sav','ab')
    pickle.dumps(trained_models,filename)


def test_model(pre): 
    f = open("train_data.json")
    test_data = json.load(f)
    domains = ["Random sequence generation","Allocation concealment","Blinding of Participants and Personnel","Incomplete outcome data","Selective reporting"]
    risk_levels = ["low risk","unclear risk","high risk"]
    trained_models = open("train_models.sav",'ab')
    for dom in domains:
        for risk in risk_levels:
            test_data_df = convert_to_text_and_get_features(test_data,risk,dom)
            X_test = pre_process(pre,test_data_df["full_text"])
            y_test= test_data_df["labels"]
            trained_model = trained_models[dom + " " + risk]
            y_pred = trained_model.predict(X_test)
            print("METRICS FOR " + dom.upper() + "WITH " + risk.upper())
            print("Precision:", metrics.precision_score(y_test, y_pred))
            print("Recall:", metrics.recall_score(y_test, y_pred))
            print("F1 score:",metrics.f1_score(y_test, y_pred))


def main(): 
    train_model("bog")
    test_model("bog")
    train_model("bigram")
    test_model("bigram")

main()