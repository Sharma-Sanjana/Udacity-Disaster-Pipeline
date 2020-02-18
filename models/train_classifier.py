import sys
# import libraries
import pandas as pd
import numpy as np
import pickle

# download necessary NLTK data
import nltk
nltk.download(['all'])

# import statements (from ML pipeline prep)
import re
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier



def load_data(database_filepath):
    """Loading the filepath"""
    name='sqlite:///' + database_filepath
    engine = create_engine(name)
    df = pd.read_sql_table('Disasters', con=engine)
    X = df.message
    Y = df.loc[:,'related':'direct_report']
    category_names=Y.columns
    
    return X, Y, category_names

def tokenize(text):
    """Tokenize and transform. Returns cleaned text"""
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    def tokenize(text):
        ##copy-pasted from previous lessons
        detected_urls = re.findall(url_regex, text)
        for url in detected_urls:
            text = text.replace(url, "urlplaceholder")

        tokenizer=RegexpTokenizer(r'\w+')
        tokens=tokenizer.tokenize(text)
        ##from previous lessons
        lemmatizer = WordNetLemmatizer()

        clean_tokens = []
        for tok in tokens:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

        return clean_tokens


def build_model():
    """Returns model with pipeline and classifier"""
    moc=MultiOutputClassifier(RandomForestClassifier(random_state=42))
    pipeline = Pipeline([('vect',CountVectorizer(tokenizer=tokenize)),
                      ('tfidf',TfidfTransformer()),
                      ('clf',moc)])
    
    parameters = {'clf__estimator__max_depth':[25,50,None],
             'clf__estimator__min_samples_leaf':[2,5,10]}
    
    cv = GridSearchCV(pipeline, parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Printing model results"""
    y_pred=pipeline.predict(X_test)
    print(classification_report(X_test, y_pred, target_names=category_names))
    results=pd.DataFrame(columns=['Category','f_score','precision','recall'])
    


def save_model(model, model_filepath):
    """Saving model as a pickle file"""
    pickle.dump(model,open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()