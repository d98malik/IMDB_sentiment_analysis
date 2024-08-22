
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
import regex as re
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("data/IMDB Dataset.csv")

df_train, df_test = train_test_split(df, test_size=0.2)

# # Text preprocessing 
def text_preprocessing(df):
    #cleaning data 
    print("removing special characters")

    corpus =[]
    for text in df['review']:
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower()
        corpus.append(text)
    
    print("removing stop words")
    
    #stop words removal
    stop_words = stopwords.words('english')
    corpus_light = []
    for review in corpus:
        review = review.split()
        review = [word for word in review if not word in stop_words]
        review = ' '.join(review)
        corpus_light.append(review)
    
   
    print("lemmatization")
    #lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_corpus = []
    for doc in corpus_light:
        tokkens = doc.split()
        for i in range(len(tokkens)):
            tokkens[i] = lemmatizer.lemmatize(tokkens[i])
        doc = ' '.join(tokkens)
        lemmatized_corpus.append(doc)
    print("extracting target variable")
    target = df["sentiment"].apply(lambda x: 1 if x == "positive" else 0)
    

    return lemmatized_corpus, target
    


corpus_features_train,corpus_target_train = text_preprocessing(df_train)
corpus_features_test,corpus_target_test = text_preprocessing(df_test)

# # converting word to vector
v = TfidfVectorizer(min_df=100,max_features=500,stop_words='english')
X_train = v.fit_transform(corpus_features_train)
X_train = X_train.toarray()
y_train = corpus_target_train

X_test = v.transform(corpus_features_test)
X_test = X_test.toarray()
y_test = corpus_target_test


## ML modelling
### import decision tree from sklearn

param_grid = {
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': [None, 'sqrt', 'log2']
}

dtr_clf = DecisionTreeClassifier()

grid_search = GridSearchCV(estimator=dtr_clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

dtr_clf = grid_search.best_estimator_
dtr_clf.fit(X_train,y_train)

y_train_predicted = dtr_clf.predict(X_train)
y_test_predicted = dtr_clf.predict(X_test)

## Scoring of model
confusion_matrix(y_train,y_train_predicted)
confusion_matrix(y_test,y_test_predicted)

f1_score(y_train,y_train_predicted), f1_score(y_test,y_test_predicted)