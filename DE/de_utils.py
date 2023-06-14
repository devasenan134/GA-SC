import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer
import re
import numpy as np

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

def tokenize_lemmatizor(frame):
    words = []
    lemma_words = []
    lemma_sentences = []
    lemmatizer = WordNetLemmatizer()

    for i in range(len(frame)):
        words = nltk.word_tokenize(frame.iloc[i])
        lemma_words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
        lemma_sentences.append(" ".join(lemma_words))

    return lemma_sentences


def re_lemmatizor(frame):
    lemmatizer = WordNetLemmatizer()
    review = []
    corpus = []

    for i in range(len(frame)):
        review = re.sub('[^a-zA-Z]', ' ', frame.iloc[i])
        review = review.lower()
        review = review.split()
        # these lines represent - words = nltk.word_tokenize(frame.cmd[i])

        review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
        # lemma_words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))])
        
        corpus.append(" ".join(review))
        # lemma_sentences.append(" ".join(lemma_words))

    return corpus

from sklearn.feature_extraction.text import CountVectorizer # bow
def tokens_to_bow(corpus_tokens, tokenizer=1):
    cv = CountVectorizer(max_features=5000)
    tokens = []
    if tokenizer == 1:
        tokens = tokenize_lemmatizor(corpus_tokens)
        X_bow = cv.fit_transform(tokens).toarray()
    else:
        tokens = re_lemmatizor(corpus_tokens)
        X_bow = cv.fit_transform(tokens).toarray()
    features = cv.get_feature_names_out()
    return X_bow, features

from sklearn.feature_extraction.text import TfidfVectorizer # tfidf
def tokens_to_tfidf(corpus_tokens, tokenizer=1):
    tfidf = TfidfVectorizer()
    tokens = []
    if tokenizer:
        tokens = tokenize_lemmatizor(corpus_tokens)
        X_tfidf = tfidf.fit_transform(tokens).toarray()
    else:
        tokens = re_lemmatizor(corpus_tokens)
        X_tfidf = tfidf.fit_transform(tokens).toarray()
    return X_tfidf, tokens

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from random import randint, choices, randrange, random, sample, shuffle


from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score

from sklearn.model_selection import train_test_split
def split(df,label):
    X_tr, X_te, Y_tr, Y_te = train_test_split(df, label, test_size=0.25, random_state=42)
    return X_tr, X_te, Y_tr, Y_te

classifiers = ['LinearSVM', 'RadialSVM', 
               'Logistic',  'RandomForest', 
               'DecisionTree', 'KNeighbors',
               'MultinomialNB']

models = [svm.SVC(kernel='linear'),
          svm.SVC(kernel='rbf'),
          LogisticRegression(max_iter = 1000),
          RandomForestClassifier(n_estimators=200, random_state=0),
          DecisionTreeClassifier(random_state=0),
          KNeighborsClassifier(),
          MultinomialNB()]

def acc_score(df,label):
    Score = pd.DataFrame({"Classifier":classifiers})
    j = 0
    acc = []
    exec_time = []
    X_train,X_test,Y_train,Y_test = split(df,label)
    for i in models:
        model = i

        st = time.time()
        model.fit(X_train,Y_train)
        et = time.time()

        predictions = model.predict(X_test)
        acc.append(accuracy_score(Y_test,predictions))
        exec_time.append(et-st)
        j = j+1     
    Score["Accuracy"] = acc
    Score['Exec_Time_secs'] = exec_time
    Score.sort_values(by="Accuracy", ascending=False,inplace = True)
    Score.reset_index(drop=True, inplace=True)
    return Score

def generate_chromo(features_count, chromo_size):
    features = sample(range(chromo_size), k=features_count)
    features.sort()
    chromo = [1 if i in features else 0 for i in range(chromo_size)]
    return np.array(chromo)

def generate_population(size, features_count, chromo_size):
    return [generate_chromo(features_count, chromo_size) for _ in range(size)]

def fitness_score(population):
    scores = []
    for chromosome in population:
        features = np.where(chromosome!=0)[0]
        logmodel.fit(X_train[:,features],Y_train)         
        predictions = logmodel.predict(X_test[:,features])
        scores.append(accuracy_score(Y_test,predictions))
    scores, population = np.array(scores), np.array(population) 
    inds = np.argsort(scores)                                    
    return list(scores[inds][::-1]), list(population[inds,:][::-1])


def de_fitness(chromo1, chromo2):
    max_score = 0
    best_chromo = None

    for chromo in [chromo1, chromo2]:
        features = np.where(chromo!=0)[0]
        logmodel.fit(X_train[:,features],Y_train)         
        predictions = logmodel.predict(X_test[:,features])
        score =  accuracy_score(Y_test,predictions)
        if max_score < score:
            max_score = score
            best_chromo = chromo

    return max_score, best_chromo                                

def de_crossover(parent1, parent2, probability):
    child = []
    parent_1, parent_2 = parent1.copy(), parent2.copy()
    chromo_len = parent_1.shape[0]

    for i in range(chromo_len):
        if random() < probability:
            child.append(parent_1[i])
        else:
            child.append(parent_2[i])
    
    child = np.array(child)
    features = np.where(child > 0)[0]
    non_features = np.where(child <= 0)[0]

    features_count = len(features)

    if len(features) > 100:
        to_remove = features_count - 100
        features = np.setdiff1d(features, sample(list(features), k=to_remove))
    else:
        to_add = 100 - features_count
        features = np.append(features, sample(list(non_features), k=to_add))
    features.sort()
    
    new_child = np.array([1 if i in features else 0 for i in range(chromo_len)])
    return new_child

def de_mutation(pop_after_fit, co_probability, n_parents):
    # getting the population size
    pop_size = len(pop_after_fit)
    
    # getting the length of the chromosome
    chromo_len = len(pop_after_fit[0])
    # print(chromo_len)

    # new variable for the mutated population
    pop_nextgen = []

    # looping throught all the parent chromos in population
    for target in range(pop_size):
        sample_space = list(range(pop_size))
        sample_space.remove(target)
        
        # random selection of target chromo, and 2 random chromos
        rv1, rv2, rv3 = sample(sample_space, k=3)
        
        target_vec = pop_after_fit[target].astype(np.float32)
        random_vec1 = pop_after_fit[rv1]
        random_vec2 = pop_after_fit[rv2]
        random_vec3 = pop_after_fit[rv3]

        # performing the DE mutation
        trail_vec = random_vec1 + (random_vec2-random_vec3)
        # print("x1", *random_vec1)
        # print("x2", *(random_vec2-random_vec3))
        # # print("x2", *scale_factor*(random_vec2-random_vec3))
        # print("u1", *trail_vec)
        # print(len(np.where(trail_vec > 0)[0]))
        
        features = np.where(trail_vec > 0)[0]
        non_features = np.where(trail_vec <= 0)[0]
        
        # there is randomization in this part, in future incase of any unexpected results, have to concentrate in this part
        if len(features) > 100:
            to_remove = len(features) - 100
            features = np.setdiff1d(features, sample(list(features), k=to_remove))
        elif len(features) < 100:
            to_add = 100 - len(features)
            features = np.append(features, sample(list(non_features), k=to_add))
        features.sort()
        
        trail_vec = np.array([1 if i in features else 0 for i in range(chromo_len)])
        trail_vec = trail_vec.astype(int)
        
        new_trail = de_crossover(target_vec, trail_vec, co_probability)   
        
        pop_nextgen.append(de_fitness(target_vec, new_trail)[1])

    return pop_nextgen

def evolution(size, features_count, chromo_size,
            n_parents,
            crossover_pb,
            n_gen):
    best_chromo= []
    best_score= []
    
    population_nextgen=generate_population(size, features_count, chromo_size)
    # scores, pop_after_fit = fitness_score(population_nextgen)
    # population_nextgen = pop_after_fit.copy()

    for i in range(n_gen):
        scores, pop_after_fit = fitness_score(population_nextgen.copy())
        best_chromo.append(pop_after_fit[0])
        best_score.append(scores[0])
        print('Best score in generation',i+1,':',scores[0], "feat_count:", np.where(pop_after_fit[0] != 0)[0].shape)
        
        population_nextgen = de_mutation(population_nextgen.copy(), crossover_pb, n_parents)
        
        print("Population size:", len(population_nextgen))
        
    return best_chromo,best_score



amazon = pd.read_csv("../dataset/amazon.csv")
frame = amazon.copy()
frame

X_bow, features = tokens_to_bow(frame.cmd, 0)
y_score = frame.score
all_terms = list(features)

X_train, X_test, Y_train, Y_test = split(X_bow, y_score)

all_models_score_table = acc_score(X_bow, y_score)
all_models_score_table

logmodel = RandomForestClassifier(n_estimators=200, random_state=0)

def run_n_evolution(n):
    result_n_runs = []
    for i in range(n):
        st = time.time()
        chromo_set_2, score_set_2 = evolution(
            size=100, 
            features_count=100,
            chromo_size=X_bow.shape[1],
            n_parents=80,
            crossover_pb=0.8,
            n_gen=100
        )
        et = time.time()
        result_n_runs.append((chromo_set_2, score_set_2, et-st))
    return result_n_runs