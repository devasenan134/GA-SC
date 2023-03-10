# -*- coding: utf-8 -*-
"""n_run_evolutions.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qAiKoxiD3RWPAGPTkZ-BaGu5DnYyY0eV
"""

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

"""# **1. Pre-Processing**
1. Tokenization
2. Stemming/lemmatization
3. Bow/TF-IDF 
"""

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

"""### 3.1. BOW"""
def custom_tokens_bow(corpus_tokens):
    tokens = re_lemmatizor(corpus_tokens)
    vocab = []
    for sentence in tokens:
        vocab.extend(sentence.split())
    
    features = set(vocab)
    bow = []
    for sentence in tokens:
        sent_dict = {}
        for word in sentence.split():
            sent_dict[word] = sent_dict.get(word, 0) + 1
        bow.append([sent_dict[feature] if feature in sent_dict.keys() else 0 for feature in features ])
    #print("Total Vocab Count:", len(features))
    return np.array(bow), features

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

"""### 3.2. TF-IDF"""

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
import random
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

def initial_population_term_selection_tf(tf_terms, tf_threshold):
    selected_indexes = []
    selected_terms = []
    tf_dict = dict(tf_terms)
    for word, tf in tf_dict.items():
        if tf >= tf_threshold:
            selected_terms.append(word)
            selected_indexes.append(all_terms.index(word))
    return selected_indexes, selected_terms

def initial_population_term_selection_idf(idf, idf_threshold):
    selected_indexes = []
    selected_terms = []
    idf_dict = dict(idf)
    for word, idf in idf_dict.items():
        if idf >= idf_threshold:
            selected_terms.append(word)
            selected_indexes.append(all_terms.index(word))
    return selected_indexes, selected_terms


def plot(score,x,y,c = "b"):
    gen = [1,2,3,4,5]
    plt.figure(figsize=(6,4))
    ax = sns.pointplot(x=gen, y=score,color = c )
    ax.set(xlabel="Generation", ylabel="Accuracy")
    ax.set(ylim=(x,y))

def generate_chromo(selected_indexes, features_count, chromo_size):
    features = sample(selected_indexes, k=features_count)
    features.sort()
    chromo = [1 if i in features else 0 for i in range(chromo_size)]
    return np.array(chromo)

def generate_population(size, features_count, chromo_size, selected_indexes):
    return [generate_chromo(selected_indexes, features_count, chromo_size) for _ in range(size)]

def single_point_crossover1(pop_after_sel, n_parents):
    shuffle(list(pop_after_sel))
    pop_nextgen = list(pop_after_sel)
    length = len(pop_nextgen)
    chromo_l = len(pop_nextgen[0])

    tf_idf_sent_score = dict(term_frequency_inverse_document_frequency(pop_after_sel))
    pop_sorted_tfidf = np.array(sorted(tf_idf_sent_score.items(), key=lambda x: x[1]))[:, 0]
    
    mid = len(pop_sorted_tfidf)//2
    pop_1 = pop_sorted_tfidf[:mid]
    pop_2 = pop_sorted_tfidf[mid:]

    for i in range(0, mid):
        parent_1, parent_2 = pop_after_sel[int(pop_1[i])], pop_after_sel[int(pop_2[i])]
        p1_features = list(np.where(np.array(parent_1) != 0)[0])
        p2_features = list(np.where(np.array(parent_2) != 0)[0])

        k = randint(0,len(p1_features))  # crossover_point
        c1_features = p1_features[:k] + p2_features[k:]
        c2_features = p2_features[:k] + p1_features[k:]
        c1_dup = list(set([i for i in c1_features if c1_features.count(i) > 1]))
        c2_dup = list(set([i for i in c2_features if c2_features.count(i) > 1]))
            # print("duplicates:", c1_dup, c2_dup)
        
        if len(c1_dup) > 0:
            sample_pop1 = [i for i in p1_features if i not in c1_features]
            k1 = sample(sample_pop1, k=len(c1_dup))
            for i in c1_dup:
                c1_features.remove(i)
            c1_features.extend(k1)
        elif len(c2_dup) > 0:
            sample_pop2 = [i for i in p2_features if i not in c2_features]
            k2 = sample(sample_pop2, k=len(c2_dup))
            for i in c2_dup:
                c2_features.remove(i)
            c2_features.extend(k2)
        new_child_1 = np.array([1 if i in c1_features else 0 for i in range(chromo_l)])
        new_child_2 = np.array([1 if i in c2_features else 0 for i in range(chromo_l)])
        pop_nextgen.append(new_child_1)
        pop_nextgen.append(new_child_2)
    
    _, pop_nextgen = fitness_score(pop_nextgen)
    return pop_nextgen[:n_parents]

def bit_flip_mutation1(pop_after_cross, mutation_rate1, mutation_rate2, features_count, n_parents):   
    range1 = int(mutation_rate1*features_count)
    range2 = int(mutation_rate2*features_count)
    pop_next_gen = list(pop_after_cross)
    # print(range1, range2)
    tf_idf_sent_score = dict(term_frequency_inverse_document_frequency(pop_after_cross))
    pop_sorted_tfidf = np.array(sorted(tf_idf_sent_score.items(), key=lambda x: x[1]))[:, 0]
    
    mid = len(pop_sorted_tfidf)//2+1
    for n in pop_sorted_tfidf:
        if mid >= 0:
            mutation_range = range1
        else:
            mutation_range = range2
            
        chromo = pop_after_cross[int(n)]
        features = list(np.where(chromo != 0)[0])
        non_features = list(np.setdiff1d(np.array(range(chromo.shape[0])), features))

        rand_posi = []

        features_pos = sample(features, k=mutation_range)
        non_features_pos = sample(non_features, k=mutation_range)
        rand_posi.extend(features_pos)
        rand_posi.extend(non_features_pos)
        for j in rand_posi:
            chromo[j] = abs(chromo[j] - 1)
        
        pop_next_gen.append(chromo)
        mid -= 1

    _, pop_next_gen = fitness_score(pop_next_gen)
    return pop_next_gen[:n_parents+20]

def population_selection(pop_after_fit, n_parents):
    population_nextgen = []
    for i in range(n_parents):
        population_nextgen.append(pop_after_fit[i])
    return population_nextgen

def fitness_score(population):
    scores = []
    for chromosome in population:
        indexes = np.where(chromosome!=0)[0]
        logmodel.fit(X_train[:,indexes],Y_train)    
        predictions = logmodel.predict(X_test[:,indexes])
        scores.append(accuracy_score(Y_test,predictions))
    scores, population = np.array(scores), np.array(population)
    inds = np.argsort(scores)
    return list(scores[inds][::-1]), list(population[inds,:][::-1])

def term_frequency(population):
    tf_sent = []
    tf_dict = {}
    total_no_terms = len(population)
    for chromosome in population:
        chromo_tf = []
        indexes = np.where(chromosome!=0)
        for i in indexes[0]:
            chromo_tf.append(chromosome[i]/total_no_terms)
            tf_dict[all_terms[i]] = tf_dict.get(all_terms[i], 0) + (chromosome[i]/total_no_terms)
        tf_sent.append(chromo_tf)
    
    tf_terms = sorted(tf_dict.items(), key=lambda x: x[1], reverse=True)
    return tf_sent, tf_terms

def inverse_document_frequency(population):
    idf = {}
    terms = np.array(list(all_terms))
    no_documents = len(population)
    for i in range(len(all_terms)):
        k = 0
        for chromosome in population:
            indexes = np.where(chromosome!=0)
            if terms[i] in terms[indexes]:
                k += 1
        idf[terms[i]] = np.log10(no_documents/k)
    idf = sorted(idf.items(), key=lambda x: x[1], reverse=True)
    return idf

def term_frequency_inverse_document_freqency(population):
    tf_sent, tf_terms = term_frequency(population)
    tf_idf = {}
    idf_dict = dict(idf)
    for i in range(len(population)):
        tf_idf_sent = []
        indexes = np.where(population[i] != 0)[0]
        for j in range(len(indexes)):
            idf_term = idf_dict[all_terms[indexes[j]]]
            tf = tf_sent[i][j]
            tf_idf_sent.append(tf*idf_term)
        tf_idf[i] = sum(tf_idf_sent)/len(indexes)
    tf_idf = sorted(tf_idf.items(), key=lambda x: x[1], reverse=True)
    return tf_idf

def evolution(size, features_count, chromo_size,
            n_parents,
            crossover_pb, mutation_pb,
            mutation_rate1,
            mutation_rate2,
            n_gen,
            idf, idf_threshold,
            tf, tf_threshold,
            tfidf_threshold):
    best_chromo= []
    best_score= []
    
    
    selected_indexes, selected_terms = initial_population_term_selection_idf(idf, idf_threshold)
    # selected_indexes, selected_terms = initial_population_term_selection_tf(tf, tf_threshold)

    population_nextgen=generate_population(size, features_count, chromo_size, selected_indexes)
    for i in range(n_gen):
        scores, pop_after_fit = fitness_score(population_nextgen)
        best_chromo.append(pop_after_fit[0])
        best_score.append(scores[0])
        print('Best score in generation',i+1,':',scores[0], "feat_count:", np.where(pop_after_fit[0] != 0)[0].shape)

        pop_after_sel = population_selection(pop_after_fit, n_parents)
        # sc_sel, pop_sel = fitness_score(pop_after_sel)
        # print('Best score in generation',i+1,':',sc_sel[0], "feat_count:", np.where(pop_sel[0] != 0)[0].shape)

        pop_after_cross = single_point_crossover1(pop_after_sel, n_parents)
        # sc_co, pop_co = fitness_score(pop_after_cross)
        # print('Best score in generation',i+1,':',sc_co[0], "feat_count:", np.where(pop_co[0] != 0)[0].shape)

        population_nextgen = bit_flip_mutation1(pop_after_cross, mutation_rate1, mutation_rate2, features_count, n_parents)
        # sc_mu, pop_mu = fitness_score(population_nextgen)
        # print('Best score in generation',i+1,':',sc_mu[0], "feat_count:", np.where(pop_mu[0] != 0)[0].shape)

        # # new next gen population will have the evolved population + the initial population after fitness_score
        # population_nextgen += pop_after_sel
        # _, population_new_nextgen = fitness_score(population_nextgen)
        # print('Best score in generation',i+1,':',_[0], "feat_count:", np.where(population_new_nextgen[0] != 0)[0].shape)
        # population_nextgen = population_selection(population_new_nextgen, n_parents)
        print(len(population_nextgen))
        
    return best_chromo,best_score

"""### Choosing the best classifier and starting evolution"""
logmodel = RandomForestClassifier(n_estimators=200, random_state=0)



amazon = pd.read_csv("../dataset/amazon.csv", encoding='latin1')
amazon

frame = amazon.copy()

X_bow, features = tokens_to_bow(frame.cmd, 0)
y_score = frame.score
all_terms = list(features)

idf = inverse_document_frequency(X_bow)
tf_sent, tf_terms = term_frequency(X_bow)


X_train, X_test, Y_train, Y_test = split(X_bow, y_score)

"""### Compare models without GA"""
all_models_score_table = acc_score(X_bow, y_score)
all_models_score_table
