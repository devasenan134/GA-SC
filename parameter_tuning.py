import pickle
import time
import numpy as np
from tabulate import tabulate

from n_run_evolutions import evolution
from n_run_evolutions import X_bow, y_score, X_train, X_test, Y_train, Y_test
from n_run_evolutions import logmodel, X_train, Y_train, X_test, Y_test, accuracy_score, all_models_score_table


def predicted_sentiment_ratio(y_test, predictions):
    positive_percent = np.count_nonzero(predictions==1)*100//len(predictions)
    negative_percent = np.count_nonzero(predictions==0)*100//len(predictions)
    # print(f"Y_test: pos/neg percentage ~ {np.count_nonzero(y_test==1)*100//len(y_test)}, {np.count_nonzero(y_test==0)*100//len(y_test)}")
    return positive_percent, negative_percent

def test_accuracy(selected_genes):
    st = time.time()
    logmodel.fit(X_train[:, selected_genes], Y_train)
    et = time.time()
    predictions = logmodel.predict(X_test[:, selected_genes])
    ratio = predicted_sentiment_ratio(Y_test, predictions)
    return accuracy_score(Y_test, predictions), et-st, ratio

def pick_top_n_genes_n_run(genes, chromo_set, n=1):
    common_index = dict()
    chromo_len_in_each_gen = []
    for chromo in chromo_set:
        if n==1:
            chromo_len_in_each_gen.append(np.count_nonzero(chromo == 1))
            for i in range(len(chromo)):
                if chromo[i] == 1:
                    common_index[i] = common_index.setdefault(i, 0) + 1
        else:
            best_gen = np.array(chromo[:, -1])
            chromo_len_in_each_gen.append(np.count_nonzero(best_gen[0] == 1))
            for i in range(len(best_gen[0])):
                if best_gen[0][i] == 1:
                    common_index[i] = common_index.setdefault(i, 0) + 1

    if genes == -1:
        genes = len(common_index)

    sorted_common_index = np.array(sorted(common_index.items(), key=lambda x: x[1], reverse=True)[:genes])
    return sorted_common_index[:, 0], len(sorted_common_index), np.array(chromo_len_in_each_gen)


accuracy = []
time_to_train = []
sentiment_ratio = []
prob = []

for i in range(10):
    prob.append((1+i)*0.1)
    
    st = time.time()
    print((1+i)*0.1)
    chromo_set, score_set = evolution(
        X_bow,
        y_score,
        size=80, 
        feat_count=100,
        n_feat=X_bow.shape[1],
        n_parents=64,
        crossover_pb=0.8,
        mutation_pb=0.05,
        mutation_rate=(1+i)*0.1,
        n_gen=30,
        X_train = X_train,
        X_test = X_test,
        Y_train = Y_train,
        Y_test = Y_test
    )
    et = time.time()
    print('Evolution time 1: ', et-st)
    
    top_n_genes, common_count, chromo_len = pick_top_n_genes_n_run(150, chromo_set)
    acc, ttt, sr = test_accuracy(top_n_genes)
    accuracy.append(acc)
    time_to_train.append(ttt)
    sentiment_ratio.append(sr)

table_data = {'Mutation rate': prob,
            'Accuracy': accuracy,
            'Time Taken to Train': time_to_train,
            'Sentiment Ratio(p/n)': sentiment_ratio
            }

print('Actual chromosome length in X_Test: ', X_test.shape[1])
print('RandomForest with full length -', all_models_score_table.iloc[0, 1:])
print('Total Common genes count ~(-1): ', common_count)
print(chromo_len)
print(tabulate(table_data, headers='keys', tablefmt="simple_grid"))

with open('pt_mt_rate.pkl', 'wb') as wf:
        pickle.dump(table_data, wf)
