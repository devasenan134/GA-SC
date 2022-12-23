import pickle
import time
from n_run_evolutions import evolution
from n_run_evolutions import X_bow, y_score, X_train, X_test, Y_train, Y_test


st = time.time()
chromo_set_1, score_set_1 = evolution(
    X_bow,
    y_score,
    size=80, 
    feat_count=100,
    n_feat=X_bow.shape[1],
    n_parents=64,
    crossover_pb=0.5,
    mutation_pb=0.04,
    mutation_rate=0.2,
    n_gen=30,
    X_train = X_train,
    X_test = X_test,
    Y_train = Y_train,
    Y_test = Y_test
)
et = time.time()
print('Evolution time 1: ', et-st)

st = time.time()
chromo_set_2, score_set_2 = evolution(
    X_bow,
    y_score,
    size=80, 
    feat_count=100,
    n_feat=X_bow.shape[1],
    n_parents=64,
    crossover_pb=0.3,
    mutation_pb=0.04,
    mutation_rate=0.2,
    n_gen=30,
    X_train = X_train,
    X_test = X_test,
    Y_train = Y_train,
    Y_test = Y_test
)
et = time.time()
print('Evolution time 1: ', et-st)


st = time.time()
chromo_set_3, score_set_3 = evolution(
    X_bow,
    y_score,
    size=80, 
    feat_count=100,
    n_feat=X_bow.shape[1],
    n_parents=64,
    crossover_pb=0.7,
    mutation_pb=0.04,
    mutation_rate=0.2,
    n_gen=30,
    X_train = X_train,
    X_test = X_test,
    Y_train = Y_train,
    Y_test = Y_test
)
et = time.time()
print('Evolution time 1: ', et-st)


with open('evolution_sr_1.pkl', 'wb') as wf:
    pickle.dump([chromo_set_1, score_set_1], wf)

with open('evolution_sr_2.pkl', 'wb') as wf:
    pickle.dump([chromo_set_2, score_set_2], wf)

with open('evolution_sr_3.pkl', 'wb') as wf:
    pickle.dump([chromo_set_3, score_set_3], wf)