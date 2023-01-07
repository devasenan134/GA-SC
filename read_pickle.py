import pickle
from tabulate import tabulate



pkl_file = 'pt_mt_prob.pkl'
with open(pkl_file, 'rb') as rf:
    data = pickle.load(rf)
    rf.close()
print(tabulate(data, headers='keys', tablefmt="simple_grid"))

pkl_file = 'pt_co_prob.pkl'
with open(pkl_file, 'rb') as rf:
    data = pickle.load(rf)
    rf.close()
print(tabulate(data, headers='keys', tablefmt="simple_grid"))

pkl_file = 'pt_mt_rate.pkl'
with open(pkl_file, 'rb') as rf:
    data = pickle.load(rf)
    rf.close()
print(tabulate(data, headers='keys', tablefmt="simple_grid"))