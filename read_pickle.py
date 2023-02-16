import pickle
from tabulate import tabulate

pkl_path = 'pickles/'
pkl_file = pkl_path+'kbga/amazon/single_run_az_kbga-es50.pkl'

with open(pkl_file, 'rb') as rf:
    data = pickle.load(rf)
    rf.close()
#print(tabulate(data, headers='keys', tablefmt="simple_grid"))
print(*data[1])


# pkl_file = pkl_path+'pt_co_prob.pkl'
# with open(pkl_file, 'rb') as rf:
#     data = pickle.load(rf)
#     rf.close()
# print(tabulate(data, headers='keys', tablefmt="simple_grid"))
# 
# pkl_file = pkl_path+'pt_mt_rate.pkl'
# with open(pkl_file, 'rb') as rf:
#     data = pickle.load(rf)
#     rf.close()
# print(tabulate(data, headers='keys', tablefmt="simple_grid"))
# 
# pkl_file = pkl_path+'single_run_table.pkl'
# with open(pkl_file, 'rb') as rf:
#     data = pickle.load(rf)
#     rf.close()
# print(tabulate(data, headers='keys', tablefmt="simple_grid"))
# 
# pkl_file = pkl_path+'n_run_table.pkl'
# with open(pkl_file, 'rb') as rf:
#     data = pickle.load(rf)
#     rf.close()
# print(tabulate(data, headers='keys', tablefmt="simple_grid"))