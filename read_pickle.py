import pickle
from tabulate import tabulate
import seaborn as sns

pkl_path = 'pickles/'
pkl_file = pkl_path+'kbga/30runs/30_run_az_kbga_vp_ts.pkl'

with open(pkl_file, 'rb') as rf:
    data = pickle.load(rf)
    rf.close()
#print(tabulate(data, headers='keys', tablefmt="simple_grid"))
print(data)


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