import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle 
from tabulate import tabulate

ga_path = '../pickles/ga/30runs/'
kbga_path = '../pickles/kbga/30runs/'

ga_files = ['30_run_az_ga_ts_pt.pkl', 
            '30_run_imdb_ga_ts_pt.pkl',
            '30_run_yelp_ga1_ts_pt.pkl',]

kbga_files = ['30_run_az_kbga_vp_ts.pkl',
              '30_run_imdb_kbga_vp_ts.pkl',
              '30_run_yelp_kbga_vp_ts.pkl',]


def average_best_fitness_value(az, imdb, yelp):
    cols = ["gen_"+str(i) for i in [1, 25, 50, 75, 100]]
    index_col = ["Amazon", "IMDB", "Yelp", "Average"]
    
    data = [
        az.iloc[-1, [0, 24, 49, 74, 99]],
        imdb.iloc[-1, [0, 24, 49, 74, 99]],
        yelp.iloc[-1, [0, 24, 49, 74, 99]],
        ]
    
    abfv = pd.DataFrame(data, columns=cols)
    abfv = abfv.append(abfv.mean(), ignore_index=True)

    abfv.index = index_col
    abfv.index.name = "Datasets"
    
    return abfv

def average_best_of_generation(az, imdb, yelp):
    cols = ["1-25", "26-50", "51-75", "76-100"]
    index_col = ["Amazon", "IMDB", "Yelp", "Average"]
    
    data = [
        [az.iloc[-1, i-25:i].mean() for i in range(25, 101, 25)],
        [imdb.iloc[-1, i-25:i].mean() for i in range(25, 101, 25)],
        [yelp.iloc[-1, i-25:i].mean() for i in range(25, 101, 25)],
    ]

    abog = pd.DataFrame(data, columns=cols)
    abog = abog.append(abog.mean(), ignore_index=True)

    abog.index = index_col
    abog.index.name = "Datasets"
    return abog

def optimization_accuracy(az, imdb, yelp):
    cols = ["gen_"+str(i) for i in [1, 25, 50, 75, 100]]
    index_col = ["Amazon", "IMDB", "Yelp", "Average"]
    
    data = [
        list(map(lambda i: (i-az_mins)/(az_maxs-az_mins), az.iloc[-1, [0, 24, 49, 74, 99]])),
        list(map(lambda i: (i-imdb_mins)/(imdb_maxs-imdb_mins), imdb.iloc[-1, [0, 24, 49, 74, 99]])),
        list(map(lambda i: (i-yelp_mins)/(yelp_maxs-yelp_mins), yelp.iloc[-1, [0, 24, 49, 74, 99]])),
    ]

    oa = pd.DataFrame(data, columns=cols)
    oa = oa.append(oa.mean(), ignore_index=True)

    oa.index = index_col
    oa.index.name = "Datasets"
    return oa


def evolutionary_leap(base, runs):
    leap = [[0 for i in range(runs)]]
    for i in range(1, 100):
        # print(ga_az.iloc[:, i] - ga_az.iloc[:, i-1])
        leap.append(list(map(lambda x: 1 if x != 0 else 0, base.iloc[:-1, i] - base.iloc[:-1, i-1])))
    leap = np.transpose(leap)

    indexes = ["run_"+str(i) for i in range(1, runs+1)]
    leap_df = pd.DataFrame(leap, columns=base.columns[:-1])
    leap_df.index = indexes

    data = {
        "gen_25": leap_df.iloc[:, :25].sum(axis=1),
        "gen_50": leap_df.iloc[:, 25:50].sum(axis=1),
        "gen_75": leap_df.iloc[:, 50:75].sum(axis=1),
        "gen_100": leap_df.iloc[:, 75:].sum(axis=1)
    }
    
    leap_count = pd.DataFrame(data)
    leap_count = leap_count.append(leap_count.mean(), ignore_index=True)
    
    indexes = ["run_"+str(i) for i in range(1, runs+1)] + ['Average']
    leap_count.index = indexes
    leap_count.index.name = 'runs'
    
    return leap_count

def likelihood_of_evolution_leap(az, imdb, yelp, runs):
    cols = ["gen_"+str(i) for i in [25, 50, 75, 100]]
    index_col = ["Amazon", "IMDB", "Yelp", "Average"]

    az_leaps = evolutionary_leap(az, runs)
    imdb_leaps = evolutionary_leap(imdb, runs)
    yelp_leaps = evolutionary_leap(yelp, runs)

    data = [
        az_leaps.iloc[-1]/runs,
        imdb_leaps.iloc[-1]/runs,
        yelp_leaps.iloc[-1]/runs,
    ]

    el = pd.DataFrame(data, columns=cols)
    el = el.append(el.mean(), ignore_index=True)

    el.index = index_col
    el.index.name = "Datasets"

    return el

def probability_of_convergence(az, imdb, yelp, success_thresh, runs):
    az_count = 0
    imdb_count = 0
    yelp_count = 0

    for i in range(runs):
        if az.iloc[i, -2] >= success_thresh:
            az_count += 1
        if imdb.iloc[i, -2] >= success_thresh:
            imdb_count += 1
        if yelp.iloc[i, -2] >= success_thresh:
            yelp_count += 1

    cols = ["P"]
    index_col = ["Amazon", "IMDB", "Yelp", "Average"]

    data = [
        az_count/runs,
        imdb_count/runs,
        yelp_count/runs
    ]

    pc = pd.DataFrame(data, columns=cols)
    pc = pc.append(pc.mean(), ignore_index=True)

    pc.index = index_col
    pc.index.name = "Datasets"

    return pc

def function_evaluations(base, success_thresh, runs):
    evolutions = 0

    mask = base.iloc[:, :-1] >= success_thresh
    for i in range(runs):
        try:
            # print(base[mask].iloc[i].dropna())
            evolutions += int(base[mask].iloc[i].dropna().index[0].split("_")[1]) 
        except:
            evolutions += 0

    return evolutions

def new_function_evaluations(base, runs):
    reached_sat_at_gen = 0
    for i in range(runs):
        highest_acc = base.iloc[i, -2]
        print(highest_acc)
        tmp = np.where(base.iloc[i, :-1] == highest_acc)[0][0]
        # print(tmp)
        reached_sat_at_gen += np.where(base.iloc[i, :-1] == highest_acc)[0][0]
    return reached_sat_at_gen//30

def average_no_of_function_evaluations(az, imdb, yelp, runs):
    az_eval = new_function_evaluations(az, runs)
    imdb_eval = new_function_evaluations(imdb, runs)
    yelp_eval = new_function_evaluations(yelp, runs)

    cols = ["AFES"]
    index_col = ["Amazon", "IMDB", "Yelp", "Average"]

    data = [
        az_eval,
        imdb_eval,
        yelp_eval
    ]

    afes = pd.DataFrame(data, columns=cols)
    afes = afes.append(afes.mean(), ignore_index=True)

    afes.index = index_col
    afes.index.name = "Datasets"

    return afes

def successful_performance(az, imdb, yelp, success_thresh, runs):
    afes = average_no_of_function_evaluations(az, imdb, yelp, runs)
    p = probability_of_convergence(az, imdb, yelp, success_thresh, runs)

    cols = ["SP"]
    index_col = ["Amazon", "IMDB", "Yelp", "Average"]

    data = [
        afes.iloc[0, 0]/p.iloc[0, 0],
        afes.iloc[1, 0]/p.iloc[1, 0],
        afes.iloc[2, 0]/p.iloc[2, 0]
    ]

    sp = pd.DataFrame(data, columns=cols)
    sp = sp.append(sp.mean(), ignore_index=True)

    sp.index = index_col
    sp.index.name = "Datasets"

    return sp

def tabulate_runs(save_path, runs):
    with open(save_path, 'rb') as gf:
        data = pickle.load(gf)
        chromos = []
        scores = []
        exec_time = []
        for run in data:
            chromos.append(run[0])
            scores.append(run[1])
            exec_time.append(run[2])

    df = pd.DataFrame()
    cols = ["gen_" + str(i) for i in range(1, 101)]
    df[cols] = pd.DataFrame(scores)
    
    df = pd.concat([df, pd.DataFrame({'exec_time': exec_time})], axis=1)
    # df = pd.concat([df, pd.DataFrame([df.mean().tolist()], columns=cols+["exec_time"])], axis=0, ignore_index=True)
    df = df.append(df.mean(), ignore_index=True)

    indexes = ["run_"+str(i) for i in range(1, runs+1)] + ['Average']
    df.index = indexes
    df.index.name = 'runs'
    
    return df




# data variables
runs = 30
ga_az = tabulate_runs(ga_path+ga_files[0], runs)
ga_imdb = tabulate_runs(ga_path+ga_files[1], runs)
ga_yelp = tabulate_runs(ga_path+ga_files[2], runs)

kbga_az = tabulate_runs(kbga_path+kbga_files[0], runs)
kbga_imdb = tabulate_runs(kbga_path+kbga_files[1], runs)
kbga_yelp = tabulate_runs(kbga_path+kbga_files[2], runs)

# base_ga = pd.concat([ga_az.iloc[-1, :], ga_imdb.iloc[-1, :], ga_yelp.iloc[-1, :]], axis=1)
# base_ga.columns = ['Amazon', "IMDB", "Yelp"]
# base_ga

az_mins = min(ga_az.min().to_list())
az_maxs = max(kbga_az.max().to_list())
imdb_mins = min(ga_imdb.min().to_list())
imdb_maxs = max(kbga_imdb.max().to_list())
yelp_mins = min(ga_yelp.min().to_list())
yelp_maxs = max(kbga_yelp.max().to_list())
print(az_mins, az_maxs)
print(imdb_mins, imdb_maxs)
print(yelp_mins, yelp_maxs)

