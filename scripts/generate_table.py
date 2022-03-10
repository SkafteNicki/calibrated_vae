from prettytable import PrettyTable
import numpy as np

data = {}
with open('results/uci_benchmark_scores.txt', 'r') as file:
    for i, line in enumerate(file):
        if i == 0:
            continue
        out = line.split(', ')
        dataset, model, ensemble_size = out[0], out[1], out[2]
        if dataset not in data:
            data[dataset] = {}
        
        time = out[3:23]; time[0] = time[0][1:]; time[-1] = time[-1][:-1]
        rmse = out[24:43]; rmse[0] = rmse[0][1:]; rmse[-1] = rmse[-1][:-1]
        nll = out[43:]; nll[0] = nll[0][1:]; nll[-1] = nll[-1][:-1]

        data[dataset][model] = {
            'time': [float(x) for x in time],
            'rmse': [float(x) for x in time],
            'nll': [float(x) for x in time],
        }

t = PrettyTable(['Dataset', 'Model', 'Time Mean', 'Time Std', 'Rmse Mean', 'Rmse Std', 'NLL Mean', 'NLL Std'])
for dataset, values in data.items():
    for model, value in values.items():
        time = value['time']
        rmse = value['rmse']
        nll = value['nll']
        t.add_row([dataset, model, np.mean(time), np.std(time), np.mean(rmse), np.std(rmse), np.mean(nll), np.std(nll)])




