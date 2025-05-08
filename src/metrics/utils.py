import os
import pandas as pd
from tqdm import tqdm
from src.metrics.tensorboard import get_scalar_values

def get_best_params(folder, metrics, param, ascending=True, metrics_to_abs=[], print_all=False):
    df = pd.DataFrame(columns=["Data", param] + metrics + ["Epoch", "Time"])
    for data in tqdm(os.listdir(folder), leave=False):
        row = [data]
        sub_df = pd.DataFrame(columns=[param] + metrics + ["Epoch", "Time"])
        if param in os.listdir(os.path.join(folder, data)):
            for p in os.listdir(os.path.join(folder, data, param)):
                v = [p]
                # epochs = []
                # ind = 0
                for i, metric in enumerate(sub_df.columns[1:-2]):
                    path = os.path.join(folder, data, param, p, [k for k in os.listdir(os.path.join(folder, data, param, p)) if k.startswith("events")][0])
                    values = get_scalar_values(path, metric)
                    if i==0:
                        if len(values)>1:
                            ind = values["value"][1:].argmin()+1
                        else:
                            ind = 0
                    v.append(values["value"][ind])
                v.append(ind)
                times = list(values.wall_time)
                v.append((times[-1]-times[0])/(60))
                sub_df.loc[len(sub_df)] = v
            n_sub_df = sub_df.copy()
            for i in metrics_to_abs:
                n_sub_df[metrics[i]] = sub_df[metrics[i]].abs()
            sub_df = sub_df.reindex(n_sub_df.sort_values(by=metrics, ascending=ascending).index)
            row.extend(list(sub_df.iloc[0]))
            df.loc[len(df)] = row
        if print_all and len(sub_df)>0:
            print("\n", data)
            display(sub_df)
            print("\n")

    n_df = df.copy()
    for i in metrics_to_abs:
        n_df[metrics[i]] = df[metrics[i]].abs()
        df = df.reindex(n_df.sort_values(by=metrics, ascending=ascending).index)
    return df

def get_best_params_fitting(folder, metrics, param, ascending=True, metrics_to_abs=[], print_all=False, ind_metric_regress=0, ind_metrics_class=1):
    df = pd.DataFrame(columns=["Data", param] + metrics + ["Epoch regress", "Epoch class", "Time regress", "Time class"])
    for data in tqdm(os.listdir(folder), leave=False):
        row = [data]
        sub_df = pd.DataFrame(columns=[param] + metrics + ["Epoch regress", "Epoch class", "Time regress", "Time class"])
        if param in os.listdir(os.path.join(folder, data)):
            for p in os.listdir(os.path.join(folder, data, param)):
                v = [p]
                # epochs = []
                # ind = 0
                for i, metric in enumerate(sub_df.columns[1:-4]):
                    path = os.path.join(folder, data, param, p, [k for k in os.listdir(os.path.join(folder, data, param, p)) if k.startswith("events")][0])
                    values = get_scalar_values(path, metric)
                    if i==ind_metric_regress:
                        ind_regress = values["value"].argmin()
                        times_regress = list(values.wall_time)
                    elif i==ind_metrics_class:
                        ind_class = values["value"].argmin()
                        times_class = list(values.wall_time)
                    L = ["Loss/total/Train", "Loss/class/Train", "Acc/total/Train", "Acc/fix/Train", "Acc/blink/Train", "Loss/total/Val", "Loss/class/Val", "Acc/total/Val", "Acc/fix/Val", "Acc/blink/Val"]
                    if not metric in L:
                        v.append(values["value"][ind_regress])
                    else:
                        v.append(values["value"][ind_class])
                v.append(ind_regress)
                v.append(ind_class)
                v.append((times_regress[-1]-times_regress[0])/(60))
                v.append((times_class[-1]-times_class[0])/(60))
                sub_df.loc[len(sub_df)] = v
            n_sub_df = sub_df.copy()
            for i in metrics_to_abs:
                n_sub_df[metrics[i]] = sub_df[metrics[i]].abs()
            sub_df = sub_df.reindex(n_sub_df.sort_values(by=metrics, ascending=ascending).index)
            row.extend(list(sub_df.iloc[0]))
            df.loc[len(df)] = row
        if print_all and len(sub_df)>0:
            print("\n", data)
            display(sub_df)
            print("\n")

    n_df = df.copy()
    for i in metrics_to_abs:
        n_df[metrics[i]] = df[metrics[i]].abs()
        df = df.reindex(n_df.sort_values(by=metrics, ascending=ascending).index)
    return df

def get_autoencoder_metrics(folder, metrics=["Train_loss"], ascending=[True]):
    df = pd.DataFrame(columns=["Data", "Size"] + metrics + ["Epoch", "Time"])
    for data in tqdm(os.listdir(folder), leave=False):
        for size in os.listdir(os.path.join(folder, data)):
            sub_df = pd.DataFrame(columns=["Size"] + metrics + ["Epoch", "Time"])
            row = [data]
            v = [size.split("=")[1]]
            for i, metric in enumerate(sub_df.columns[1:-2]):
                path = os.path.join(folder, data, size, [k for k in os.listdir(os.path.join(folder, data, size)) if k.startswith("events")][0])
                values = get_scalar_values(path, metric)
                if i==0:
                    if len(values)>1:
                        ind = values["value"][1:].argmin()+1
                    else:
                        ind = 0
                v.append(values["value"][ind])
            v.append(ind+1)
            times = list(values.wall_time)
            v.append((times[-1]-times[0])/(60))
            sub_df.loc[len(sub_df)] = v
            n_sub_df = sub_df.copy()
            sub_df = sub_df.reindex(n_sub_df.sort_values(by=metrics, ascending=ascending).index)
            row.extend(list(sub_df.iloc[0]))
            df.loc[len(df)] = row

    n_df = df.copy()
    df = df.reindex(n_df.sort_values(by=metrics, ascending=ascending).index)
    return df