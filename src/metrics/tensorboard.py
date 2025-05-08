import pandas as pd
from numpy import ndarray
from tensorboard.backend.event_processing import event_accumulator

def get_Tags(tfevents_filepath):
    accumulator = event_accumulator.EventAccumulator(tfevents_filepath).Reload()
    return accumulator.Tags()

def get_scalar_values(tfevents_filepath, tag):
    accumulator = event_accumulator.EventAccumulator(tfevents_filepath).Reload()
    return pd.DataFrame(accumulator.Scalars(tag))

def add_metrics_to_writer(metrics, epoch, writer):
    for i in metrics.keys():
        if isinstance(metrics[i], ndarray):
            writer.add_histogram(i, metrics[i], epoch, "auto")
        else:
            writer.add_scalar(i,metrics[i], epoch)