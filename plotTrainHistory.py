import json
import matplotlib.pyplot as plt
import os
import numpy as np


COLORS = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "black", "grey", "brown"]


def plotHistory(files, scalar=["loss", "val_loss"]):
    """ Plot tensorboard loss from json.
        Possible Scalars: val_loss, val_iou_score, val_f1-score, loss, iou_score, f1-score, lr

    Args:
        files (_type_): _description_
        scalar (list, optional): list of scalars to plot. Defaults to ["loss"].
    """
    scalar_data = {}
    max_len = 0
    for s in scalar: 
         scalar_data[s] = []
    for file in files:
        with open(file, "r") as f:
            data = json.load(f)
            for s in scalar:
                d = []
                for k,v in data[s].items():  
                    d.append(v)
                scalar_data[s] += d
                if len(scalar_data[s]) > max_len:
                    max_len = len(scalar_data[s])
    
    x = np.arange(max_len)
    for i,(k,v) in enumerate(scalar_data.items()):
        plt.plot(x, v, COLORS[i], label=k)
    plt.legend()
    plt.show()

files = [
    os.path.join(os.getcwd(), "tmp/train_20230615-014346/history.json"),
    os.path.join(os.getcwd(), "tmp/train_20230615-152539/history.json"),
]
plotHistory(files)
plotHistory(files, scalar=["iou_score", "val_iou_score"])
