import numpy as np
import pandas as pd
import os
from sklearn import metrics

import seaborn as sns
import torch
import pickle
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
#  Reliability Diagrams

f = open('demo.pckl', 'rb')
output, _, label = pickle.load(f)
softmax = F.softmax(output, dim=1).cpu().data
label = label.cpu().data



def calc_ece(softmax, label, bins=15):
    '''
    Calculates the Expected Calibration Error (ECE).
    
    input:
    softmax: list or torch tensor, shape (n, m), n is the number of samples, m is the number of classes
    label: list or torch tensor, shape (n,)
    '''   
    # 分箱 
    bin_boundaries = torch.linspace(0, 1, bins + 1)
    # 上下界
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # 转化为torch tensor
    softmax = torch.tensor(np.array(softmax))
    labels = torch.tensor(label)
    
    # get the max prob and the pred
    msp, predictions = torch.max(softmax, 1)
    
    correctness = predictions.eq(labels.long())
    # don't need the long() is ok
    # correctness = predictions.eq(labels)
    
    # save the ECE
    ece = torch.zeros(1)
    accuracy_in_bin = []
    avg_confidence_in_bin = []
    # calculate bin by bin
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        
        # define the in_bin data
        in_bin = msp.gt(bin_lower.item()) * msp.le(bin_upper.item())
        
        prop_in_bin = in_bin.float().mean()

        avg_confidence_item = msp[in_bin].mean()
        accuracy_item = correctness[in_bin].float().mean()
        
        accuracy_in_bin.append(accuracy_item)
        avg_confidence_in_bin.append(avg_confidence_item)
        if prop_in_bin.item() > 0.0:
            ece += torch.abs(avg_confidence_item - accuracy_item) * prop_in_bin
    print("ECE {0:.2f} ".format(ece.item()*100))
    
    
    return ece.item()
    

def draw_reliability_diagram(softmax, labels, bins=15):
    
    '''
    Plots a reliability diagram.

    input:
    softmax: list or torch tensor, shape (n, m), n is the number of samples, m is the number of classes
    labels: list or torch tensor, shape (n,)
    '''       
    
    msp, predictions = torch.max(softmax, 1)

    correctness = predictions.eq(labels.long())
    
    # bin_edges = np.linspace(0., 1. + 1e-8, bins + 1)
    # bin_lowers = bin_edges[:-1] 
    # bin_uppers = bin_edges[1:]

    fig, ax = plt.subplots(figsize=(8, 6))
    prob_true, prob_pred = calibration_curve(correctness, msp, n_bins=15, strategy='uniform')
    ax.plot(prob_pred, prob_true, 's-', label='Calibration curve')
    
    ax.plot([0, 1], [0, 1], 'k--')
    # accuracy_in_bin = [0 if torch.isnan(x)  else x for x in accuracy_in_bin]
    
    # ax.bar(bin_lowers, accuracy_in_bin, width=1.0/bins, alpha=0.3, color='blue', edgecolor='black', label='Outputs')
    # ax.bar(bin_lowers, bin_counts / float(np.sum(bin_counts)), width=1.0/bins, alpha=0.3, color='blue', edgecolor='black', label='Outputs')
    # prob_true, prob_pred = calibration_curve(correctness, msp, n_bins=15, strategy='uniform')

    # ax2 = ax.twinx()
    # ax2.set_ylabel('Fraction of Positives')
    # Finalize the plot
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title('Reliability Diagram with Distplots')
    ax.legend(loc='upper left')
    ax.grid(True)
    
    plt.show()

    return fig

ece, fig = calc_ece(softmax, label, draw=True)
print(ece)
# plt.show()