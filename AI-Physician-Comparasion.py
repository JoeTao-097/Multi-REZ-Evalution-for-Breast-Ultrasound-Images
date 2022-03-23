#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 14:13:28 2021

@author: joe
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix,roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score, f1_score
import seaborn as sns

"""
Load AI-Physician Comparasion result
"""
df_md = pd.read_csv('./model and physicians performance on AI-Physician Comparasion set/AI-Physician Comparasion result.csv')
def cm_metrics(y_true,y_pred):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            if y_pred[i] == 1:
                TP += 1
            else:
                FN += 1
        else:
            if y_pred[i] == 1:
                FP += 1
            else:
                TN += 1
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)
    return Precision, Recall, TPR, FPR

Experience_Age = ['senior-1','senior-2','junior-1','junior-2','entry-1','entry-2']
Color = ["orange","blue","green",'brown','cyan']     
Presision = []
Recall = []
TP_rate = []
FP_rate = []
for i in range(6,12):

    # calculate
    
    y_true = df_md['y_true']
    y_pred = df_md.iloc[:,i].astype(int)
    P, R, TPR, FPR = cm_metrics(y_true, y_pred)
    Presision.append(P)
    Recall.append(R)
    TP_rate.append(TPR)
    FP_rate.append(FPR)

# plot functions
def get_precision_recall(ax, y_true, y_pred, title, boostrap=5,name=None,color=None, plot=True):

    def delta_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        return h

    ap_score=[]
    for i in range(boostrap):
        pred_bt, y_bt = resample(y_pred, y_true)
        ap_score.append(average_precision_score(y_bt, pred_bt))

    AP = average_precision_score(y_true, y_pred)
    precision, recall, thresholds=precision_recall_curve(y_true, y_pred)

    if plot:
        delta = delta_confidence_interval(ap_score)

        sns.set_style('ticks')
    #    plt.figure()
        ax.plot(recall, precision, color=color, lw=2,
                 label='{}-AUC = {:.3f}, \n95% C.I. = [{:.3f}, {:.3f}]'.format(name, AP, AP-delta, AP+delta), alpha=.8)
        # for i in range(len(Experience_Age)):
        #     if i%2 == 0:
        #         ax.plot(Recall[i],Presision[i],'ro',color=Color[i], label=Experience_Age[i])
        #     else:
        #         ax.plot(Recall[i],Presision[i],'^',color=Color[i], label=Experience_Age[i])
        ax.set_xlabel('Recall', fontsize=16, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=16, fontweight='bold')
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='lower left')
        ax.grid()
    return thresholds

def get_auc(ax, y_true, y_score, title, name=None, color=None, plot=True):
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_true, y_score)
    auc_keras = auc(fpr_keras, tpr_keras)

    optimal_idx = np.argmax(tpr_keras - fpr_keras)
    optimal_threshold = thresholds_keras[optimal_idx]

    if plot:
        ci = get_CI(y_true, y_score)

        sns.set_style('ticks')
    #    plt.figure()
        ax.plot(fpr_keras, tpr_keras, color= color, lw=2,
                 label='{}-AUC = {:.3f}, \n95% C.I. = [{:.3f}, {:.3f}]'.format(name,auc_keras, ci[0], ci[1]), alpha=.8)
        ax.set_xlabel('1 - Specificity', fontsize=16, fontweight='bold')
        ax.set_ylabel('Sensitivity', fontsize=16, fontweight='bold')
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='lower right')
        ax.grid()
    return optimal_threshold

def get_CI(y_true, y_score, alpha=0.95):
    auc, auc_cov = delong_roc_variance(y_true, y_score)
    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    ci = stats.norm.ppf(lower_upper_q, loc=auc, scale=auc_std)
    ci[ci > 1] = 1

    print('AUC:', auc)
    print('AUC COV:', auc_cov)
    print('95% AUC CI:', ci)
    return ci

def delong_roc_variance(ground_truth, predictions, sample_weight=None):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
        ground_truth, sample_weight)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count, ordered_sample_weight)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov

def compute_ground_truth_statistics(ground_truth, sample_weight):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight

def fastDeLong(predictions_sorted_transposed, label_1_count, sample_weight):
    if sample_weight is None:
        return fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)
    else:
        return fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight)


def fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank_weight(positive_examples[r, :], sample_weight[:m])
        ty[r, :] = compute_midrank_weight(negative_examples[r, :], sample_weight[m:])
        tz[r, :] = compute_midrank_weight(predictions_sorted_transposed[r, :], sample_weight)
    total_positive_weights = sample_weight[:m].sum()
    total_negative_weights = sample_weight[m:].sum()
    pair_weights = np.dot(sample_weight[:m, np.newaxis], sample_weight[np.newaxis, m:])
    total_pair_weights = pair_weights.sum()
    aucs = (sample_weight[:m]*(tz[:, :m] - tx)).sum(axis=1) / total_pair_weights
    v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights
    v10 = 1. - (tz[:, m:] - ty[:, :]) / total_positive_weights
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def fastDeLong_no_weights(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating
              Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)

def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2
def plot_confusion_matrix(ax, cm, target_names, title='Confusion matrix', cmap=None, normalize=True, fontsize=16):
    """
    given a sklearn confusion matrix (cm), make a nice plot
    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix
    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']
    title:        the text to display at the top of the matrix
    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues
    normalize:    If False, plot the raw numbers
                  If True, plot the proportions
    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(target_names)
        ax.set_yticklabels(target_names, rotation=90)
        ax.xaxis.set_tick_params(labelsize=fontsize)
        ax.yaxis.set_tick_params(labelsize=fontsize)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title, fontsize=fontsize, fontweight='bold')
    ax.tick_params(labelsize=fontsize-3)

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            ax.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=fontsize)
        else:
            ax.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=fontsize)

    ax.set_ylabel('True label', fontsize=fontsize, fontweight='bold')
    ax.set_xlabel('Predicted label\naccuracy={:0.2f}; misclass={:0.2f}'.format(accuracy, misclass), fontsize=fontsize, fontweight='bold')







import scipy
from scipy import stats
from sklearn.utils import resample



fig, axes = plt.subplots(nrows=1, ncols=2)
fig.tight_layout(pad=2, w_pad=2.)
fig.set_figheight(12)
fig.set_figwidth(20)

colors = ["red","brown",'cyan',"green",'blue','purple']
curvename = df_md.columns.tolist()[1:6]
curve = curvename
# shapes = ['ro','^','ro','^','ro','^']
shapes = ['ro','ro','^','^','*','*']
# ROC-AUC
for i in range(5):
    y_true = df_md['y_true']
    y_pred = df_md[curve[i]]
    get_auc(axes[0], np.array(y_true), np.array(y_pred), 'Malignancy=0 vs 1',name= curvename[i],color=Color[i])
for i in range(6):
    print(i)
    axes[0].plot(FP_rate[i], TP_rate[i],shapes[i],color=colors[i], label=Experience_Age[i], ms=12)
axes[0].legend(fontsize=16, loc='lower right')
    
# ROC-PR

for i in range(5):
    y_true = df_md['y_true']
    y_pred = df_md[curve[i]]
    get_precision_recall(axes[1], np.array(y_true), np.array(y_pred), 'Malignancy=0 vs 1',name= curvename[i],color=Color[i])
for i in range(6):
    axes[1].plot(Recall[i], Presision[i],shapes[i],color=colors[i], label=Experience_Age[i],ms=12)
axes[1].legend(fontsize=16, loc='lower left')































