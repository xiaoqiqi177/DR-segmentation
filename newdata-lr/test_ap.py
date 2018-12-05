"""
File: test_ap.py
Created by: Qiqi Xiao
Email: xiaoqiqi177<at>gmail<dot>com
"""

import os
from optparse import OptionParser
from sklearn.metrics import precision_recall_curve, average_precision_score
import glob
import numpy as np
import matplotlib.pyplot as plt
import config_test as config

titles = [config.LESION_NAME]
logdir = config.TEST_OUTPUT_DIR

def plot_precision_recall_all(predicted, gt):
    colors = ['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal']
    plt.figure(figsize=(7, 8))
    lines = []
    labels = []
    n_number = 1
    for i in range(n_number):
        precision, recall, _ = precision_recall_curve(gt[i], predicted[i])
        l, = plt.plot(recall, precision, color=colors[i], lw=2)
        ap = average_precision_score(gt[i], predicted[i])
        lines.append(l)
        labels.append('Precision-recall for {}: AP = {}'.format(titles[i], ap))
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
        lines.append(l)
    labels.append('iso-f1 curves')

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curves')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    plt.savefig(os.path.join(logdir, 'precision_recall.png'))

if __name__ == '__main__':
    soft_npy_paths = glob.glob(os.path.join(logdir, '*soft*.npy'))
    true_npy_paths = glob.glob(os.path.join(logdir, '*true*.npy'))
    soft_npy_paths.sort()
    true_npy_paths.sort()
    soft_masks_all = []
    true_masks_all = []
    for soft_npy_path, true_npy_path in zip(soft_npy_paths, true_npy_paths):
        soft_masks = np.load(soft_npy_path)
        true_masks = np.load(true_npy_path)
        soft_masks_all.extend(soft_masks)
        true_masks_all.extend(true_masks)
    soft_masks_all = np.array(soft_masks_all)
    true_masks_all = np.array(true_masks_all)
    
    needed_idxes = np.where(true_masks_all.sum(axis=(1,2,3)))
    soft_masks_all = soft_masks_all[needed_idxes]
    true_masks_all = true_masks_all[needed_idxes]
    
    predicted = np.transpose(soft_masks_all, (1, 0, 2, 3))
    predicted = predicted.round(2)
    gt = np.transpose(true_masks_all, (1, 0, 2, 3))
    predicted = np.reshape(predicted, (predicted.shape[0], -1))
    gt = np.reshape(gt, (gt.shape[0], -1))
    aps = []
    plot_precision_recall_all(predicted, gt)
