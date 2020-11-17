"""
#################################
# plot functions for visualization
#################################
"""
#########################################################
# import libraries

import random
import pickle
import itertools
import numpy as np
from skimage.io import imshow
import matplotlib.pyplot as plt


#########################################################
# Function definition

def plot_training(result, type_model, layers_len):
    (fig, ax) = plt.subplots(2, 1, figsize=(13, 13))
    epochs = len(result.history['accuracy'])
    ax[0].set_title("Loss", fontsize=14, fontweight='bold')
    ax[0].set_xlabel("Epoch #", fontsize=14, fontweight="bold")
    ax[0].set_ylabel("Loss", fontsize=14, fontweight="bold")
    ax[0].plot(np.arange(1, epochs+1), result.history['loss'], label='Loss', linewidth=2.5, linestyle='-', marker='o',
               markersize='10', color='red')
    ax[0].plot(np.arange(1, epochs+1), result.history['val_loss'], label='Validation_loss', linewidth=2.5, marker='x',
               linestyle='--', markersize='10', color='blue')
    ax[0].grid(True)
    ax[0].legend(prop={'size': 14, 'weight': 'bold'})
    ax[0].tick_params(axis='both', which='major', labelsize=15)

    plt.subplots_adjust(hspace=0.3)

    ax[1].set_title("Accuracy", fontsize=14, fontweight="bold")
    ax[1].set_xlabel("Epoch #", fontsize=14, fontweight="bold")
    ax[1].set_ylabel("Accuracy", fontsize=14, fontweight="bold")
    ax[1].plot(np.arange(1, epochs+1), result.history['bin_accuracy'], label='Accuracy', linewidth=2.5, linestyle='-',
               marker='o', markersize='10', color='red')
    ax[1].plot(np.arange(1, epochs+1), result.history['val_bin_accuracy'], label='Validation_accuracy', linewidth=2.5,
               linestyle='--', marker='x', markersize='10', color='blue')
    ax[1].grid(True)
    ax[1].legend(prop={'size': 14, 'weight': 'bold'}, loc='best')
    ax[1].tick_params(axis='both', which='major', labelsize=15)
    file_figobj = 'Output/FigureObject/%s_%d_EPOCH_%d_layers_opt.fig.pickle' % (type_model, epochs, layers_len)
    file_pdf = 'Output/Figures/%s_%d_EPOCH_%d_layers_opt.pdf' % (type_model, epochs, layers_len)

    pickle.dump(fig, open(file_figobj, 'wb'))
    fig.savefig(file_pdf, bbox_inches='tight')


def plot_metrics(history):
    metrics = ['loss', 'auc', 'precision', 'recall', 'bin_accuracy']
    epochs = len(history.history['accuracy'])
    (fig, ax) = plt.subplots(1, 5, figsize=(20, 5))
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        ax[n].plot(history.epoch, history.history[metric], linewidth=2.5, linestyle='-', marker='o', markersize='10',
                   color='blue', label='Train')
        ax[n].plot(history.epoch, history.history['val_'+metric], linewidth=2.5, linestyle='--', marker='x',
                   markersize='10', color='blue', label='Val')
        ax[n].grid(True)
        # plt.xlabel('Epoch')
        # plt.ylabel(name)
        ax[n].set_xlabel("Epoch", fontsize=14, fontweight="bold")
        ax[n].set_ylabel(name, fontsize=14, fontweight="bold")
        ax[n].legend(prop={'size': 14, 'weight': 'bold'}, loc='best')
        ax[n].tick_params(axis='both', which='major', labelsize=15)

    file_figobj = 'Output/FigureObject/Metric_%d_EPOCH.fig.pickle' % epochs
    file_pdf = 'Output/Figures/Metric_%d_EPOCH.pdf' % epochs

    pickle.dump(fig, open(file_figobj, 'wb'))
    fig.savefig(file_pdf, bbox_inches='tight')


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
    """
    fig_conf = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', size=12, fontweight='bold')
    plt.xlabel('Predicted label', size=12, fontweight='bold')
    # file_pdf = 'Output/Figures/confusion_matrix.pdf'
    file_figobj = 'Output/FigureObject/confusion_matrix.fig.pickle'
    pickle.dump(fig_conf, open(file_figobj, 'wb'))


def plot_roc(name, fpr, tpr, **kwargs):
    (fig, ax) = plt.subplots(1, 1, figsize=(13, 13))
    ax.plot(100*fpr, 100*tpr, label=name, linewidth=2, **kwargs)
    ax.xlabel("False Positive [%]")
    ax.ylabel("True Positive [%]")
    ax.grid(True)
    ax.set_aspect('equal')


def plot_scheduling():
    obs_int_flight_40 = [5.3589, 5.3589, 5.2759, 4.851, 5.33, 5.29, 2.74, 2.74, 4.25, 2.69, 4.235, 3.292, 3.13, 2.668,
                         1.806, 0.987, 0.987, 0.987, 0.987, 0.987, 0.987]
    obs_int_flight_50 = [5.26, 5.26, 5.23, 4.431, 5.223, 5.104, 3.542, 4.785, 4.785, 2.617, 2.617, 2.617, 2.617, 2.617,
                         2.617, 0.991, 0.991, 0.991, 0.991, 0.991, 0.991]
    obs_int_flight_60 = [5.187, 5.187, 5.437, 5.395, 4.466, 4.466, 5.133, 3.327, 4.212, 1.813, 2.516, 2.516, 2.516,
                         2.397, 2.055, 0.992, 0.992, 0.992, 0.992, 0.992, 0.992]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.set_xlabel("Observation Interval (min)", size=12, fontweight="bold")
    ax.set_ylabel("Number of required UAVs", size=12, fontweight="bold")
    ax.plot(np.arange(5, 26), obs_int_flight_40, color="blue", linestyle='-', linewidth=2, label="Flight time: 40min",
            marker='o', markersize=8)

    ax.plot(np.arange(5, 26), obs_int_flight_50, color="red", linestyle='--', linewidth=2, label="Flight time: 50min",
            marker='+', markersize='8')

    ax.plot(np.arange(5, 26), obs_int_flight_60, color="black", linestyle='-', linewidth=2,
            label="Flight time: 60min", marker='+', markersize='8')

    ax.legend(loc='best')
    fig.canvas.draw()

    file_figobj = 'Output/FigureObject/required_UAV.fig.pickle' % ()
    file_pdf = 'Output/Figures/required_UAV.pdf' % ()
    pickle.dump(fig, open(file_figobj, 'wb'))
    fig.savefig(file_pdf, bbox_inches='tight')

    plt.show()


def plot_interval(pile_times):
    number_piles = len(pile_times)
    interval_output = [[] for _ in range(0, number_piles)]
    for i in range(0, number_piles):
        interval_output[i] = [t - s for s, t in zip(pile_times[i], pile_times[i][1:])]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.set_xlabel("Number of observations", size=12, fontweight="bold")
    ax.set_ylabel("Consecutive interval for observation (min)", size=12, fontweight="bold")

    ax.plot(np.arange(1, len(interval_output[0])+1), interval_output[0], color="blue", linestyle='--', linewidth=2,
            label="First Pile", marker='o', markersize=8)

    ax.plot(np.arange(1, len(interval_output[1])+1), interval_output[1], color="red", linestyle='--', linewidth=2,
            label="Second Pile", marker='X', markersize=8)

    ax.plot(np.arange(1, len(interval_output[2])+1), interval_output[2], color="black", linestyle='--', linewidth=2,
            label="Third Pile", marker='P', markersize=8)

    ax.plot(np.arange(1, len(interval_output[3])+1), interval_output[3], color="green", linestyle='--', linewidth=2,
            label="Fourth Pile", marker='*', markersize=8)

    ax.plot(np.arange(1, len(interval_output[4])+1), interval_output[4], color="magenta", linestyle='--', linewidth=2,
            label="Fifth Pile", marker='+', markersize=8)

    ax.plot(np.arange(1, len(interval_output[4])+1), interval_output[4], color="brown", linestyle='--', linewidth=2,
            label="Sixth Pile", marker='s', markersize=8)

    ax.legend(loc='best')
    fig.canvas.draw()

    file_figobj = 'Output/FigureObject/Consecutive_interval.fig.pickle' % ()
    file_pdf = 'Output/Figures/Consecutive_interval.pdf' % ()
    pickle.dump(fig, open(file_figobj, 'wb'))
    fig.savefig(file_pdf, bbox_inches='tight')

    plt.show()


def plot_segmentation_test(xval, yval, ypred, num_samples):
    fig = plt.figure(figsize=(16, 13))
    for i in range(0, num_samples):
        plt.subplot(3, num_samples, (0 * num_samples) + i + 1)
        ix_val = random.randint(0, len(ypred) - 1)
        title = str(i+1)
        plt.title(title)
        imshow(xval[ix_val])
        plt.axis('off')

        plt.subplot(3, num_samples, (1 * num_samples) + i + 1)
        plt.imshow(np.squeeze(yval[ix_val]))
        plt.title('gTruth')
        plt.axis('off')

        plt.subplot(3, num_samples, (2 * num_samples) + i + 1)
        plt.imshow(np.squeeze(ypred[ix_val]))
        plt.title('Mask')
        plt.axis('off')
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    file_figobj = 'Output/FigureObject/segmentation_test.fig.pickle' % ()
    file_pdf = 'Output/Figures/segmentation_test.pdf' % ()
    pickle.dump(fig, open(file_figobj, 'wb'))
    fig.savefig(file_pdf, bbox_inches='tight')
