import matplotlib.pyplot as plt
import numpy as np
# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
	# TODO: Make plots for loss curves and accuracy curves.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	plt.figure()
	plt.plot(np.arange(len(train_losses)), train_losses, label='Train')
	plt.plot(np.arange(len(valid_losses)), valid_losses, label='Validation')
	plt.ylabel('Loss')
	plt.xlabel('epoch')
	plt.legend(loc="best")
	plt.savefig("losses_curves.png")
    
	plt.figure()
	plt.plot(np.arange(len(train_accuracies)), train_accuracies, label='Train')
	plt.plot(np.arange(len(valid_accuracies)), valid_accuracies, label='Validation')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(loc="best")
	plt.savefig("accuracies_curves.png")

	pass

# reference : https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(results, class_names):
	# TODO: Make a confusion matrix plot.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
    #print(results)
    #print(class_names)
    # 			results.extend(list(zip(y_true, y_pred)))
    transpose = np.array(results).transpose()
    y_true = list(transpose[0])
    y_pred = list(transpose[1])

    count = 0
    for i in y_true: 
        if i == 0: 
            count = count + 1
    count = 0
    for i in y_pred: 
        if i == 0: 
            count = count + 1
    
    title = 'Normalized confusion matrix'
    cm = confusion_matrix(y_true, y_pred)
    classes = class_names
    cmap=plt.cm.Blues
    cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,title=title,
           ylabel='True',
           xlabel='Predicted')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    
    plt.savefig("confusion_matrix.png")    
    pass
