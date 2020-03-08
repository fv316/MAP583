
'''
Plotter class for plotting various things
'''
from sklearn.metrics import roc_curve, auc
import io
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import sys
import copy
import itertools
import matplotlib as mpl
mpl.use('Agg')


'''
             .                 .    o8o
           .o8               .o8    `"'
 .oooo.o .o888oo  .oooo.   .o888oo oooo   .ooooo.
d88(  "8   888   `P  )88b    888   `888  d88' `"Y8
`"Y88b.    888    .oP"888    888    888  888
o.  )88b   888 . d8(  888    888 .  888  888   .o8
8""888P'   "888" `Y888""8o   "888" o888o `Y8bod8P'
'''


# get plot data from logger and plot to image file
def save_plot(args, logger, tags=['train', 'val'], name='loss', title='loss curves', labels=None):
    var_dict = copy.copy(logger.logged)
    labels = tags if labels is None else labels

    epochs = None
    for tag in tags:
        if epochs is None:
            epochs = np.array([x for x in var_dict[tag][name].keys()])

        curr_line = np.array([x for x in var_dict[tag][name].values()])
        plt.plot(epochs, curr_line)

    plt.grid(True)
    plt.xlabel('epochs')
    plt.title('{} - {}'.format(title, args.name))
    plt.legend(labels=labels)

    out_fn = os.path.join(args.log_dir, 'pics',
                          '{}_{}.png'.format(args.name, name))
    plt.savefig(out_fn, bbox_inches='tight', dpi=150)
    plt.gcf().clear()
    plt.close()


def save_as_best(is_best, out_fn, extension='png'):
    if is_best:
        shutil.copyfile(out_fn, out_fn.replace(
            '.' + 'extension', '_best.' + 'extension'))


def plot_confusion_matrix(cm, out_fn, classnames=None, normalize=False, cmap=plt.cm.Blues, tb_writer=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if classnames == None:
        classnames = [str(i) for i in range(len(cm))]

    # classnames =  np.array(classnames)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = "Normalized confusion matrix"
        print(title)
    else:
        title = 'Confusion matrix, without normalization'
        print(title)

    figure = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    plt.ylabel('True label', fontsize=10)
    plt.xlabel('Predicted label', fontsize=10)
    plt.colorbar()
    tick_marks = np.arange(len(classnames))

    plt.xticks(tick_marks, classnames, rotation=0, fontsize=8)
    plt.yticks(tick_marks, classnames, rotation=0, fontsize=8)

    formating = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], formating),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #if tb_writer:
    #    tb_writer.add_image(title, plot_to_image(figure))

    print('saving plot to {out_fn}')
    plt.savefig(out_fn, bbox_inches='tight', dpi=300)
    plt.gcf().clear()
    plt.close()


def plot_roc_curve(ground, scores, out_fn, tb_writer=None):
    """
    This function prints and plots the roc curve.
    """
    fpr, tpr, _ = roc_curve(ground, scores, pos_label=0)
    roc_auc = auc(fpr, tpr)

    figure = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw,
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=10)
    plt.ylabel('True Positive Rate', fontsize=10)
    plt.title('Receiver operating characteristic curve', fontsize=20)
    plt.legend(loc="lower right")

    #if tb_writer:
    #    tb_writer.add_image("ROC curve", plot_to_image(figure))

    print('saving plot to {out_fn}')
    plt.savefig(out_fn, bbox_inches='tight', dpi=300)
    plt.gcf().clear()
    plt.close()


def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    canvas = mpl.backends.backend_agg.FigureCanvas(figure)
    canvas.draw()       # draw the canvas, cache the renderer
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    return image
