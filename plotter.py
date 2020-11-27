import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_alignment(alignment, path, title, max_len=None):
    if max_len is not None:
        alignmnet = alignment[:,:max_len]
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)

    im = ax.imshow(
            alignment,
            aspect='auto',
            origin='lower',
            interpolation='none')
    fig.colorbar(im,ax=ax)
    xlabel = 'Result'
    ylabel = 'Text'
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path,format='png')
    plt.close()

