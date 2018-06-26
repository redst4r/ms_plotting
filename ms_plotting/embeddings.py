import matplotlib.pylab as plt
import numpy as np
from matplotlib import offsetbox
from matplotlib import cm
import progressbar
#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X_embedded, X_original=None, color=None, minDist=4e-3, adjMatrix=None,
                   title=None, cmap=cm.cubehelix, ax=None, markersize=20, noColorbar_FLAG=False,
                   zoom=0.5):
    """
    plots a 2D embeding of image patches, plotting the patches at their position in latent space
    only a fraction of patches is plotted to uniformly cover the whole plot
    :param X_embedded: the embedding in 2D
    :param X_original: the original image data, must be 3D: samples * X * Y
    :param color: color the datapoints accordinf to some label
    :param minDist:  controls the density of the patches: smaller-> more dense (default 4e-3)
    :param title:
    :param cmap: colormap for the scatter
    :param ax: optionally pass an axis handle. the scatter plot will be put into those axis (usful for subplots)
    :return:
    """

    x_min, x_max = np.min(X_embedded, 0), np.max(X_embedded, 0)
    X_embedded = (X_embedded - x_min) / (x_max - x_min)

    assert X_embedded.shape[1] == 2, "X_embed must only have two columns"

    if ax is None:
        plt.figure()
        ax = plt.subplot(111)

    plt.sca(ax)

    plt.scatter(X_embedded[:,0], X_embedded[:,1], c=color, cmap=cmap, s=markersize, linewidths=0)
    # for i in range(X_embedded.shape[0]):
    #     plt.text(X_embedded[i, 0], X_embedded[i, 1], str(digits.target[i]),
    #              color=plt.cm.Set1(y[i] / 10.),
    #              fontdict={'weight': 'bold', 'size': 9})


    if X_original is not None:
        assert X_embedded.shape[0] == X_original.shape[0], "X_embedded and X_original have different number of samples"
        assert len(X_original.shape) == 3 or (len(X_original.shape) ==4 and X_original.shape[-1] == 3), \
            "X_original must be 3D: samples * x * y, or 3D with RGB channel"

        if hasattr(offsetbox, 'AnnotationBbox'):
            # only print thumbnails with matplotlib > 1.0
            shown_images = np.array([[1., 1.]])  # just something big
            for i in range(X_original.shape[0]):
                dist = np.sum((X_embedded[i,:] - shown_images) ** 2, 1)
                if np.min(dist) < minDist:
                    # don't show points that are too close
                    continue
                shown_images = np.r_[shown_images, [X_embedded[i,:]]]
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(X_original[i,:,:], cmap=cm.gray, zoom=zoom),
                    X_embedded[i,:2],  pad=0) #bboxprops={'facecolor':color} bboxprops = dict(facecolor='wheat',boxstyle='round',color='black')
                ax.add_artist(imagebox)

    if adjMatrix is not None:
        assert adjMatrix.shape== (X_embedded.shape[0], X_embedded.shape[0]), 'dim of X_embedded and ajaceny matrix dont mathc'
        # for i in range(X_embedded.shape[0]):
        #     for j in range(X_embedded.shape[0]):
        #         if adjMatrix[i,j] != 0:
        #             plt.plot([X_embedded[i,0], X_embedded[j,0]], [X_embedded[i,1], X_embedded[j,1]], c='b' )

        # faster
        x,y = adjMatrix.nonzero()  # get all nonzero entries. convenient functino of the spasrse matrix
        for i in range(x.shape[0]):
            ix1, ix2 = x[i], y[i]
            plt.plot([X_embedded[ix1,0], X_embedded[ix2,0]], [X_embedded[ix1,1], X_embedded[ix2,1]], c='b' )


    #
    # percentage = 0.01
    # ix = np.random.random_integers(0,X_embedded.shape[0], int(X_embedded.shape[0] * percentage))
    # if hasattr(offsetbox, 'AnnotationBbox'):
    #     # only print thumbnails with matplotlib > 1.0
    #     shown_images = np.array([[1., 1.]])  # just something big
    #     for i in range(ix.shape[0]):
    #         imagebox = offsetbox.AnnotationBbox(
    #             offsetbox.OffsetImage(X_original[ix[i],:,:], cmap=cm.gray),
    #             X_embedded[i,:2], pad=0)
    #         ax.add_artist(imagebox)

    if color is not None and not noColorbar_FLAG:
        plt.colorbar()

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


def plot_embedding_multi(X_embedded, X_original=None, color=None, minDist=4e-3, adjMatrix=None,title=None, cmap=cm.cubehelix):
    """
    do a plotmatrix of the X_embedd
    :param X_embedded:
    :param X_original:
    :param color:
    :param minDist:
    :param adjMatrix:
    :param title:
    :param cmap:
    :return:
    """
    plt.figure()

    nDim = X_embedded.shape[1]

    bar = progressbar.ProgressBar(max_value=(nDim**2)/2 - nDim/2)
    counter=0
    for i in range(nDim):
        for j in range(i+1, nDim):
            # print(i,j)
            # print(i*(nDim-1)+j)
            ax = plt.subplot(nDim, nDim, i*(nDim-1)+j)
            plot_embedding(X_embedded[:, [i,j]], X_original=X_original, color=color, minDist=minDist, adjMatrix=adjMatrix,title=title, cmap=cmap, ax=ax, markersize=int(50/nDim),noColorbar_FLAG=True)
            counter+=1
            bar.update(counter)
    plt.colorbar()


import bokeh.plotting
from bokeh.plotting import ColumnDataSource, figure,gridplot,show
from bokeh.models import HoverTool
def plot_embedding_bokeh(X_embedded, X_original, color=None, minDist=4e-3, title=None):

    outfilename = '/tmp/plot_embedded_bokeh.html'
    bokeh.plotting.output_file(outfilename)

    source = ColumnDataSource(data=dict(x1=X_embedded[:,0], x2=X_embedded[:,1], x3=X_embedded[:,2]))

    TOOLS = "box_select,lasso_select,help"

    x1x2 = figure(tools=TOOLS, width=300, height=300, title=None,
                  x_axis_label='x1', y_axis_label='x2')
    x1x2.scatter('x1', 'x2', source=source, color=color)

    x1x3 = figure(tools=TOOLS, width=300, height=300, title=None,
                  x_axis_label='x1', y_axis_label='x3')
    x1x3.scatter('x1', 'x3', source=source, color=color)

    x2x3 = figure(tools=TOOLS, width=300, height=300, title=None,
                  x_axis_label='x2', y_axis_label='x3')
    x2x3.scatter('x2', 'x3', source=source, color=color)


    p = gridplot([[x1x2], [x1x3, x2x3]])

    show(p)