import matplotlib.pyplot as plt
import numpy as np
import itertools

def scatter_color_markers(X, groups, cmap=plt.cm.Set1, size=10, alpha=0.5):
    """
    encodes groups as a combo of color and marker
    :param X:
    :param groups:
    :return:
    """
    # plt.figure()
    assert isinstance(X, np.ndarray)
    assert isinstance(groups, np.ndarray)
    # make sure that we get the same coloring all the time, no permutations
    ix_ = np.argsort(groups)
    X = X[ix_]
    groups = groups[ix_]


    n_groups = len(set(groups))
    markers = ['o','x','d','*','+', 'v','>','1','2','3','4','s','p','|','_']
    colors = cmap(np.linspace(0,1,9))
    markers_colors = itertools.product(markers, colors )

    leg=[]
    for g, m_c in zip(np.unique(groups), markers_colors):
        ix = groups == g
        plt.scatter(X[ix,0], X[ix,1], c=m_c[1], marker=m_c[0], alpha=alpha, s=size)
        leg.append(g)
    if len(np.unique(groups)) > 15:
        plt.legend(leg, prop={'size': 7})
    else:
        plt.legend(leg)



def scattermatrix(*X, **kwargs):
    """
    a series of @D scatter plots of a single or multiple datasets *X
    :param X:
    :param kwargs:
        - node_alpha
        - highlight
    :return:
    """
    fh = plt.figure(figsize=(20, 20))
    n_row_col = np.minimum(5, X[0].shape[1])

    node_alpha = 0.1 if 'node_alpha' not in kwargs else kwargs['node_alpha']
    node_size = 2 if 'node_size' not in kwargs else kwargs['node_size']
    for i in range(n_row_col):
        for j in range(i,n_row_col): #ZZ1.shape[1]
            plt.subplot(n_row_col,n_row_col,1+j*n_row_col+i)
            for xname, y in enumerate(X):

                if 'color' in kwargs:
                    plt.scatter(y[:, i], y[:, j], s=node_size, c=kwargs['color'], alpha=node_alpha, label=xname)
                else:
                    plt.scatter(y[:,i], y[:,j], s=node_size, alpha=node_alpha, label=xname)
                if 'highlight' in kwargs:
                    plt.scatter(y[kwargs['highlight'],i], y[kwargs['highlight'],j], s=1, label='%d_high'%xname)

            if 'pairs' in kwargs:
                for jj in range(len(X[0])):
                        plt.arrow(X[0][jj, i],
                                  X[0][jj, j],
                                  X[1][jj, i] - X[0][jj, i],
                                  X[1][jj, j] - X[0][jj, j], alpha=0.01, edgecolor=None, width=0.00001)
            # plt.xlabel(f'latent space {i}')
            # plt.ylabel(f'latent space {j}')
    plt.legend()

    return fh


def scatter_pairs(x1,x2, node_color=None, edge_color=None, edge_alpha=0.4, node_alpha=0.4, node_size=3 ):
    """
    displaying pairwise relation between two datasets in a scatterplot ()
    :param x1:
    :param x2:
    :param node_color:
    :param edge_color:
    :param edge_alpha:
    :param node_alpha:
    :return:
    """
    assert len(x1) == len(x2)

    plt.scatter(x1[:, 0], x1[:, 1], alpha=node_alpha, s=node_size)
    plt.scatter(x2[:, 0], x2[:, 1], alpha=node_alpha, s=node_size)

    for jj in range(len(x1)):
        if edge_color:
            plt.arrow(x1[jj, 0],
                      x1[jj, 1],
                      x2[jj, 0] - x1[jj, 0],
                      x2[jj, 1] - x1[jj, 1], alpha=edge_alpha, color=edge_color[jj], edgecolor=None, width=0.00001)
            # plt.plot(x1[jj, 0],
            #           x1[jj, 1],
            #           x2[jj, 0],
            #           x2[jj, 1], alpha=edge_alpha, color=edge_color[jj])
        else:
            plt.arrow(x1[jj, 0],
                      x1[jj, 1],
                      x2[jj, 0] - x1[jj, 0],
                      x2[jj, 1] - x1[jj, 1], alpha=edge_alpha, edgecolor=None, width=0.00001)
            # plt.plot(x1[jj, 0],
            #           x1[jj, 1],
            #           x2[jj, 0],
            #           x2[jj, 1], alpha=edge_alpha)



def scatter_triplets(x1,x2,x3, node_color=None, edge_color=None, edge_alpha=0.4, node_alpha=0.4 ):

    assert len(x1) == len(x2)

    plt.scatter(x1[:, 0], x1[:, 1], alpha=node_alpha, s=3, label='x1')
    plt.scatter(x2[:, 0], x2[:, 1], alpha=node_alpha, s=3, label='x2')
    plt.scatter(x3[:, 0], x3[:, 1], alpha=node_alpha, s=3, label='x3')

    for jj in range(len(x1)):
        tmp_edge_color = 'black' if not edge_color else edge_color[jj]
        plt.arrow(x1[jj, 0],
                  x1[jj, 1],
                  x2[jj, 0] - x1[jj, 0],
                  x2[jj, 1] - x1[jj, 1], alpha=edge_alpha, color=tmp_edge_color, edgecolor=None)
        plt.arrow(x2[jj, 0],
                  x2[jj, 1],
                  x3[jj, 0] - x2[jj, 0],
                  x3[jj, 1] - x2[jj, 1], alpha=edge_alpha, color=tmp_edge_color, edgecolor=None)

    plt.legend()


def scatter_highlight_lineages(z, lineages, highlight, marker='o'):
    """
    scatter plot, highlighting some of the lineages
    :param z:
    :param lineages: same length as z
    :param highlight: subset of lineages, whihc to highlight
    :return:
    """
    import matplotlib.pyplot as plt
    assert len(z) == len(lineages)
    plt.scatter(z[:,0], z[:,1], c='grey', alpha=0.1, marker=marker)
    leg = ['all']
    for i in highlight:
        ix = lineages==i
        plt.scatter(z[ix, 0], z[ix, 1], alpha=0.5, marker=marker)
        leg.append(i)
    plt.legend(leg)