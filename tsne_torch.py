#
#  tsne_torch.py
#
# Implementation of t-SNE in pytorch. The implementation was tested on pytorch
# > 1.0, and it requires Numpy to read files. In order to plot the results,
# a working installation of matplotlib is required.
#
#
# The example can be run by executing: `python tsne_torch.py`
#
#
#  Created by Xiao Li on 23-03-2020.
#  Copyright (c) 2020. All rights reserved.
import numpy as np
import matplotlib.pyplot as pyplot
import argparse
import torch
from sklearn import manifold
import pickle
import numpy
import sys
parser = argparse.ArgumentParser()
parser.add_argument("--xfile", type=str, default="mnist2500_X.txt", help="file name of feature stored")
parser.add_argument("--yfile", type=str, default="mnist2500_labels.txt", help="file name of label stored")
parser.add_argument("--cuda", type=int, default=1, help="if use cuda accelarate")
parser.add_argument('--data',type=str,default='data_pretrained')
parser.add_argument('--init_dim',type=int,default=256)
parser.add_argument('--perplex',type=float,default=20.0)
parser.add_argument('--init',type=str,default='random')
parser.add_argument('--lr',type=float,default=200)
parser.add_argument('--niter',type=int,default=1000)
opt = parser.parse_args()
print("get choice from args", opt)
xfile = opt.xfile
yfile = opt.yfile

if opt.cuda:
    print("set use cuda")
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    torch.set_default_tensor_type(torch.DoubleTensor)


def Hbeta_torch(D, beta=1.0):
    P = torch.exp(-D.clone() * beta)

    sumP = torch.sum(P)

    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    P = P / sumP

    return H, P


def x2p_torch(X, tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape

    sum_X = torch.sum(X*X, 1)
    D = torch.add(torch.add(-2 * torch.mm(X, X.t()), sum_X).t(), sum_X)

    P = torch.zeros(n, n)
    beta = torch.ones(n, 1)
    logU = torch.log(torch.tensor([perplexity]))
    n_list = [i for i in range(n)]

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        # there may be something wrong with this setting None
        betamin = None
        betamax = None
        Di = D[i, n_list[0:i]+n_list[i+1:n]]

        (H, thisP) = Hbeta_torch(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while torch.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].clone()
                if betamax is None:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].clone()
                if betamin is None:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta_torch(Di, beta[i])

            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, n_list[0:i]+n_list[i+1:n]] = thisP

    # Return final P-matrix
    return P


def pca_torch(X, no_dims=50):
    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - torch.mean(X, 0)

    (l, M) = torch.eig(torch.mm(X.t(), X), True)
    # split M real
    for i in range(d):
        if l[i, 1] != 0:
            M[:, i+1] = M[:, i]
            i += 1

    Y = torch.mm(X, M[:, 0:no_dims])
    return Y


def tsne(X, no_dims=2, initial_dims=50, perplexity=30.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should not have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca_torch(X, initial_dims)
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = torch.randn(n, no_dims)
    dY = torch.zeros(n, no_dims)
    iY = torch.zeros(n, no_dims)
    gains = torch.ones(n, no_dims)

    # Compute P-values
    P = x2p_torch(X, 1e-5, perplexity)

    print('1',P)
    P = P + P.t()
    print('2',P)
    #print(P[P==numpy.nan])
    #exit()
    P[P==numpy.nan]=0
    # for i,t in enumerate(P):
    #     print(t)
    #     print(torch.sum(t))
    #     if torch.isnan(torch.sum(t)):
    #         print(i)
    # print(torch.sum(P))
    P = P / torch.sum(P)

    #print(torch.sum(P))
    print('3',P)
    P = P * 4.    # early exaggeration
    print("get P shape", P.shape)
    print('P',P)
    P = torch.max(P, torch.tensor([1e-21]))
    print('P',P)
    exit()
    #print('Y',Y)
    # Run iterations
    for iter in range(max_iter):
        print('Y',Y)
        # Compute pairwise affinities
        sum_Y = torch.sum(Y*Y, 1)
        num = -2. * torch.mm(Y, Y.t())
        num = 1. / (1. + torch.add(torch.add(num, sum_Y).t(), sum_Y))
        num[range(n), range(n)] = 0.
        print('num',num)
        Q = num / torch.sum(num)
        Q = torch.max(Q, torch.tensor([1e-12]))
        print('Q',Q)
        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = torch.sum((PQ[:, i] * num[:, i]).repeat(no_dims, 1).t() * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)).double() + (gains * 0.8) * ((dY > 0.) == (iY > 0.)).double()
        print('gain',gains,'min_gain',min_gain)
        print('momentum',momentum)
        print('iY',iY,'dY',dY)
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - torch.mean(Y, 0)
        print('iY',iY)
        print('Y',Y)
        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = torch.sum(P * torch.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y

def draw_scat():
    with open('{}.pkl'.format(opt.data), 'rb') as f:
        X, labels = pickle.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = labels.cpu().numpy().tolist()
    # X = np.loadtxt(xfile)
    # X = torch.Tensor(X)
    # labels = np.loadtxt(yfile).tolist()
    print(X.shape, len(labels))
    print(X[:10], labels[:10])
    # confirm that x file get same number point than label file
    # otherwise may cause error in scatter
    assert (len(X[:, 0]) == len(X[:, 1]))
    assert (len(X) == len(labels))
    # X= X[:100]
    # labels = labels[:100]
    X = X.cpu().numpy()
    print(np.isnan(X).sum())
    print(np.isinf(X).sum())

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=opt.perplex)
    Y = tsne.fit_transform(X)
    # with torch.no_grad():
    #     Y = tsne(X, 2, opt.init_dim, opt.perplex)
    # Y = Y.tolist()
    # print(Y,type(Y))
    Y0, Y1, new_labels = [], [], []
    for i, y in enumerate(Y):
        if y[0] > 50 or y[1] > 50:
            continue
        Y0.append(y[0])
        Y1.append(y[1])
        new_labels.append(labels[i])

    # print(Y)
    labels = new_labels
    # if opt.cuda:
    #     Y = Y.cpu().numpy()

    # You may write result in two files
    # print("Save Y values in file")
    # Y1 = open("y1.txt", 'w')
    # Y2 = open('y2.txt', 'w')
    # for i in range(Y.shape[0]):
    #     Y1.write(str(Y[i,0])+"\n")
    #     Y2.write(str(Y[i,1])+"\n")
    # dir = sys.argv[1]

    pyplot.scatter(Y0, Y1, 15, labels)
    pyplot.savefig('./{}_init{}_pp{}.png'.format(opt.data, opt.perplex))


if __name__ == "__main__":
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    with open('{}.pkl'.format(opt.data),'rb') as f:
        X,labels = pickle.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = labels.cpu().numpy().tolist()
    # X = np.loadtxt(xfile)
    # X = torch.Tensor(X)
    # labels = np.loadtxt(yfile).tolist()
    print(X.shape,len(labels))
    print(X[:10],labels[:10])
    # confirm that x file get same number point than label file
    # otherwise may cause error in scatter
    assert(len(X[:, 0])==len(X[:,1]))
    assert(len(X)==len(labels))
    # X= X[:100]
    # labels = labels[:100]
    X = X.cpu().numpy()
    print(np.isnan(X).sum())
    print(np.isinf(X).sum())

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,perplexity=opt.perplex,learning_rate=opt.lr)
    Y = tsne.fit_transform(X)
    # with torch.no_grad():
    #     Y = tsne(X, 2, opt.init_dim, opt.perplex)
    #Y = Y.tolist()
    #print(Y,type(Y))
    Y0,Y1,new_labels = [],[],[]
    for i,y in enumerate(Y):
        # if y[0]>-250 or y[1]>500:
        #     continue
        Y0.append(y[0])
        Y1.append(y[1])
        new_labels.append(labels[i])

    #print(Y)
    labels=new_labels
    # if opt.cuda:
    #     Y = Y.cpu().numpy()

    # You may write result in two files
    # print("Save Y values in file")
    # Y1 = open("y1.txt", 'w')
    # Y2 = open('y2.txt', 'w')
    # for i in range(Y.shape[0]):
    #     Y1.write(str(Y[i,0])+"\n")
    #     Y2.write(str(Y[i,1])+"\n")
    #dir = sys.argv[1]

    pyplot.scatter(Y0, Y1, 20, labels)
    print('./{}_pp{}_2.png')
    pyplot.savefig('./{}_pp{}_lr{}.png'.format(opt.data,opt.perplex,opt.lr))
    pyplot.show()
