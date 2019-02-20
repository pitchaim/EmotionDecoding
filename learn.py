import os
import sys
import csv
import numpy as np
import numpy.random as npr
import string
import re
from scipy.stats import ttest_ind
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.tensor
import pickle as pkl

class EmoNet(nn.Module):

    def __init__(self):
        super(EmoNet, self).__init__()
        self.conv1 = nn.Conv1d(1,3,5)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv1d(3,6,5)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        #self.lin1 = nn.Linear(2560,100)
        self.lin1 = nn.Linear(15312,100)
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        #self.lin2 = nn.Linear(200,100)
        #self.lin3 = nn.Linear(100,50)
        self.lin2 = nn.Linear(100,3)
        torch.nn.init.xavier_uniform_(self.lin2.weight)

    def forward(self, x):
        #x = torch.transpose(x, 0, 1)
        #x = x.view(x.size(0), -1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        #x = F.relu(self.lin3(x))
        #x = F.relu(self.lin4(x))
        #x = F.softmax(x)
        #return torch.unsqueeze(torch.argmax(x),1)
        return x

    #taken from PyTorch documentation/tutorial
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class NetRunner:

    def __init__(self, l_r, epochs, cuda_avail):
        self.cuda_avail = cuda_avail
        self.net = EmoNet()
        if self.cuda_avail:
            self.net = self.net.cuda()
        self.l_r = l_r
        self.epochs = epochs

    #train until loss drops below 0.2 (chosen arbitrarily)
    def train(self, X, Y):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=self.l_r)
        lossval = 1.0
        delta = 0.0
        dd = 0.0
        for epoch in range(self.epochs):
            print('Epoch: %d' % epoch)
            for i in range(len(X)):
                if i % 100 == 0:
                    print('Training loss: %.4f' % lossval)
                ft = X[i]
                ft = [v/sum(X[i]) for v in ft]
                input = Variable(torch.unsqueeze(torch.unsqueeze(torch.Tensor(ft),0),0))
                if self.cuda_avail:
                    input = input.cuda()
                #input = Variable(torch.Tensor(X[i]))
                target = Variable(torch.unsqueeze(torch.Tensor(Y[i]),1))
                target = target.type(torch.LongTensor)
                if self.cuda_avail:
                    target = target.cuda()
                #target = Variable(torch.Tensor(Y[i]))
                #optimizer.zero_grad()
                out = self.net(input)
                loss = criterion(out, torch.max(target,0)[1])
                #dd = np.absolute(lossval - loss.data.item()) - np.absolute(delta)
                det_loss = torch.tensor(loss.data, requires_grad=True)
                delta = lossval - loss.data.item()
                lossval = loss.data.item()
                optimizer.zero_grad()
                det_loss.backward()
                optimizer.step()
        print('Training complete, loss = %.4f' % lossval)

    def test(self, X, Y):
        criterion = nn.CrossEntropyLoss()
        accs = []
        for i in range(len(X)):
            ft = X[i]
            ft = [v/sum(X[i]) for v in ft]
            input = Variable(torch.unsqueeze(torch.unsqueeze(torch.Tensor(ft),0),0))
            if self.cuda_avail:
                input = input.cuda()
            #input = Variable(torch.Tensor(X[i]))
            target = Variable(torch.unsqueeze(torch.Tensor(Y[i]),1))
            #target = Variable(torch.Tensor(Y[i]))
            target = target.type(torch.LongTensor)
            if self.cuda_avail:
                target = target.cuda()
            output = self.net(input)
            #print('TRYING TO GET VALUE OUT: %d' % output.item())
            loss = criterion(output, torch.max(target,0)[1])
            c = 0
            tt = torch.Tensor(Y[i]).requires_grad_(False)
            if torch.argmax(output).clone().detach().item() == torch.argmax(tt,0).item():
                c = 1
            accs.append(c)
        av = sum(accs)/len(accs)
        print('Mean accuracy: %.4f' % av)
        return av

class Util:

    def __init__(self):
        self.features = []
        self.classes = []

    #load in features and classes
    def load_data(self):
        self.features = pkl.load(open('features.pkl','rb'))
        self.classes = pkl.load(open('classes.pkl','rb'))

if __name__ == "__main__":
    #set everything up
    util = Util()
    util.load_data()
    l_r = 0.1
    epochs = 100
    folds = 1
    if len(sys.argv) > 1:
        l_r = float(sys.argv[1])
        epochs = int(sys.argv[2])

    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        print('CUDA IS AVAILABLE!!!')

    #run both over 10 folds
    test_mark = np.zeros((1,len(util.features)))
    nn_acc = []
    print('Running neural net')
    for k in range(folds):
        print('Creating splits for fold %d' % (k+1))
        test = npr.choice([i for i in range(len(util.features))],np.floor(len(util.features)/10).astype(int),False)
        for t in range(len(test)):
            if test_mark[0][test[t]] == 0:
                test_mark[0][test[t]] = 1
            else:
                while test_mark[0][test[t]] == 1 or test[t] in np.concatenate([test[:t], test[t+1:]]):
                    test[t] = npr.randint(0,len(util.features))
        train = [i for i in range(len(util.features)) if i not in test]
        train = npr.permutation(train) #shuffle order
        X = [util.features[i].tolist() for i in train]
        Y = [util.classes[i] for i in train]
        print('Sanity check: length of first feature:')
        print(np.shape(X[0]))
        print('Sanity check: length of first class:')
        print(np.shape(Y[0]))
        X_test = [util.features[i].tolist() for i in test]
        Y_test = [util.classes[i] for i in test]
        print('Train-test split created for fold %d' % (k+1))

        netrunner = NetRunner(l_r, epochs, cuda_avail)
        print('Training...')
        netrunner.train(X, Y)
        print('Testing...')
        nn_acc.append(netrunner.test(X_test, Y_test))
        del netrunner

    nn_av = sum(nn_acc)/len(nn_acc)
    nn_sd = np.std(nn_acc)
    print('Neural network complete. Average accuracy: %.4f SD: %.4f' % (nn_av, nn_sd))
    with open('nn-acc.txt', 'ab') as f:
        f.write('%.4f\n' % nn_av)
    f.close()
    print('\n\n')
