#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 16:27:22 2018

@author: lsk
"""
'''
import matplotlib.pyplot as plt
from logger import Logger as log

import torch as t
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 20               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001              # learning rate
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,
)

# plot one example
print(train_data.train_data.size())                 # (60000, 28, 28)
print(train_data.train_labels.size())               # (60000)
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()

train_loader=Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)

test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x=t.unsqueeze(test_data.test_data,dim=1).type(t.FloatTensor)[:2000].cuda()/255
test_y=test_data.test_labels[:2000].cuda()

class CNN(nn.Module):
    def __init__(self,in_chn,out_chn):
        super(CNN,self).__init__()
        self.cnn1=nn.Sequential(
                nn.Conv2d(in_chn,16,3,1,1),
                nn.ReLU6(),
                nn.MaxPool2d(2)
                )
        self.cnn2=nn.Sequential(
                nn.Conv2d(16,32,3,1,1),
                nn.ReLU6(),
                nn.MaxPool2d(2)
                )
        self.out=nn.Linear(32 * 7 * 7, out_chn)
        
    def forward(self,x):
        x=self.cnn1(x)
        x=self.cnn2(x)
        x=x.view(x.size(0),-1)
        o=self.out(x)
        return o,x

class Net(nn.Module):
    def __init__(self,in_num,out_num):
        super(Net,self).__init__()
        self.h1=nn.Linear(in_num,10)
        self.h2=nn.Linear(10,20)
        self.h3=nn.Linear(20,10)
        self.out1=nn.Linear(10,out_num)
        
    def forward(self,x):
        x=F.relu6(self.h1(x))
        x=F.relu6(self.h2(x))
        x=F.relu6(self.h3(x))
        x=self.out1(x)
        return x



net=CNN(1,10)
net.cuda()
log.info(net)

optimizer=t.optim.Adam(net.parameters(), lr=0.02)
loss_func=nn.CrossEntropyLoss()

# following function (plot_with_labels) is for visualization, can be ignored if not interested
from matplotlib import cm
try: from sklearn.manifold import TSNE; HAS_SK = False
except: HAS_SK = False; print('Please install sklearn for layer visualization')
def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

plt.ion()

for epoch in range(EPOCH):
    for step,(b_x,b_y) in enumerate(train_loader):
        b_x=b_x.cuda()
        b_y=b_y.cuda()
        predict=net(b_x)[0]
        loss=loss_func(predict,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            test_output, last_layer = net(test_x)
            pred_y = t.max(test_output, 1)[1].cuda().data.squeeze()
            accuracy = float(sum(pred_y == test_y)) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)
            if HAS_SK:
                # Visualization of trained flatten layer (T-SNE)
                tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                plot_only = 500
                low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                labels = test_y.numpy()[:plot_only]
                plot_with_labels(low_dim_embs, labels)

plt.ioff()

# print 10 predictions from test data
test_output, _ = net(test_x[:10])
pred_y = t.max(test_output, 1)[1].cpu().numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].cpu().numpy(), 'real number')
'''
'''
for itr in range(10):
    predict=net(x)
    loss=loss_func(predict,y)
    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients
    
    if itr % 2 == 0:
        # plot and show learning process
        #plt.cla()
        prediction = t.max(predict, 1)[1]
        pred_y = prediction.numpy()#.squeeze()
        target_y = y.data.numpy()
        #plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/200.
        #plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        #plt.pause(0.1)
        log.info(prediction)
        print(prediction)
        log.info(accuracy)

#plt.ioff()
#plt.show()
'''