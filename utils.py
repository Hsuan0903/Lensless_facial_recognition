import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import os
from torchvision.datasets import ImageFolder
import csv
from tqdm import tqdm
import random
from typing import Optional, Callable, Any, Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class MultiEpochsDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

def dataset(PATH, BATCH_SIZE,num_workers):
    # path 
    train_path = os.path.join(PATH,'train')
    test_path = os.path.join(PATH,'test')
    # transform of loader
    transform=transforms.Compose([ 
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        ])
    # data set
    train_set = ImageFolder(train_path, transform = transform)
    test_set = ImageFolder(test_path, transform = transform)
    train_loader = MultiEpochsDataLoader(dataset=train_set,batch_size=BATCH_SIZE,shuffle=True,pin_memory=True,num_workers=num_workers,drop_last=True)
    test_loader = MultiEpochsDataLoader(dataset=test_set,batch_size=BATCH_SIZE,shuffle=True,pin_memory=True,num_workers=num_workers,drop_last=True)
    return train_loader, test_loader

    
def train(loader, model, metric_fc, criterion, optimizer, device, scheduler):
    model.train()
    metric_fc.train()
    
    for batch_idx,(input, label) in enumerate(tqdm(loader,leave=False)):

        input, label = input.to(device), label.to(device)
        
        features = model(input)
        
        output = metric_fc(features,label)
        
        loss = criterion(output,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
# testing step
def test(loader, model, metric_fc, criterion, device):

    total_loss, total_ac, count = 0., 0., 0.
    features_list, labels_list = [], []
    model.eval()
    metric_fc.eval()
    with torch.no_grad():
        for batch_idx,(input,label) in enumerate(tqdm(loader,leave=False)):
            count += input.size()[0]

            input, label = input.to(device), label.to(device)

            features = model(input)
            output = metric_fc(features,label)
            loss = criterion(output,label)
            
            total_loss += loss.cpu().data.numpy()
            
            _, prediction = torch.max(output,1)
            pred_y = prediction.cpu().data.numpy().squeeze()
            target_y = label.cpu().data.numpy()
            ac = sum(pred_y == target_y)
            total_ac += ac
            features_list.append(features)
            labels_list.append(label)
    total_loss = total_loss/(batch_idx+1)
    total_ac = total_ac/count

    return total_ac, total_loss, features_list,labels_list

# plot learning curve
def plot_learning_curve(train_lc,test_lc, display_lc, path, epoch):
    plt.figure(facecolor='white')
    plt.plot(train_lc,label='training data')
    plt.plot(test_lc,label='testing data')

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc = 'best')
    plt.grid()
    plt.title('Learning curve')
    plt.xticks(list(range(0,epoch+1,10)))
    plt.savefig(os.path.join(path,'learning curve.png'))
    if display_lc == True:
        plt.show()

# plot accuracy curve
def plot_accuracy_curve(train_ac,test_ac, display_lc, path, epoch):
    plt.figure(facecolor='white')
    plt.plot(train_ac,label='training data')
    plt.plot(test_ac,label='testing data')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc = 'best')
    plt.grid()
    plt.xticks(list(range(0,epoch+1,10)))
    plt.title('Accuracy curve')
    plt.savefig(os.path.join(path,'accuracy curve.png'))
    if display_lc == True:
        plt.show()

def print_opt(opt):
    print('MODEL: {}'.format(opt.model))
    print('NMUBER OF CLASS: {}'.format(opt.num_class))
    print('EMBEDDING SIZE: {}'.format(opt.embedding_size))
    print('METRIC FUNCTION: {}'.format(opt.metric))
    if opt.metric != 'softmax' or 'adacos':
        print('MARGIN_S: {}'.format(opt.margin_s))
        print('MARGIN_M: {}'.format(opt.margin_m))
    print('EPOCH: {}'.format(opt.epochs))
    print('BATCH SIZE: {}'.format(opt.batch_size))
    print('LEARNING RATE: {}'.format(opt.learning_rate))
    print('DEVICE: {}'.format(opt.device))
    print('DATA PATH: {}'.format(opt.blur_path))
    

def save_opt(output_path,opt):
    file = os.path.join(output_path,'training parameter.csv')
    w = csv.writer(open(file,'w',newline=''))
    for key, val in vars(opt).items():
        w.writerow([key,val])

def create_folder(path,folder):
    output = os.path.join(path,folder)
    try:
        os.makedirs(output)
    except OSError:
        print('Rename the folder')
        i=1
        while True:
            new = output+'_'+str(i)
            if not os.path.exists(new):
                os.makedirs(new)
                output = new
                break
            i +=1
    return output

def visualize(features, labels, epoch, writer,title,method):
    features = features.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    if method == 'PCA':
        embedded = PCA(n_components=2).fit_transform(features)
    elif method == 'TSNE':
        embedded = TSNE(n_components=2).fit_transform(features)
        
    colors = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    
    fig = plt.figure(figsize=(5, 4), dpi=100)
    fig.clf()
    for i in range(10):
        plt.plot(embedded[labels == i, 0], embedded[labels == i, 1], ".", c=colors[i], label=i)
    plt.legend(loc="upper right") 
    plt.title('{} visualization'.format(method))
    plt.grid()
    
    writer.add_figure('{}/visualization'.format(title),fig,epoch)