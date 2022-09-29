import os 
import numpy as np
import torch
import torch.nn as nn
from tkinter import Tk
from tkinter.filedialog import askdirectory,askopenfilename
import matplotlib.pyplot as plt
from model.iresnet import iresnet18
from model.resnet import resnet18
from metric import ArcFace, Softmax, AdaCos
from utils import *
import argparse


def eval(opt):
    
    # load data set
    train_loader, test_loader = dataset(opt.blur_path, opt.batch_size, 4)
    
    # Model dictionary
    model_dict = {'iresnet18':iresnet18,'resnet18': resnet18,}

    metric_dict = {'arcface':ArcFace, 'softmax':Softmax, 'adacos':AdaCos}
    # device(cuda or CPU
    device = opt.device
    
    # load model
    checkpoint = torch.load(opt.net_file)
    model = model_dict[checkpoint['model']](input_channel=checkpoint['input_channel'],
                                            embedding_size=checkpoint['embedding_size'],
                                            fc_scale=16).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if checkpoint['metric'] = 'arcface':
        metric = metric_dict[checkpoint['metric']](num_classes=opt.num_class, 
                                                embedding_size=checkpoint['embedding_size'],
                                                margin_m=checkpoint['margin_m'],
                                                margin_s=checkpoint['margin_s']
                                                ).to(device)
    elif:
        metric = metric_dict[checkpoint['metric']](num_classes=opt.num_class, 
                                                embedding_size=checkpoint['embedding_size']
                                                ).to(device)
    metric.load_state_dict(checkpoint['metric_state_dict'])
    model.eval(),metric.eval()
    train_total_ac = 0.0
    test_total_ac = 0.0
    train_count = 0
    test_count = 0
    with torch.no_grad():
        for _,(input,label) in enumerate(train_loader):
            train_count += input.size()[0]
            
            input, label = input.to(device), label.to(device)
            
            features = model(input)
            output = metric(features,label)

            _, prediction = torch.max(output,1)
            pred_y = prediction.cpu().data.numpy().squeeze()
            target_y = label.cpu().data.numpy()
            ac = sum(pred_y == target_y)
            train_total_ac += ac
        train_acc = train_total_ac/train_count
        for _,(input,label) in enumerate(test_loader):
            test_count += input.size()[0]
            
            input, label = input.to(device), label.to(device)
            
            features = model(input)
            output = metric(features,label)

            _, prediction = torch.max(output,1)
            pred_y = prediction.cpu().data.numpy().squeeze()
            target_y = label.cpu().data.numpy()
            ac = sum(pred_y == target_y)
            test_total_ac += ac
        test_acc = test_total_ac/test_count
    
    print('train_accuracy:{}'.format(train_acc*100))
    print('test_accuracy:{}'.format(test_acc*100))
    

    return train_acc,test_acc
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_file', type = str, default=askopenfilename(title='Select pretrain model'), help='location where pretrain model is saved')
    parser.add_argument('--blur_path', type=str, default=askdirectory(title='Location for blurry images'), help='directory where blur data are loaded')
    parser.add_argument('--batch_size', type=int, default=5, help='size of each batch')
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='whether to use cuda if available')
    parser.add_argument('--num_class', type=int, default=10, help='number of class')
    opt = parser.parse_args('')
    
    train_acc,test_acc = eval(opt)
