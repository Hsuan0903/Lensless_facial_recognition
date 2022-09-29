from email.policy import default
import os 
import numpy as np
from tqdm import tqdm
import argparse
from tkinter import Tk
from tkinter.filedialog import askdirectory
from torch.utils.tensorboard import SummaryWriter
import torch
from model.iresnet import iresnet18
from model.resnet import resnet18
from metric import ArcFace, Softmax, AdaCos
from utils import *

def main(opt):
    # set the output path
    if opt.metric == 'softmax':
        output_path = create_folder(opt.save_path,opt.model+'_'+opt.metric)
    elif opt.metric == 'arcface':
        output_path = create_folder(opt.save_path,opt.model+'_'+opt.metric
                                    +'_s_'+str(opt.margin_s)+'_m_'+str(opt.margin_m))
    elif opt.metric == 'adacos':
        output_path = create_folder(opt.save_path,opt.model+'_'+opt.metric)
          
    save_opt(output_path,opt)
    # tensorboard setting
    tensorboard_path = create_folder(output_path ,'log/')
    tensorboard_writer = SummaryWriter(tensorboard_path)

    # load data set
    train_loader, test_loader = dataset(opt.blur_path, opt.batch_size,num_workers=4)

    # device(cuda or CPU)
    device = opt.device
    
    # create model 
    model_dict = {'iresnet18':iresnet18,'resnet18': resnet18,}
    model = model_dict[opt.model](input_channel=opt.input_channel,embedding_size=opt.embedding_size,fc_scale=16).to(device)
    
    if opt.metric == 'softmax':
        metric_fc = Softmax(num_classes=opt.num_class, embedding_size=opt.embedding_size).to(device)
    elif opt.metric == 'arcface':
        metric_fc = ArcFace(num_classes=opt.num_class,embedding_size=opt.embedding_size,margin_m=opt.margin_m,margin_s=opt.margin_s).to(device)
    elif opt.metric == 'adacos':
        metric_fc = AdaCos(num_classes=opt.num_class,embedding_size=opt.embedding_size).to(device)
    else:
        raise ValueError('The loss function must be adacos, arcface or softmax!!!')

    # setup the optimizer, learning rate scheduler and criterion
    optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                  betas=[0.9,0.99], lr=opt.learning_rate, weight_decay=0.0001)
  
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[15,30,45],gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    # create the empty lists for learning curve and accuracy curve
    train_lc, test_lc, train_ac, test_ac = [], [], [], []
    
    epoch_tqdm = tqdm(range(opt.epochs))
    for epoch in epoch_tqdm:
        
        # Training step
        train(train_loader,model,metric_fc,criterion,optimizer,device,scheduler)

        # calculate loss and accuracy  (training data)
        acc, loss,features_list, labels_list = test(train_loader,model,metric_fc,criterion,device)
        train_lc.append(loss)
        train_ac.append(acc)
        features = torch.cat(features_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        
        visualize(features,labels,epoch,tensorboard_writer,'Training',opt.visualize)
        
        
        
        # calculate loss and accuracy  (testing data) 
        acc, loss,features_list, labels_list = test(test_loader,model,metric_fc,criterion,device)
        test_lc.append(loss)
        test_ac.append(acc)
        features = torch.cat(features_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        visualize(features,labels,epoch,tensorboard_writer,'Testing',opt.visualize)


        # tensorboard 
        tensorboard_writer.add_scalar('Loss/Testing ', test_lc[epoch], epoch)
        tensorboard_writer.add_scalar('Loss/Training ', train_lc[epoch], epoch)
        tensorboard_writer.add_scalar('Accuracy/Testing ', test_ac[epoch], epoch)
        tensorboard_writer.add_scalar('Accuracy/Training ', train_ac[epoch], epoch)

        # display loss and accuracy by tqdm module
        epoch_tqdm.set_description(f"Epochs [{epoch}/{opt.epochs}]")
        epoch_tqdm.set_postfix(Accuracy_Training = '%.4f' % (train_ac[epoch]), 
                                Accuracy_Testing = '%.4f' % (test_ac[epoch]),
                                Loss_Training = '%.4f' % (train_lc[epoch]),
                                Loss_Testing = '%.4f' % (test_lc[epoch]),)

    # plot result
    plot_learning_curve(train_lc,test_lc,opt.display_lc,output_path,epoch)
    plot_accuracy_curve(train_ac,test_ac,opt.display_lc,output_path,epoch)

    # save checkpoint and result
    torch.save({
                'model': opt.model,
                'metric': opt.metric,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metric_state_dict': metric_fc.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'input_channel': opt.input_channel,
                'embedding_size': opt.embedding_size,
                'num_class': opt.num_class,
                'margin_m': opt.margin_m,
                'margin_s': opt.margin_s,},
                os.path.join(output_path,'classification.pt'))
    np.savez(os.path.join(output_path,'learning_curve.npz'),train_lc=train_lc, test_lc=test_lc)
    np.savez(os.path.join(output_path,'accuracy_curve.npz'),train_ac=train_ac, test_ac=test_ac)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hyper parameter of classification model')
    parser.add_argument('--blur_path', type=str, default=askdirectory(title='Location for blurry images'), help='directory where blur data are loaded')
    parser.add_argument('--save_path', type=str, default=askdirectory(title='Location for output data'), help='directory where output data are saved')
    parser.add_argument('--epochs', type=int, default=50, help='number of epoch')
    parser.add_argument('--batch_size', type=int, default=8, help='size of each batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    
    parser.add_argument('--embedding_size', type=int, default=512, help='embedding size of feature')
    parser.add_argument('--input_channel', type=int, default=1, help='number of input channel')
    parser.add_argument('--num_class', type=int, default=10, help='number of class')
    
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='whether to use cuda if available' )
    
    parser.add_argument('--model', type=str, default='resnet18',choices=['iresnet18','resnet18'],help='iresnet18,resnet18')
    parser.add_argument('--metric', type=str, default='arcface',choices=['softmax','arcface','adacos'],help='softmax, arcface, adacos')
    parser.add_argument('--margin_s', type=float, default=32.,help='margin s for arcface metric function')
    parser.add_argument('--margin_m', type=float, default=0.1,help='margin m for arcface metric function')
    
    parser.add_argument('--visualize', type=str, default='PCA', choices=['PCA','TSNE'],help='PCA or TSNE, the method of visualizing the embedding feature')
    parser.add_argument('--display_lc', default=False, action='store_true',help='whether to display the learning curve')
    opt = parser.parse_args('')
    
    Tk().iconify()
    print_opt(opt)
    main(opt)
