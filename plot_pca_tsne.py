import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import importlib
from tqdm import tqdm
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
from model.resnet import *
from model.iresnet import *
from metric import ArcFace, Softmax, AdaCos
from utils import *
from torch.autograd import Variable
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

class SaveFeatures():
    def __init__(self,module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self,module,input,output):
        self.features = output

    def close(self):
        self.hook.remove()

# enable HiDPI scaling if using Qt as backend
spec = importlib.util.find_spec('PyQt5')
if spec is not None:
    import PyQt5 as qt
    if plt.get_backend() == 'Qt5Agg':
        qt.QtWidgets.QApplication.setAttribute(qt.QtCore.Qt.AA_EnableHighDpiScaling, True)
        
#%% parameters
BS=40
layer = -1 # n-th layer
n_components = 3 # number of principal components
print_target_layer = True # if print target layer
method = 'PCA' # dimensionality reduction method, 'PCA' or 'TSNE'
dim_z = 512 # latent dimensions
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%%
Tk().iconify()
netfile_name = askopenfilename(title='Select netfile for analyzing')
test_path = askdirectory(title='Location for blurry images')
checkpoint = torch.load(netfile_name)
train_loader, test_loader = dataset(test_path, BS, 0)
print(netfile_name)
print(test_path)
#%%
model_dict = {'iresnet18':iresnet18,'resnet18': resnet18,}
model = model_dict[checkpoint['model']](input_channel=checkpoint['input_channel'],embedding_size=checkpoint['embedding_size'],fc_scale=16).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

children = list(model.children())
print(children[-1])

activations = SaveFeatures(children[-1])
for param in model.parameters():
    param.requires_grad = False
#%%
def reducing_dimension(test_loader,method):
     
    descriptors = np.zeros((1,dim_z))
    label = np.zeros((1,))
    for step,(b_x,b_y) in enumerate(test_loader):
        b_x = Variable(b_x).to(device)
        b_y = Variable(b_y).to(device)
        output = model(b_x)
        descriptors = np.append(descriptors, activations.features.cpu(), axis=0)
        label = np.append(label, b_y.cpu())

    descriptors = np.delete(descriptors, 0, 0)
    label = np.delete(label, 0)

    print('reducing dimension')

    if method == 'TSNE':
        embedded = TSNE(n_components=n_components).fit_transform(descriptors)

    elif method == 'PCA':
        embedded = PCA(n_components=n_components).fit_transform(descriptors)
    else:
        exit('{method} method is not available')
        
    group = np.array(label)
    return group,embedded

test_group,test_embedded = reducing_dimension(test_loader,method)
train_group,train_embedded = reducing_dimension(train_loader,method)
#%%
color_set  = ["orange","purple","olive","brown","gray","cyan","magenta","red","green","blue",'greenyellow']

def plot2d(group,embedded,name='pca_2d.png'):
    plt.figure()
    plt.rcParams.update({'font.family':"Times New Roman",
                        'font.size': 14,
                        'mathtext.fontset':"stix"})
    plt.tight_layout()
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid()
    for g in range(10):
        ix = np.where(group==g)
        plt.scatter(embedded[ix,0],embedded[ix,1], label=g+1,c=color_set[g])
        plt.annotate(g+1,(np.mean(embedded[ix[0][0:-1],0]),
                          np.mean(embedded[ix[0][0:-1],1])),c='black')
        plt.fill(embedded[ix[0][0:-1],0],
                 embedded[ix[0][0:-1],1],
                 c=color_set[g],alpha=0.3)
        plt.legend(title='class', shadow=True,bbox_to_anchor=(1, 1.03),loc='upper left')

    plt.savefig(name,bbox_inches="tight",dpi=1000)        
    plt.show()


def rot_fig(angle):
    ax.view_init(azim=angle)

def plot3d(group,embedded,name='pca_3d.png'):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plt.rcParams.update({'font.family':"Times New Roman",
                        'font.size': 14,
                        'mathtext.fontset':"stix"})
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.grid()
    plt.tight_layout()
    
    for g in range(10):
        ix = np.where(group == g)
        ax.scatter(embedded[ix,0],embedded[ix,1],embedded[ix,2],label=g+1,c=color_set[g])
        mean_x,mean_y,mean_z = np.mean(embedded[ix[0][0:-1],0]),np.mean(embedded[ix[0][0:-1],1]),np.mean(embedded[ix[0][0:-1],2])
        ax.text(mean_x,mean_y,mean_z,g+1)

    plt.legend(title='Class', shadow=True,bbox_to_anchor=(1.02, 1.03),loc='upper left',prop={'size': 14})


    plt.show()  
    angle = 1
    ani = animation.FuncAnimation(fig,rot_fig,frames=np.arange(0, 360, angle), interval=50)
    ani.save('exp2_pca_3d-2.gif', writer=animation.PillowWriter(fps=20))
    plt.show()

if __name__ == '__main__':
    plot2d(train_group,train_embedded)
    plot3d(train_group,train_embedded)
