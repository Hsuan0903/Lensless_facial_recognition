# Lensless facial recognition

## Introduction

This project is the implementation for face recognition with optical encryption system via diffuser.

 You can create an environment for this project by `environment.yml` in Windows.

## Function description

`main.py` is the implementation for training classification model.

`model/resnet.py` and `iresnet.py` are the resnet model used in this project.

`metric.py` is the metric function used in this project.

`eval.py` is the implementation for evaluating the general checkpoint (pretrain classification model).

`utils.py` include the functions using in the projects.

`plot_pca_tsne.py` is the implementation for visualizing the features of latent(embedding) space.

`metric_on_MNIST.ipynb` is implementation for the toy examples under the different metric loss on MNIST dataset.

### The **code** and the **dataset** as the following structure:

```
├── Lensless_facial_recognition
  ├── main.py
  ├── eval.py
  ├── utils.py
  ├── plot_pca_tsne.py
  ├── metric_on_MNIST.ipynb
  └── model
    ├── resnet.py
    └── iresnet.py
```
```
├── dataset
  └── train
    ├──class 1
    ├──class 2
    ├──...
    ├──class N
  └── test
    ├──class 1
    ├──class 2
    ├──...
    └──class N
```

## Training stage
* To train the model you can run `main.py`.
* Select the image folder with the same folder structure. 
* Select the saved folder for output data which contain the training parameter, learning curve, accuracy curve, and general checkpoint for inference or resuming training. 

### Naming conventions of saved foledr : (ModelArchitecture_MetricFunction => resnet18_softmax)

```
├── saved folder 
  ├──training parameter.csv
  ├──learning_curve.png(.npz)
  ├──accuracy_curve.png(.npz)
  └──classification.pt(checkpoint of trained model)
```
### Object contents of the `classification.pt` :
```py
# classification.pt
torch.save({
            'model': opt.model,             # Model architecture(resnet18 or iresnet18)
            'metric': opt.metric,           # Metric Function(Softmax, ArcFace, or AdaCos)
            'epoch': epoch,                 
            'model_state_dict': model.state_dict(),
            'metric_state_dict': metric_fc.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'embedding_size': opt.embedding_size,},     # Embedding size of feature
            os.path.join(output_path,'classification.pt'))
```

## Evaluation stage
* To evaluate the model you can run `eval.py`.
* Select the pretrain model file`(classification.pt)`. 
* Select the image folder. 
* It will return the accuracy for training data and testing data.