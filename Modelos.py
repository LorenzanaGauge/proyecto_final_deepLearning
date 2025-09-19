from collections import OrderedDict
from inspect import classify_class_attrs
from numpy import average
import pytorch_lightning as pl
import torch
import torchmetrics
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
# For PyTorch Lightning 2.x and above
from lightning.pytorch.core import LightningModule
from torch.nn import functional as F
from torchmetrics import Accuracy,ConfusionMatrix,AveragePrecision
from torchvision import models
import torchvision
from coatnet import CoAtNet


def calcular_matriz_confusion(num_classes,outputs):
    preds = torch.cat([elem['preds'] for elem in outputs ])
    targets = torch.cat([elem['targets'] for elem in outputs])
    preds = preds.cpu()
    targets = targets.cpu()
    conf = ConfusionMatrix(task="multiclass",num_classes=num_classes,normalize='true')
    matriz = conf(preds,targets)
    fig = plt.figure(figsize = (10,8))
    ax = sns.heatmap(matriz, annot=False, square=True, xticklabels=5, yticklabels=5, cmap="inferno")
    return fig


class AgoModel(pl.LightningModule):
    def __init__(self,num_clases,lr=0.001,optim='SGD', ridge=0, dropout=0.3):
        super().__init__()
        # self.modelo = models.resnext50_32x4d(pretrained=True)
        # self.modelo.fc = nn.Linear(2048,num_clases)
        # self.modelo.aux_logits = False
        try:
            assert num_clases is not None
        except AssertionError:
            raise ValueError("num_clases no puede ser None")
        self.num_clases=num_clases
        self.lr=lr
        self.ridge=ridge
        self.optim=optim
        self.dropout=dropout
        self.save_hyperparameters(ignore='num_clases')
        #self.log("params",self.hparams)
        print(f"Usando LR:{self.lr}")
        self.ConfusionMatrix = ConfusionMatrix(task="multiclass", num_classes=self.num_clases)
        self.train_accuracy= torchmetrics.Accuracy(task="multiclass",top_k=1, num_classes=self.num_clases)
        self.train_accuracy_top3= torchmetrics.Accuracy(task="multiclass",top_k=3, num_classes=self.num_clases)
        self.train_accuracy_top5= torchmetrics.Accuracy(task="multiclass",top_k=5, num_classes=self.num_clases)
        self.avg_train_precision = torchmetrics.AveragePrecision(task="multiclass",num_classes=num_clases)
        self.avg_train_precision_weighted = torchmetrics.AveragePrecision(task="multiclass",num_classes=num_clases,average='weighted')
        
        self.val_accuracy=torchmetrics.Accuracy(task="multiclass",top_k=1, num_classes=self.num_clases)
        self.val_accuracy_top3=torchmetrics.Accuracy(task="multiclass",top_k=3, num_classes=self.num_clases)
        self.val_accuracy_top5=torchmetrics.Accuracy(task="multiclass",top_k=5, num_classes=self.num_clases)
        self.avg_val_precision = torchmetrics.AveragePrecision(task="multiclass",num_classes=self.num_clases)
        self.avg_val_precision_weighted = torchmetrics.AveragePrecision(task="multiclass",num_classes=self.num_clases,average='weighted')
        self.validation_outputs = []
        self.training_outputs = []
    def forward(self,x):
        return self.modelo(x)

    # Using custom or multiple metrics (default_hp_metric=False)
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/val_loss": 0,
                                                   "hp/val_acc":0})

    def configure_optimizers(self):
        if self.optim=='SGD':
            opt = torch.optim.SGD(self.modelo.parameters(),lr= self.lr, weight_decay=self.ridge, momentum=0.9)
        elif self.optim=='Adam':
            opt = torch.optim.Adam(self.modelo.parameters(),lr=self.lr, weight_decay=self.ridge)
        elif self.optim=='Adagrad':
            opt = torch.optim.Adagrad(self.modelo.parameters(),lr=self.lr, weight_decay=self.ridge)
        
        #scheduler =ReduceLROnPlateau(optimizer=opt,mode='min',factor=0.7, patience=3, threshold=0.25, 
        #min_lr=self.lr*0.1, verbose=True )
        return {
            'optimizer':opt,
            'lr_scheduler':{
                'scheduler':ReduceLROnPlateau(optimizer=opt,mode='min',factor=0.7, patience=3, threshold=0.25, 
        min_lr=self.lr*0.1),
                'monitor':'val_loss',
                'interval':'epoch'
            }
        }
    
    def training_step(self,batch,batch_idx):
        self.modelo=self.modelo.train()
        imgs, tag = batch
        pred = self.modelo(imgs)
        loss = F.cross_entropy(pred,tag) 
        #self.log('train_acc_step',self.train_accuracy(pred,tag))
        #self.log('train_loss_step',loss.detach())
        
        self.train_accuracy(pred, tag)
        self.train_accuracy_top3(pred, tag)
        self.train_accuracy_top5(pred, tag)
        self.avg_train_precision(pred,tag)
        self.avg_train_precision_weighted(pred,tag)
        
        self.log('train_acc', self.train_accuracy, on_step=False, on_epoch=True)
        self.log('train_acc_t3', self.train_accuracy_top3, on_step=False, on_epoch=True)
        self.log('train_acc_t5', self.train_accuracy_top5, on_step=False, on_epoch=True)
        self.log('train_loss',loss.detach(),on_epoch=True, on_step=False)
        self.log('train_avg_precision',self.avg_train_precision,on_step=False,on_epoch=True)
        self.log('train_avg_precision_w',self.avg_train_precision_weighted,on_step=False,on_epoch=True)

        output = {'loss':loss,'preds':pred.detach(),'targets':tag}
        self.training_outputs.append(output)
        return output
    
    def on_train_epoch_end(self):
        outs = self.training_outputs
        self.logger.experiment.add_figure('train_confussion_matrix',
                                          calcular_matriz_confusion(self.num_clases, outs),
                                          global_step=self.current_epoch)
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)
        self.training_outputs = []  # reset for next epoch
    
    def validation_step(self,batch,batch_idx):
        self.modelo=self.modelo.eval()
        imgs, tag = batch
        pred = self.modelo(imgs)
        loss = F.cross_entropy(pred,tag) 
        #self.log('val_acc_step',self.val_accuracy(pred,tag))
        #self.log('val_loss_step',loss.detach())

        self.val_accuracy(pred, tag)
        self.val_accuracy_top3(pred, tag)
        self.val_accuracy_top5(pred, tag)
        self.avg_val_precision(pred,tag)
        self.avg_val_precision_weighted(pred,tag)
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True)
        self.log('val_acc_t3', self.val_accuracy_top3, on_step=False, on_epoch=True)
        self.log('val_acc_t5', self.val_accuracy_top5, on_step=False, on_epoch=True)
        self.log('val_loss',loss.detach(),on_epoch=True, on_step=False)
        self.log('val_avg_precision',self.avg_val_precision,on_step=False,on_epoch=True)
        self.log('val_avg_precision_w',self.avg_val_precision_weighted,on_step=False,on_epoch=True)

        self.log('hp/val_loss',loss.detach(),on_epoch=True, on_step=False)
        self.log('hp/val_acc', self.val_accuracy, on_step=False, on_epoch=True)

        output = {'loss':loss,'preds':pred.detach(),'targets':tag}
        self.validation_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        outs = self.validation_outputs
        self.logger.experiment.add_figure('val_confussion_matrix',
                                          calcular_matriz_confusion(self.num_clases, outs),
                                          global_step=self.current_epoch)
        self.validation_outputs = []  # reset for next epoch

class AgoEfficienet(AgoModel):
    def __init__(self,num_classes,lr=0.001,optim='SGD'):
        super().__init__(num_classes,lr,optim)
        
        m = models.efficientnet_b7(pretrained=True)
        m.classifier  = torch.nn.Sequential(
                            nn.Dropout(0.5), nn.Linear(2560,num_classes,bias=True)
                        )                                      
        self.modelo = m



class AgoInception(AgoModel):
    def __init__(self,num_classes,lr=0.001,optim='SGD'):
        super().__init__(num_classes,lr,optim)
        self.modelo = models.inception_v3(pretrained=True,progress=True)
        self.modelo.fc = nn.Linear(2048,num_classes)
        self.modelo.aux_logits=False

class AgoResnext50(AgoModel):
    def __init__(self,num_classes,lr=0.001,optim='SGD',ridge=0):
        super().__init__(num_classes,lr,optim)
        self.modelo = models.resnext50_32x4d(pretrained=True,progress=True)
        self.modelo.fc = nn.Linear(2048,num_classes)

class AgoResnext101(AgoModel):
    def __init__(self,num_classes,lr=0.001,optim='SGD',ridge=0):
        super().__init__(num_classes,lr,optim)
        self.modelo = models.resnext101_32x8d(pretrained=True,progress=True)
        self.modelo.fc = nn.Linear(2048,num_classes)

class AgoResnext102(AgoModel):
    def __init__(self,num_classes,lr=0.001,optim='SGD'):
        super().__init__(num_classes,lr,optim)
        self.modelo = models.resnext101_32x8d(pretrained=True,progress=True)
        self.modelo.fc = nn.Sequential(OrderedDict([
                                            ('fc1',nn.Linear(2048,1000)),
                                            ('relu1',nn.ReLU()),
                                            ('drop1',nn.Dropout(p=0.3)),
                                            ('fc2',nn.Linear(1000,500)),
                                            ('relu2', nn.ReLU()),
                                            ('drop2', nn.Dropout(p=0.3)),
                                            ('fc3', nn.Linear(500,100)),
                                            ('relu3', nn.ReLU()),
                                            ('drop3', nn.Dropout(p=0.3)),
                                            ('fc4', nn.Linear(100,100)),
                                            ('relu4', nn.ReLU()),
                                            ('drop4', nn.Dropout(p=0.3)),
                                            ('output', nn.Linear(100, num_classes))
                                        ]))

class AgoResnext103(AgoModel):
    def __init__(self,num_classes,lr=0.001,optim='SGD'):
        super().__init__(num_classes,lr,optim)
        self.modelo = models.resnext101_32x8d(pretrained=True,progress=True)
        self.modelo.fc = nn.Sequential(OrderedDict([
                                            ('fc1',nn.Linear(2048,1000)),
                                            ('relu1',nn.ReLU()),
                                            ('drop1',nn.Dropout(p=0.3)),
                                            ('fc4', nn.Linear(1000,100)),
                                            ('relu4', nn.ReLU()),
                                            ('drop4', nn.Dropout(p=0.3)),
                                            ('output', nn.Linear(100, num_classes))
                                        ]))


class AgoResnet50(AgoModel):
    def __init__(self,num_classes,lr=0.001,optim='SGD'):
        super().__init__(num_classes,lr,optim)
        self.modelo = models.resnet50(pretrained=True,progress=True)
        self.modelo.fc = nn.Linear(2048,num_classes)

class AgoResnet18(AgoModel):
    def __init__(self,num_classes,lr=0.001,optim='SGD'):
        super().__init__(num_classes,lr,optim)
        self.modelo = models.resnet18(pretrained=True,progress=True)
        self.modelo.fc = nn.Linear(2048,num_classes)

class AgoMobilenet(AgoModel):
    def __init__(self,num_classes,lr=0.001,optim='SGD'):
        super().__init__(num_classes,lr,optim)
        self.modelo = models.mobilenet.mobilenet_v2(pretrained=True,progress=True)
        self.modelo.classifier = nn.Sequential(
                      nn.Dropout(0.2),
                      nn.Linear(self.modelo.last_channel, num_classes),
                      )

class AgoSqueeze(AgoModel):
    def __init__(self,num_classes,lr=0.001,optim='SGD'):
        super().__init__(num_classes,lr,optim)
        self.modelo = models.squeezenet1_1(pretrained=True,progress=True)
        final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        self.modelo.classifier =  nn.Sequential(
                                  nn.Dropout(p=0.5),
                                  final_conv,
                                  nn.ReLU(inplace=True),
                                  nn.AdaptiveAvgPool2d((1, 1))
                                  )

class AgoVgg16(AgoModel):
    def __init__(self, num_classes,lr=0.001,optim='SGD', ridge=0):
        super().__init__(num_classes,lr,optim, ridge)
        self.modelo=models.vgg16(pretrained=True, progress=True)
        clasificador=nn.Linear(1000, num_classes)

        self.modelo =nn.Sequential(
            self.modelo,
            clasificador
        )

class AgoResnet18(AgoModel):
    def __init__(self,num_classes,lr=0.001,optim='SGD', ridge=0):
        super().__init__(num_classes,lr,optim,ridge)
        self.modelo = models.resnet18(pretrained=True,progress=True)
        self.modelo.fc=nn.Linear(self.modelo.fc.in_features, num_classes)
        #clasificador = nn.Linear(2048,num_classes)

        #self.modelo =nn.Sequential(
        #    self.modelo,
        #    clasificador
        #)
class AgoResnet34(AgoModel):
    def __init__(self,num_classes,lr=0.001,optim='SGD', ridge=0):
        super().__init__(num_classes,lr,optim,ridge)
        self.modelo = models.resnet34(pretrained=True,progress=True)
        self.modelo.fc=nn.Linear(self.modelo.fc.in_features, num_classes)

class AgoResnet50(AgoModel):
    def __init__(self,num_classes,lr=0.001,optim='SGD', ridge=0):
        super().__init__(num_classes,lr,optim,ridge)
        self.modelo = models.resnet50(pretrained=True,progress=True)
        self.modelo.fc=nn.Linear(self.modelo.fc.in_features, num_classes)
        
class AgoResnet101(AgoModel):
    def __init__(self,num_classes,lr=0.001,optim='SGD', ridge=0):
        super().__init__(num_classes,lr,optim,ridge)
        self.modelo = models.resnet101(pretrained=True,progress=True)
        self.modelo.fc=nn.Linear(self.modelo.fc.in_features, num_classes)
        
class AgoResnet152(AgoModel):
    def __init__(self,num_classes,lr=0.001,optim='SGD', ridge=0):
        super().__init__(num_classes,lr,optim,ridge)
        self.modelo = models.resnet152(pretrained=True,progress=True)
        self.modelo.fc=nn.Linear(self.modelo.fc.in_features, num_classes)
        
##################################Handkrafted Models################################


################################################################################################
class Model000(nn.Module):
    def __init__(self,num_classes ,dropout=0.3):
        super().__init__()
        self.model=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding='same')),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm2d(num_features=3)),
            ('flat', nn.Flatten()),
            ('drop1', nn.Dropout(p=dropout)),
            ('dense1', nn.Linear(in_features=400*400*3, out_features=500)),
            ('activation2', nn.Sigmoid()),
            ('batch2', nn.BatchNorm1d(num_features=500)),
            ('drop2', nn.Dropout(p=dropout)),
            ('dense2', nn.Linear(in_features=500, out_features=500)),
            ('activation3', nn.Sigmoid()),
            ('batch3', nn.BatchNorm1d(num_features=500)),
            ('drop3', nn.Dropout(p=dropout)),
            ('fc', nn.Linear(in_features=500, out_features=num_classes))
        ]))
        
    def forward(self,x):
        return self.model(x)
    
class AgoModel000(AgoModel):
    def __init__(self,num_classes,lr=0.001,optim='SGD', ridge=0, dropout=0.3):
        super().__init__(num_classes,lr,optim,ridge, dropout)
        self.modelo=Model000(num_classes, dropout)
        
##############################################################################################

##############################################################################################

class Model0009(nn.Module):
    def __init__(self, num_classes, dropout):
        
        self.initbloque=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, padding='same')),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm2d(num_features=12)),
            
        ]))
        
        self.bloque1=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding='same')),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm2d(num_features=12)),
            
        ]))
        
        self.fc=nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(in_features=200*200*12, out_features=500)),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm1d(num_features=500)),
            ('drop1', nn.Dropout(p=dropout)),
            ('linear2', nn.Linear(in_features=500, out_features=500)),
            ('activation2', nn.Sigmoid()),
            ('batch2', nn.BatchNorm1d(num_features=500)),
            ('drop2', nn.Dropout(p=dropout)),
            ('linear3', nn.Linear(in_features=500, out_features=num_classes))
            
        ]))
        
        self.modelo=nn.Sequential(OrderedDict([
            
            ('bloque1', self.initbloque),
            ('bloque2', self.bloque1),
            ('max1', nn.MaxPool2d(kernel_size=2)),
            ('bloque3', self.bloque1),
            ('bloque4', self.bloque1),
            ('bloque5', self.bloque1),
            ('flat', nn.Flatten()),
            ('drop1', nn.Dropout(p=dropout)),
            ('fc', self.fc)           
            
        ]))
        
    def forward(self,x):
        return self.modelo(x)
    
class AgoModel0009(AgoModel):
    def __init__(self,num_classes,lr=0.001,optim='SGD', ridge=0, dropout=0.3):
        super().__init__(num_classes,lr,optim,ridge, dropout)
        self.modelo=Model0009(num_classes, dropout)
        
#######################################################################################

#######################################################################################

class Model0008(nn.Module):
    def __init__(self, num_classes, dropout):
        self.initbloque=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, padding='same')),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm2d(num_features=12)),
            
        ]))
        
        self.bloque1=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding='same')),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm2d(num_features=12)),
            
        ]))
        
        self.fc=nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(in_features=100*100*12, out_features=500)),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm1d(num_features=500)),
            ('drop1', nn.Dropout(p=dropout)),
            ('linear2', nn.Linear(in_features=500, out_features=500)),
            ('activation2', nn.Sigmoid()),
            ('batch2', nn.BatchNorm1d(num_features=500)),
            ('drop2', nn.Dropout(p=dropout)),
            ('linear3', nn.Linear(in_features=500, out_features=num_classes))
            
        ]))
        
        self.modelo=nn.Sequential(OrderedDict([
            ('bloque1', self.initbloque),
            ('bloque2', self.bloque1),
            ('max1', nn.MaxPool2d(kernel_size=2)),
            ('bloque3', self.bloque1),
            ('bloque4', self.bloque1),
            ('bloque5', self.bloque1),
            ('max2', nn.MaxPool2d(kernel_size=2)),
            ('flat', nn.Flatten()),
            ('drop1', nn.Dropout(p=dropout)),
            ('fc', self.fc)           
            
        ]))
        
    def forward(self,x):
        return self.modelo(x)
    
class AgoModel0008(AgoModel):
    def __init__(self,num_classes,lr=0.001,optim='SGD', ridge=0, dropout=0.3):
        super().__init__(num_classes,lr,optim,ridge, dropout)
        self.modelo=Model0008(num_classes, dropout)

################################################################################

#################################################################################

class Model0007(nn.Module):
    def __init__(self, num_classes, dropout):
        
        self.initbloque=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, padding='same')),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm2d(num_features=12)),
            
        ]))
        
        self.bloque1=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding='same')),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm2d(num_features=12)),
            
        ]))
        
        self.fc=nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(in_features=200*200*12, out_features=500)),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm1d(num_features=500)),
            ('drop1', nn.Dropout(p=dropout)),
            ('linear2', nn.Linear(in_features=500, out_features=500)),
            ('activation2', nn.Sigmoid()),
            ('batch2', nn.BatchNorm1d(num_features=500)),
            ('drop2', nn.Dropout(p=dropout)),
            ('linear3', nn.Linear(in_features=500, out_features=num_classes))
            
        ]))
        
        self.modelo=nn.Sequential(OrderedDict([
            ('bloque1', self.initbloque),
            ('bloque2', self.bloque1),
            ('max1', nn.MaxPool2d(kernel_size=2)),
            ('bloque3', self.bloque1),
            ('bloque4', self.bloque1),
            ('flat', nn.Flatten()),
            ('drop1', nn.Dropout(p=dropout)),
            ('fc', self.fc)           
            
        ]))
        
    def forward(self,x):
        return self.modelo(x)
    
class AgoModel0007(AgoModel):
    def __init__(self,num_classes,lr=0.001,optim='SGD', ridge=0, dropout=0.3):
        super().__init__(num_classes,lr,optim,ridge, dropout)
        self.modelo=Model0007(num_classes, dropout)
        
##################################################################################

##################################################################################

class Model0006(nn.Module):
    def __init__(self, num_classes, dropout):
        
        self.initbloque=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, padding='same')),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm2d(num_features=12)),
            
        ]))
        
        self.bloque1=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding='same')),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm2d(num_features=12)),
            
        ]))
        
        self.fc=nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(in_features=200*200*12, out_features=500)),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm1d(num_features=500)),
            ('drop1', nn.Dropout(p=dropout)),
            ('linear2', nn.Linear(in_features=500, out_features=500)),
            ('activation2', nn.Sigmoid()),
            ('batch2', nn.BatchNorm1d(num_features=500)),
            ('drop2', nn.Dropout(p=dropout)),
            ('linear3', nn.Linear(in_features=500, out_features=num_classes))
            
        ]))
        
        self.modelo=nn.Sequential(OrderedDict([
            ('bloque1', self.initbloque),
            ('bloque2', self.bloque1),
            ('max1', nn.MaxPool2d(kernel_size=2)),
            ('bloque3', self.bloque1),
            ('flat', nn.Flatten()),
            ('drop1', nn.Dropout(p=dropout)),
            ('fc', self.fc)           
            
        ]))
        
    def forward(self,x):
        return self.modelo(x)
    
class AgoModel0006(AgoModel):
    def __init__(self,num_classes,lr=0.001,optim='SGD', ridge=0, dropout=0.3):
        super().__init__(num_classes,lr,optim,ridge, dropout)
        self.modelo=Model0009(num_classes, dropout)
        
        
#################################################################################

#################################################################################

class Model0005(nn.Module):
    def __init__(self, num_classes, dropout):
        
        self.initbloque=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, padding='same')),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm2d(num_features=12)),
            
        ]))
        
        self.bloque1=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding='same')),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm2d(num_features=12)),
            
        ]))
        
        self.fc=nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(in_features=400*400*12, out_features=500)),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm1d(num_features=500)),
            ('drop1', nn.Dropout(p=dropout)),
            ('linear2', nn.Linear(in_features=500, out_features=500)),
            ('activation2', nn.Sigmoid()),
            ('batch2', nn.BatchNorm1d(num_features=500)),
            ('drop2', nn.Dropout(p=dropout)),
            ('linear3', nn.Linear(in_features=500, out_features=num_classes))
            
        ]))
        
        self.modelo=nn.Sequential(OrderedDict([
            ('bloque1', self.initbloque),
            ('bloque2', self.bloque1),
            ('flat', nn.Flatten()),
            ('drop1', nn.Dropout(p=dropout)),
            ('fc', self.fc)           
            
        ]))
        
    def forward(self,x):
        return self.modelo(x)
    
class AgoModel0005(AgoModel):
    def __init__(self,num_classes,lr=0.001,optim='SGD', ridge=0, dropout=0.3):
        super().__init__(num_classes,lr,optim,ridge, dropout)
        self.modelo=Model0005(num_classes, dropout)
        
        
##################################################################################

##################################################################################

class Model0004(nn.Module):
    def __init__(self, num_classes, dropout):
        
        self.initbloque=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, padding='same')),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm2d(num_features=12)),
            
        ]))
        
        self.bloque1=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding='same')),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm2d(num_features=12)),
            
        ]))
        
        self.fc=nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(in_features=400*400*12, out_features=500)),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm1d(num_features=500)),
            ('drop1', nn.Dropout(p=dropout)),
            ('linear2', nn.Linear(in_features=500, out_features=500)),
            ('activation2', nn.Sigmoid()),
            ('batch2', nn.BatchNorm1d(num_features=500)),
            ('drop2', nn.Dropout(p=dropout)),
            ('linear3', nn.Linear(in_features=500, out_features=num_classes))
            
        ]))
        
        self.modelo=nn.Sequential(OrderedDict([
            ('bloque1', self.initbloque),
            ('flat', nn.Flatten()),
            ('drop1', nn.Dropout(p=dropout)),
            ('fc', self.fc)           
            
        ]))
        
    def forward(self,x):
        return self.modelo(x)
    
class AgoModel0004(AgoModel):
    def __init__(self,num_classes,lr=0.001,optim='SGD', ridge=0, dropout=0.3):
        super().__init__(num_classes,lr,optim,ridge, dropout)
        self.modelo=Model0004(num_classes, dropout)
        
##################################################################################

##################################################################################

class Model0003(nn.Module):
    def __init__(self, num_classes, dropout):
        
        self.initbloque=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=3, out_channels=9, kernel_size=3, padding='same')),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm2d(num_features=9)),
            
        ]))
        
        self.bloque1=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=9, out_channels=9, kernel_size=3, padding='same')),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm2d(num_features=9)),
            
        ]))
        
        self.fc=nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(in_features=400*400*9, out_features=500)),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm1d(num_features=500)),
            ('drop1', nn.Dropout(p=dropout)),
            ('linear2', nn.Linear(in_features=500, out_features=500)),
            ('activation2', nn.Sigmoid()),
            ('batch2', nn.BatchNorm1d(num_features=500)),
            ('drop2', nn.Dropout(p=dropout)),
            ('linear3', nn.Linear(in_features=500, out_features=num_classes))
            
        ]))
        
        self.modelo=nn.Sequential(OrderedDict([
            ('bloque1', self.initbloque),
            ('flat', nn.Flatten()),
            ('drop1', nn.Dropout(p=dropout)),
            ('fc', self.fc)           
            
        ]))
        
    def forward(self,x):
        return self.modelo(x)
    
class AgoModel0003(AgoModel):
    def __init__(self,num_classes,lr=0.001,optim='SGD', ridge=0, dropout=0.3):
        super().__init__(num_classes,lr,optim,ridge, dropout)
        self.modelo=Model0003(num_classes, dropout)
        
####################################################################################

#####################################################################################

class Model0002(nn.Module):
    def __init__(self, num_classes, dropout):
        super().__init__()
        
        self.initbloque=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding='same')),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm2d(num_features=6)),
            
        ]))
        
        self.bloque1=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, padding='same')),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm2d(num_features=9)),
            
        ]))
        
        self.fc=nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(in_features=400*400*6, out_features=500)),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm1d(num_features=500)),
            ('drop1', nn.Dropout(p=dropout)),
            ('linear2', nn.Linear(in_features=500, out_features=500)),
            ('activation2', nn.Sigmoid()),
            ('batch2', nn.BatchNorm1d(num_features=500)),
            ('drop2', nn.Dropout(p=dropout)),
            ('linear3', nn.Linear(in_features=500, out_features=num_classes))
            
        ]))
        
        self.model=nn.Sequential(OrderedDict([
            ('bloque1', self.initbloque),
            ('flat', nn.Flatten()),
            ('drop1', nn.Dropout(p=dropout)),
            ('fc', self.fc)           
            
        ]))
        
    def forward(self,x):
        return self.model(x)
    
class AgoModel0002(AgoModel):
    def __init__(self,num_classes,lr=0.001,optim='SGD', ridge=0, dropout=0.3):
        super().__init__(num_classes,lr,optim,ridge, dropout)
        self.modelo=Model0002(num_classes, dropout)
        
################################################################################

###############################################################################

class Model0001(nn.Module):
    def __init__(self, num_classes, dropout):
        super().__init__()
        self.initbloque=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding='same')),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm2d(num_features=32)),
            
        ]))
        
        self.bloque1=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm2d(num_features=32)),
            
        ]))
        
        self.fc=nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(in_features=400*400*32, out_features=500)),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm1d(num_features=500)),
            ('drop1', nn.Dropout(p=dropout)),
            ('linear2', nn.Linear(in_features=500, out_features=500)),
            ('activation2', nn.Sigmoid()),
            ('batch2', nn.BatchNorm1d(num_features=500)),
            ('drop2', nn.Dropout(p=dropout)),
            ('linear3', nn.Linear(in_features=500, out_features=num_classes))
            
        ]))
        
        self.model=nn.Sequential(OrderedDict([
            ('bloque1', self.initbloque),
            ('flat', nn.Flatten()),
            ('drop1', nn.Dropout(p=dropout)),
            ('fc', self.fc)           
            
        ]))
        
    def forward(self,x):
        return self.model(x)
    
class AgoModel0001(AgoModel):
    def __init__(self,num_classes,lr=0.001,optim='SGD', ridge=0, dropout=0.3):
        super().__init__(num_classes,lr,optim,ridge, dropout)
        self.modelo=Model0001(num_classes, dropout)
        
#################################################################################

#################################################################################

class Model0(nn.Module):
    def __init__(self, num_classes, dropout):
        
        self.initbloque=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding='same')),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm2d(num_features=32)),
            
        ]))
        
        self.bloque1=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm2d(num_features=32)),
            
        ]))
        
        self.fc=nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(in_features=200*200*32, out_features=500)),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm1d(num_features=500)),
            ('drop1', nn.Dropout(p=dropout)),
            ('linear2', nn.Linear(in_features=500, out_features=500)),
            ('activation2', nn.Sigmoid()),
            ('batch2', nn.BatchNorm1d(num_features=500)),
            ('drop2', nn.Dropout(p=dropout)),
            ('linear3', nn.Linear(in_features=500, out_features=num_classes))
            
        ]))
        
        self.modelo=nn.Sequential(OrderedDict([
            ('bloque1', self.initbloque),
            ('maxpool1', nn.MaxPool2d(kernel_size=2)),
            ('flat', nn.Flatten()),
            ('drop1', nn.Dropout(p=dropout)),
            ('fc', self.fc)           
            
        ]))
        
    def forward(self,x):
        return self.modelo(x)
    
class AgoModel0(AgoModel):
    def __init__(self,num_classes,lr=0.001,optim='SGD', ridge=0, dropout=0.3):
        super().__init__(num_classes,lr,optim,ridge, dropout)
        self.modelo=Model0(num_classes, dropout)

####################################################################################

####################################################################################

class Model1(nn.Module):
    def __init__(self, num_classes, dropout):
        
        self.initbloque=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding='same')),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm2d(num_features=32)),
            
        ]))
        
        self.bloque1=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm2d(num_features=32)),
            
        ]))
        
        self.fc=nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(in_features=200*200*32, out_features=500)),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm1d(num_features=500)),
            ('drop1', nn.Dropout(p=dropout)),
            ('linear2', nn.Linear(in_features=500, out_features=500)),
            ('activation2', nn.Sigmoid()),
            ('batch2', nn.BatchNorm1d(num_features=500)),
            ('drop2', nn.Dropout(p=dropout)),
            ('linear3', nn.Linear(in_features=500, out_features=num_classes))
            
        ]))
        
        self.modelo=nn.Sequential(OrderedDict([
            ('bloque1', self.initbloque),
            ('bloque2', self.bloque1),
            ('maxpool1', nn.MaxPool2d(kernel_size=2)),
            ('flat', nn.Flatten()),
            ('drop1', nn.Dropout(p=dropout)),
            ('fc', self.fc)           
            
        ]))
        
    def forward(self,x):
        return self.modelo(x)
    
class AgoModel1(AgoModel):
    def __init__(self,num_classes,lr=0.001,optim='SGD', ridge=0, dropout=0.3):
        super().__init__(num_classes,lr,optim,ridge, dropout)
        self.modelo=Model1(num_classes, dropout)
        
######################################################################################

######################################################################################

class Model11(nn.Module):
    def __init__(self, num_classes, dropout):
        
        self.initbloque=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding='same')),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm2d(num_features=64)),
            
        ]))
        
        self.bloque1=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same')),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm2d(num_features=64)),
            
        ]))
        
        self.fc=nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(in_features=200*200*64, out_features=500)),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm1d(num_features=500)),
            ('drop1', nn.Dropout(p=dropout)),
            ('linear2', nn.Linear(in_features=500, out_features=500)),
            ('activation2', nn.Sigmoid()),
            ('batch2', nn.BatchNorm1d(num_features=500)),
            ('drop2', nn.Dropout(p=dropout)),
            ('linear3', nn.Linear(in_features=500, out_features=num_classes))
            
        ]))
        
        self.modelo=nn.Sequential(OrderedDict([
            ('bloque1', self.initbloque),
            ('bloque2', self.bloque1),
            ('maxpool1', nn.MaxPool2d(kernel_size=2)),
            ('flat', nn.Flatten()),
            ('drop1', nn.Dropout(p=dropout)),
            ('fc', self.fc)           
            
        ]))
        
    def forward(self,x):
        return self.modelo(x)
    
class AgoModel11(AgoModel):
    def __init__(self,num_classes,lr=0.001,optim='SGD', ridge=0, dropout=0.3):
        super().__init__(num_classes,lr,optim,ridge, dropout)
        self.modelo=Model11(num_classes, dropout)
        
###################################################################################

###################################################################################

class Model2(nn.Module):
    def __init__(self, num_classes, dropout):
        self.initbloque=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding='same')),
            ('activation1', nn.Sigmoid()),
            
        ]))
        
        self.bloque1=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')),
            ('activation1', nn.Sigmoid()),
            ('maxpool1', nn.MaxPool2d(kernel_size=2)),
            ('batch1', nn.BatchNorm2d(num_features=32)),
        ]))
        
        self.bloque2=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same')),
            ('activation1', nn.Sigmoid()),
            
        ]))
        
        self.bloque3=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same')),
            ('activation1', nn.Sigmoid()),
            ('maxpool1', nn.MaxPool2d(kernel_size=2)),
            ('batch1', nn.BatchNorm2d(num_features=64)),
        ]))
        
        self.fc=nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(in_features=100*100*64, out_features=500)),
            ('activation1', nn.Sigmoid()),
            ('batch1', nn.BatchNorm1d(num_features=500)),
            ('drop1', nn.Dropout(p=dropout)),
            ('linear2', nn.Linear(in_features=500, out_features=500)),
            ('activation2', nn.Sigmoid()),
            ('batch2', nn.BatchNorm1d(num_features=500)),
            ('drop2', nn.Dropout(p=dropout)),
            ('linear3', nn.Linear(in_features=500, out_features=num_classes))
            
        ]))
        
        self.model=nn.Sequential(OrderedDict([
            ('bloqueInit', self.initbloque),
            ('bloque1', self.bloque1),
            ('bloque2', self.bloque2),
            ('bloque3', self.bloque3),
            ('flat', nn.Flatten()),
            ('drop1', nn.Dropout(p=dropout)),
            ('fc', self.fc)           
            
        ]))
        
    def forward(self,x):
        return self.modelo(x)
    
class AgoModel2(AgoModel):
    def __init__(self,num_classes,lr=0.001,optim='SGD', ridge=0, dropout=0.3):
        super().__init__(num_classes,lr,optim,ridge, dropout)
        self.modelo=Model2(num_classes, dropout)

class AgoCoatnet(AgoModel):
    def __init__(self, num_classes, lr=0.001, optim='SGD', ridge=0, 
        num_blocks = [1, 1, 1, 4, 5] ,
        channels = [6, 6, 7,7,7], # Canales de salida para cada etapa
        block_types = ['C', 'C', 'T', 'T'] # Layout C-C-T-T optimizado
    ):
        super().__init__(num_classes,lr,optim,ridge)
        self.modelo = CoAtNet((224, 224), 3, num_blocks, channels, block_types=block_types)
        self.modelo.fc=nn.Linear(self.modelo.fc.in_features, num_classes)