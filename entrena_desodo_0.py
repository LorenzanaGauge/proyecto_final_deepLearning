import torch
from torch import nn
from torch.functional import norm
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision import transforms
from torchvision import models
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl
from torchmetrics import Accuracy, MetricCollection, Precision, Recall, ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.profilers import Profiler, SimpleProfiler, AdvancedProfiler
import os.path as path
from Modelos import *
#from Modelos_Emilio import *
from transformaciones import transforms_entr,transforms_val
import pickle

from util.cutout import Cutout


from collections import OrderedDict, namedtuple
from itertools import product

torch.set_float32_matmul_precision('medium')
########################################################################################################################
class RunBuilder():
    @staticmethod
    def get_runs(params):
        Run=namedtuple('Run', params.keys())

        runs=[]

        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs
##########################################################################################################################
class FilteredImageFolder(ImageFolder):
    def find_classes(self,directory:str):
        classes = []
        with open(path.join(directory,"coincidentes.txt") ) as f:
            for linea in f:
                classes.append(linea.strip())
        classes = sorted(classes)
        class_to_idx = {upc:i for i,upc in enumerate(classes)}

        return classes,class_to_idx
        
def obtener_upcs_coincidentes(dataset1,dataset2):
    # obtenemos los upcs del primer dataset y los convierto en un conjunto
    upc_indice1 = set(dataset1.class_to_idx.keys())
    # analogamente con el segundo dataset
    upc_indice2 = set(dataset2.class_to_idx.keys())
    
    coincidentes = list(upc_indice1 & upc_indice2)

    ruta1 = path.join (dataset1.root,"coincidentes.txt")
    ruta2 = path.join (dataset2.root,"coincidentes.txt")
    with open(ruta1,'w') as archivo1:
        for upc in coincidentes:
            archivo1.write(upc+'\n')
    with open(ruta2,'w') as archivo2:
        for upc in coincidentes:
            archivo2.write(upc+'\n')
    
    faltantes_dataset1 = upc_indice2-upc_indice1
    faltantes_dataset2 = upc_indice1-upc_indice2

    # devolvemos los upcs que se encuentran en ambos
    return coincidentes,faltantes_dataset1,faltantes_dataset2

def guardar_reporte_upcs(faltantes_entrenamiento,faltantes_validacion,upcs_coincidentes):
    
    with open('faltantes_train.txt','w') as faltantes_train:
        for upc in faltantes_entrenamiento:
            faltantes_train.write(f"{upc}\n")

    with open('faltantes_val.txt','w') as faltantes_val:
        for upc in faltantes_validacion:
            faltantes_val.write(f"{upc}\n")

    with open('coincidentes.txt','w') as coincidentes:
        for upc in upcs_coincidentes:
            coincidentes.write(f"{upc}\n")

def guardar_traduccion_clases_ids(ruta,diccionario_traducciones):
    '''Guarda en ruta una traduccion del diccionario_traducciones, en este archivo nuevo
    se usa como llave el indice de la red neuronal y el contenido sera el UPC original'''
    #El diccionario original tiene llave UPC y contenido indice,
    #por lo que creamos un nuevo diccionario que tenga como llave el
    #indice y devuelva el UPC del producto
    id_to_UPC = {v: k for k, v in diccionario_traducciones.items()}

    with open(path.join(ruta,'traduccion_clases.pkl'), 'wb') as output:
        pickle.dump(id_to_UPC, output, protocol=pickle.HIGHEST_PROTOCOL)


####################################################################################################


"""
Dado que el código se actualiza constantemente, es necesario establecer una sección
en donde se puedan mover a mano las variables y/o hiperparámetros necesarios, de tal modo que 
se pueda ejecutar con facilidad el código, sin la neesidad
"""

catName='DESODO'

params=OrderedDict(
        model=['Coatnet'], #modelos a probar, integrados en el script Modelos.py
        optim=['Adam'], #En la clase AgoModel se configuraron los optimizadores, de tal forma
                       #que el usuario solo ponga una cadena como entrada, y el script
                       #identifica en PyTorch nativo el optimizador a trabajar
        ridge=[0.01,0.001,0.0001], #Es el término denominado "alfa" en la regularización Rdige
                                   #que escala la penalización
        lr=[0.001], #es el learning rate con el que se actualizarán los parámetros del modelo
        #drop=[0.1,0.2,0.3,0.4,0.5],
        amp=['O2'],# Es el nivel de Automatic Mixed Precision con el que se entrenará 
        epochs=[10], #El número de épocas 
        batch=[256], #El tamaño del lote, que se especifica en el DataLoader
        holes=[1], #Aplicando la técnica de CutOut para evitar sobreajuste,
                   #este hiperparámetro nos define cuántos hoyos se van a aplicar a la imagen
        length=[40], #es la dimensión con la que se aplicará el hoyo de CutOut
        patience=[5], #es el número de épocas en EarlyStopping que espera hasta que se logre
                       #el cambio especificado
        gpu=[0] #es la GPU con la que se quiere trabajar
)


models={'000':AgoModel000, '0001':AgoModel0001, '0002':AgoModel0002, '0003':AgoModel0003,
        '0004':AgoModel0004, '0005':AgoModel0005, '0006':AgoModel0006,
        'Resnet101':AgoResnet101,'Resnet152':AgoResnet152,'Resnet50':AgoResnet50, 'Resnet18':AgoResnet18,
        'Resnext50':AgoResnext50, 'Resnext101':AgoResnext101, 
        'Resnet18':AgoResnet18, 'Coatnet':AgoCoatnet}

######################################################################################################

if __name__=='__main__':
    for run in RunBuilder().get_runs(params):
        categoria=f"./DS_{catName}/{catName}" #línea  cambiar para datos actuales
        # data
        random_transformation1=[transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                transforms.RandomAdjustSharpness(sharpness_factor=2),
                ]
                
        random_transformation2=[transforms.RandomAutocontrast(),
                transforms.RandomHorizontalFlip(p=0.5)]
            
        transformations=transforms.Compose([
                transforms.RandomApply(transforms=random_transformation1),
                transforms.RandomApply(transforms=random_transformation2),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # Cutout(n_holes=1, length=10)
                ])
        
        train_dataset = ImageFolder(f'{categoria}_train',transform = transformations)
        val_dataset = ImageFolder(f'{categoria}_val',transform = transforms_val)

        if train_dataset.class_to_idx==val_dataset.class_to_idx:
            pass
        else:
            upcs_coincidentes = obtener_upcs_coincidentes(train_dataset,val_dataset)
            #guardar_reporte_upcs(train_dataset,val_dataset,upcs_coincidentes)
            print("Se entrenaran solo los UPCS coincidentes")
            train_dataset = FilteredImageFolder(train_dataset.root,transform=transforms_val)
            val_dataset = FilteredImageFolder(val_dataset.root,transform=transformations)
            assert train_dataset.class_to_idx==val_dataset.class_to_idx

        

        train_loader = DataLoader(train_dataset, batch_size=run.batch,num_workers=2,shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=run.batch,num_workers=2,shuffle=False)
        # model
        num_clases = len(train_dataset.class_to_idx)

        import os
        if not os.path.exists(f'{categoria}_traduccion'):
            os.mkdir(f'{categoria}_traduccion')
        guardar_traduccion_clases_ids(f"{categoria}_traduccion",train_dataset.class_to_idx)




        early_stop_callback = EarlyStopping(
                min_delta=0.1,
                monitor='val_acc',
                patience=run.patience,
                verbose=False,
                mode='max',
                strict=True
                )
        model_checkpoint = ModelCheckpoint(monitor='val_acc',mode='max',save_last=True,save_top_k=3,filename=f'{catName}'+'-{epoch:02d}-{val_acc:.2f}-'+str(num_clases))
        print(f"Total de productos a clasificar: {num_clases}")
        model = models[run.model](num_clases,optim=run.optim,lr=run.lr, ridge=run.ridge)
        print(f"El modelo entrenandose es: {(run.model,run.optim,run.lr)}")
        name= str(run.model)+"_"+str(run.optim)+"_AMP-level-"+str(run.amp)+"_Batch"+str(run.batch)+"_lr"+str(run.lr)+'RidgePenalty_'+str(run.ridge)+'_schedulerPlateu_gamma-0.7_EarlyStopp-val_acc_'+'Cutout:holes-'+str(run.holes)+'_lenght-'+str(run.length)#+'_Dropout-'+str(run.drop)
        logger = TensorBoardLogger(f"./resultados/resultadosTB/tb_{catName}", name=name, default_hp_metric=False)
        profiler=AdvancedProfiler(dirpath=f'./resultados/reportesProfiler/{catName}', filename=name)
        # training
        trainer = pl.Trainer(
                            accelerator='gpu',
                            devices=[run.gpu],
                            logger=logger, 
                            callbacks=[model_checkpoint, early_stop_callback],
                            precision="16-mixed" if run.amp in ['O1','O2'] else 32,
                            #limit_train_batches=0.8,
                            max_epochs=run.epochs,
                            profiler=profiler                    
                            )
        trainer.fit(model, train_loader, val_loader)
    # print("Probando")
    # model = BenjaModel.load_from_checkpoint("/home/ago_ai/entrenador-redes/tb_logs/SHAMP/version_2/checkpoints/last.ckpt",num_classes=535)
    # trainer.validate(model,val_loader)
