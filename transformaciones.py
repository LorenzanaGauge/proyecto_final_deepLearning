

# Bibliotecas
from torchvision import transforms
from imgaug import augmenters as iaa
import imgaug as ia
import numpy as np

# Aumentado de datos con imgaug
class ImgAugTransform:
    '''
    Args:
        dropout (bool): Indica si se aplicara la oprecion de 
            CoarseDropout.
    Returns:
        Devuelve el conjunto de operaciones de la biblio
        imgaug que seran aplicadas a las imagenes en cjto
        con otras operaciones de torch.
    Operaciones aplicadas:
        PadToAspectRatio: Aplica paddin a la imagen de 
            tal forma que se tenga una imagen cuadrada 
            pero se respeten las dim  de la imagen original.
        CoarseDropout: Crea rectangulos vacios dentro de
            la imagen.
    '''
    
    def __init__(self, dropout):
    
        if(dropout):
            self.aug = iaa.Sequential([
                            #iaa.size.PadToAspectRatio(1, position = 'center'), 
                            iaa.Sometimes(0.5,iaa.GaussianBlur(sigma=(0, 0.5))),
                            iaa.Sometimes(0.5,iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)),
                            iaa.CoarseDropout(0.1, size_percent=1.0), #0.5
                            iaa.Resize(224)])  
        else:
            self.aug = iaa.Sequential([
                            iaa.size.PadToAspectRatio(1, position = 'center'),
                            iaa.Resize(224),])

    def __call__(self, img):
        
        # Las op de imgaug necesitan que pasemos PIL --> np.array
        img = np.array(img)
        return self.aug.augment_image(img)


# Los siguientes parametros son los que especifican en la doc de pytorch
# para normalizar las imagenes que entranr a los modelos preentrenados

torch_mean = [0.485, 0.456, 0.406]
torch_std = [0.229, 0.224, 0.225]

# Transformacion para entrenamiento, usa CoarseDropout
transforms_entr = transforms.Compose([
                            ImgAugTransform(dropout = True),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = torch_mean, std = torch_std)
                            ])

# Transformacion para validacion, no usa CoarseDropout
transforms_val = transforms.Compose([
                            ImgAugTransform(dropout = False),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = torch_mean, std = torch_std)
                            ])

# Transformacion empleada sobre las imagenes que se visualizaran en inspecciones de rendimiento
transforms_visual = transforms.Compose([
                                ImgAugTransform(dropout = False),
                                transforms.ToTensor()
                                ])
