import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from D_AnomalyDetection.Util.Loss import SSIM

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

OriginalImage_path = 'D:/1-2. MvTec_Anomaly/1.Bottel/train/good/190.png'
GenImage_path = 'D:/0. Model_Save_Folder/output_images/epoch_99_0.png'

OriginalImage = Image.open(OriginalImage_path)
GenImage = Image.open(GenImage_path)

TensorOriginalImage = transform(OriginalImage).unsqueeze(0)
TensorGenImage = transform(GenImage).unsqueeze(0)

SSIM_func = SSIM()

tensorLoss = SSIM_func(TensorOriginalImage, TensorGenImage)

npLoss = np.array(tensorLoss)

print(npLoss)