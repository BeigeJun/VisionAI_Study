import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from D_AnomalyDetection.Util.Loss import SSIM

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

OriginalImage_path = 'D:/0. Model_Save_Folder/output_test_images/80_input_FALSE_1.png'
GenImage_path = 'D:/0. Model_Save_Folder/output_test_images/80_output_FALSE_1.png'

OriginalImage = Image.open(OriginalImage_path)
GenImage = Image.open(GenImage_path)

TensorOriginalImage = transform(OriginalImage).unsqueeze(0)
TensorGenImage = transform(GenImage).unsqueeze(0)

SSIM_func = SSIM(False)
tensorLoss = SSIM_func(TensorOriginalImage, TensorGenImage)
npLoss = np.array(tensorLoss)

print(npLoss)

#------------------------------시각화--------------------------------
SSIM_func = SSIM(True)
tensorLoss = SSIM_func(TensorOriginalImage, TensorGenImage)
npLoss = np.array(tensorLoss)

#print(npLoss.shape)

npLoss = npLoss.squeeze(0)
npLoss = npLoss.mean(axis=0)
npLoss = np.clip(npLoss * 255.0, 0, 255)
# binaryLossMap = np.where(npLoss > 100 , 0, 255)
binaryLossMap = npLoss.astype(np.uint8)

img = Image.fromarray(binaryLossMap)
img.show()