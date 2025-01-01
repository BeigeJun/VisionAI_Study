import zipfile

def extract_imagenet(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)


zip_path = "D:/ImageNet/ImageNet.zip"
extract_path = "D:/ImageNet/ImageNet"

extract_imagenet(zip_path, extract_path)