import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import matplotlib.pyplot as plt

model = ResNet50(weights='imagenet')

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def generate_cam(feature_conv, weight_softmax, class_idx):
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    return cv2.resize(cam_img, size_upsample)

def visualize_cam(image_path):
    image_tensor = preprocess_image(image_path)
    
    last_conv_layer = model.get_layer('conv5_block3_out')
    
    grad_model = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
    
    with tf.GradientTape() as tape:
        preds, conv_outputs = grad_model(image_tensor)
        pred_index = tf.argmax(preds[0])
        class_output = preds[:, pred_index]
    
    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape
    
    heatmap = cv2.resize(heatmap, (width, height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = heatmap * 0.4 + img
    superimposed_img = superimposed_img / np.max(superimposed_img)
    superimposed_img = np.uint8(255 * superimposed_img)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(heatmap)
    plt.title('Heatmap')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(superimposed_img)
    plt.title('CAM Result')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    preds = preds.numpy()
    print('Predicted:', decode_predictions(preds, top=1)[0])

if __name__ == "__main__":
    image_path = ''
    visualize_cam(image_path)
