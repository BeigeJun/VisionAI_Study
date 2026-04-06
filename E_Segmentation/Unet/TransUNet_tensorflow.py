import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# LABEL_MAP = {'good': 0, 'broken': 1, 'contamination': 2}
# CLASS_NAMES = ['Good', 'Broken', 'Contamination']
# CLASS_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

LABEL_MAP = { 'good': 0,'crack': 1, 'cut': 2, 'hole': 3, 'print': 4}
CLASS_NAMES = ['Good','Crack', 'Cut', 'Hole', 'Print']
CLASS_COLORS = [(0, 0, 0), (0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]

def transformer_block(x, dim, heads, mlp_dim):
    res = x
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.MultiHeadAttention(num_heads=heads, key_dim=dim // heads)(x, x)
    x = layers.Add()([res, x])
    res = x
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(mlp_dim, activation='gelu')(x)
    x = layers.Dense(dim)(x)
    x = layers.Add()([res, x])
    return x

def decoder_block(x, skip, out_c):
    x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = layers.Concatenate()([x, skip])
    for _ in range(2):
        x = layers.Conv2D(out_c, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
    return x

def build_transunet(input_shape=(512, 512, 3), num_classes=3):
    inputs = layers.Input(shape=input_shape)

    resnet = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)
    
    s1 = resnet.get_layer("conv1_relu").output            # 256x256, 64ch
    s2 = resnet.get_layer("conv2_block3_out").output      # 128x128, 256ch
    s3 = resnet.get_layer("conv3_block4_out").output      # 64x64, 512ch
    s4 = resnet.get_layer("conv4_block6_out").output      # 32x32, 1024ch

    b = layers.Conv2D(512, 1, padding='same')(s4)
    b = layers.Reshape((1024, 512))(b)
    b = layers.Conv1D(512, 1, padding='same', activation='relu')(b)

    for _ in range(4):
        b = transformer_block(b, 512, 8, 1024)

    b = layers.Reshape((32, 32, 512))(b)

    # --- Decoder ---
    d1 = decoder_block(b, s3, 256)   # 32x32 -> 64x64
    d2 = decoder_block(d1, s2, 128)  # 64x64 -> 128x128
    d3 = decoder_block(d2, s1, 64)   # 128x128 -> 256x256

    # --- Final Output ---
    x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d3) # 512x512
    
    if num_classes == 1:
        outputs = layers.Conv2D(num_classes, 1, activation='sigmoid')(x)
    else:
        outputs = layers.Conv2D(num_classes, 1, activation='softmax')(x)

    return models.Model(inputs, outputs, name="TransUNet_Final")

def load_and_preprocess(imgPath, maskPath, inputSize, labelIdx):
    inputSize = int(inputSize)
    lIdx = int(labelIdx)
    
    if hasattr(imgPath, 'numpy'): imgPath = imgPath.numpy().decode('utf-8')
    if hasattr(maskPath, 'numpy'): maskPath = maskPath.numpy().decode('utf-8')

    try:
        imgRaw = tf.io.read_file(imgPath)
        img = tf.image.decode_png(imgRaw, channels=3)
        img = tf.image.resize(img, [inputSize, inputSize])
        img = tf.cast(img, tf.float32) / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    except:
        img = tf.zeros([inputSize, inputSize, 3], dtype=tf.float32)

    mask = np.zeros((inputSize, inputSize, 1), dtype=np.int32)

    if lIdx > 0 and maskPath and os.path.exists(maskPath):
        try:
            m = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                m = cv2.resize(m, (inputSize, inputSize), interpolation=cv2.INTER_NEAREST)
                mask[m > 128] = lIdx
        except:
            pass
    
    mask_tensor = tf.convert_to_tensor(mask, dtype=tf.int32)
    return img, mask_tensor

def get_dataset(root_path, split='train', input_size=512, batch_size=8):
    root = Path(root_path)
    image_paths = sorted(glob.glob(str(root / split / '**' / '*.png'), recursive=True))
    img_list, mask_list, label_indices = [], [], []
    
    for p in image_paths:
        p = Path(p)
        anomaly_type = p.parent.name
        l_idx = LABEL_MAP.get(anomaly_type, 0)
        img_list.append(str(p))
        label_indices.append(l_idx)
        m_path = root / 'ground_truth' / anomaly_type / f'{p.stem}_mask.png'
        mask_list.append(str(m_path) if m_path.exists() else "")

    dataset = tf.data.Dataset.from_tensor_slices((img_list, mask_list, label_indices))
    dataset = dataset.map(lambda i, m, l: tf.py_function(load_and_preprocess, [i, m, input_size, l], [tf.float32, tf.int32]), 
                          num_parallel_calls=tf.data.AUTOTUNE)
    
    def set_shapes(img, mask):
        img.set_shape([input_size, input_size, 3])
        mask.set_shape([input_size, input_size, 1])
        return img, mask

    dataset = dataset.map(set_shapes)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def draw_results(img_np, pred_mask):
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    overlay = img_cv.copy()
    for cls_id in range(1, len(CLASS_NAMES)):
        cls_mask = (pred_mask == cls_id).astype(np.uint8)
        color = CLASS_COLORS[cls_id]
        overlay[cls_mask > 0] = color
        contours, _ = cv2.findContours(cls_mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 10: continue
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_cv, (x, y), (x + w, y + h), color, 1)
            cv2.putText(img_cv, CLASS_NAMES[cls_id], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return cv2.addWeighted(overlay, 0.4, img_cv, 0.6, 0)

def visualize_all_test_set_tf(model, root_path, input_size=512):
    test_ds = get_dataset(root_path, split='test', input_size=input_size, batch_size=1)
    
    plt.ion() 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for idx, (img_batch, mask_batch) in enumerate(test_ds):
        preds = model.predict(img_batch, verbose=0)
        pred_mask = np.argmax(preds[0], axis=-1).astype(np.uint8)
        
        img_np = img_batch[0].numpy()
        img_vis = ((img_np * std + mean).clip(0, 1) * 255).astype(np.uint8)
        
        res_img = draw_results(img_vis, pred_mask)

        ax1.clear()
        ax1.imshow(img_vis)
        ax1.set_title(f"Test Image [{idx+1}] - Original")
        ax1.axis('off')

        ax2.clear()
        ax2.imshow(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB))
        ax2.set_title(f"Test Image [{idx+1}] - Prediction")
        ax2.axis('off')
        
        plt.draw()
        plt.pause(3)
        if not plt.fignum_exists(fig.number):
            break

    plt.ioff()
    plt.show()

def main():
    root_path = "D:/1. DataSet/hazelnut"
    input_size = 512
    batch_size = 4
    
    model = build_transunet(input_shape=(input_size, input_size, 3), num_classes=len(CLASS_NAMES))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    train_ds = get_dataset(root_path, split='train', input_size=input_size, batch_size=batch_size)
    
    model.fit(train_ds, epochs=100)
    model.save_weights("transunet_weights.weights.h5")
    visualize_all_test_set_tf(model, root_path, input_size=input_size)

if __name__ == "__main__":
    main()