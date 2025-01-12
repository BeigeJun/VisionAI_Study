import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

def load_and_preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return img_array

def build_model():
    base_model = tf.keras.applications.ResNet50(
        input_shape=(224, 224, 3),
        include_top=True,
        weights='imagenet',
        pooling='avg'
    )

    x = base_model.layers[-2].output
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    
    model = tf.keras.Model(base_model.input, outputs)

    for layer in base_model.layers[:-10]:
        layer.trainable = False
    
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['acc']
    )
    
    return model

image_path = ''

image = load_and_preprocess_image(image_path)

model = build_model()

cam_model = tf.keras.Model(
    model.input, 
    outputs=(model.get_layer('conv5_block3_out').output, model.output)
)
gap_weights = model.layers[-1].get_weights()[0]

def show_cam(image_value, features, results, label):
    features_for_img = features[0]
    prediction = results[0]
    
    class_activation_weights = gap_weights[:, label]
    class_activation_features = sp.ndimage.zoom(features_for_img, (224/7, 224/7, 1), order=2)
    cam_output = np.dot(class_activation_features, class_activation_weights)
    cam_output = tf.reshape(cam_output, (224, 224))
    
    print(f'Softmax output: {results}')
    print(f"Prediction: {'dog' if tf.argmax(results[0]) == 1 else 'cat'}")
    plt.figure(figsize=(8, 8))
    plt.imshow(cam_output, cmap='jet', alpha=0.5)
    plt.imshow(tf.squeeze(image_value), alpha=0.5)
    plt.axis('off')
    plt.title('Class Activation Map')
    plt.show()


features, results = cam_model.predict(image)

predicted_class = tf.argmax(results[0]).numpy()

show_cam(image, features, results, predicted_class)
