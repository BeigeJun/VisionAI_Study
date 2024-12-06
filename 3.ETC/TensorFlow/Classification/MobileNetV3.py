import tensorflow as tf
from tensorflow.keras import layers, Model, datasets, utils
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy


def _ensure_divisible(number, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_num = max(min_value, int(number + divisor / 2) // divisor * divisor)
    if new_num < 0.9 * number:
        new_num += divisor
    return new_num


class H_sigmoid(layers.Layer):
    def call(self, inputs):
        return tf.nn.relu6(inputs + 3) / 6

class H_swish(layers.Layer):
    def call(self, inputs):
        return inputs * tf.nn.relu6(inputs + 3) / 6


class SEModule(layers.Layer):
    def __init__(self, in_channels_num, reduction_ratio=4):
        super(SEModule, self).__init__()
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.fc = tf.keras.Sequential([
            layers.Dense(in_channels_num // reduction_ratio, use_bias=False),
            layers.ReLU(),
            layers.Dense(in_channels_num, use_bias=False),
            H_sigmoid()
        ])

    def call(self, inputs):
        x = self.avg_pool(inputs)
        x = self.fc(x)
        x = tf.reshape(x, [-1, 1, 1, x.shape[-1]])
        return inputs * x


class Bottleneck(layers.Layer):
    def __init__(self, in_channels_num, exp_size, out_channels_num, kernel_size, stride, use_SE, NL, BN_momentum):
        super(Bottleneck, self).__init__()
        self.use_residual = (stride == 1 and in_channels_num == out_channels_num)
        NL = NL.upper()
        use_HS = NL == 'HS'

        if exp_size == in_channels_num:
            self.conv = tf.keras.Sequential([
                layers.DepthwiseConv2D(kernel_size, strides=stride, padding='same', use_bias=False),
                layers.BatchNormalization(momentum=BN_momentum),
                SEModule(exp_size) if use_SE else layers.Layer(),
                H_swish() if use_HS else layers.ReLU(),
                layers.Conv2D(out_channels_num, kernel_size=1, strides=1, padding='same', use_bias=False),
                layers.BatchNormalization(momentum=BN_momentum)
            ])
        else:
            self.conv = tf.keras.Sequential([
                layers.Conv2D(exp_size, kernel_size=1, strides=1, padding='same', use_bias=False),
                layers.BatchNormalization(momentum=BN_momentum),
                H_swish() if use_HS else layers.ReLU(),
                layers.DepthwiseConv2D(kernel_size, strides=stride, padding='same', use_bias=False),
                layers.BatchNormalization(momentum=BN_momentum),
                SEModule(exp_size) if use_SE else layers.Layer(),
                H_swish() if use_HS else layers.ReLU(),
                layers.Conv2D(out_channels_num, kernel_size=1, strides=1, padding='same', use_bias=False),
                layers.BatchNormalization(momentum=BN_momentum)
            ])

    def call(self, inputs):
        x = self.conv(inputs)
        if self.use_residual:
            return x + inputs
        else:
            return x


class MobileNetV3(Model):
    def __init__(self, mode='small', classes_num=1000, input_size=224, width_multiplier=1.0, dropout=0.2, BN_momentum=0.1):
        super(MobileNetV3, self).__init__()
        mode = mode.lower()
        s = 2 if input_size > 32 else 1

        if mode == 'large':
            configs = [
                [3, 16, 16, False, 'RE', 1],
                [3, 64, 24, False, 'RE', s],
                [3, 72, 24, False, 'RE', 1],
                [5, 72, 40, True, 'RE', 2],
                [5, 120, 40, True, 'RE', 1],
                [5, 120, 40, True, 'RE', 1],
                [3, 240, 80, False, 'HS', 2],
                [3, 200, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 480, 112, True, 'HS', 1],
                [3, 672, 112, True, 'HS', 1],
                [5, 672, 160, True, 'HS', 2],
                [5, 960, 160, True, 'HS', 1],
                [5, 960, 160, True, 'HS', 1]
            ]
        else:
            configs = [
                [3, 16, 16, True, 'RE', s],
                [3, 72, 24, False, 'RE', 2],
                [3, 88, 24, False, 'RE', 1],
                [5, 96, 40, True, 'HS', 2],
                [5, 240, 40, True, 'HS', 1],
                [5, 240, 40, True, 'HS', 1],
                [5, 120, 48, True, 'HS', 1],
                [5, 144, 48, True, 'HS', 1],
                [5, 288, 96, True, 'HS', 2],
                [5, 576, 96, True, 'HS', 1],
                [5, 576, 96, True, 'HS', 1]
            ]

        first_channels_num = 16
        last_channels_num = 1280 if mode == 'large' else 1024
        divisor = 8

        input_channels_num = _ensure_divisible(first_channels_num * width_multiplier, divisor)
        last_channels_num = _ensure_divisible(last_channels_num * width_multiplier, divisor) if width_multiplier > 1 else last_channels_num

        self.features = [tf.keras.Sequential([
            layers.Conv2D(input_channels_num, kernel_size=3, strides=s, padding='same', use_bias=False),
            layers.BatchNormalization(momentum=BN_momentum),
            H_swish()
        ])]

        for kernel_size, exp_size, out_channels_num, use_SE, NL, stride in configs:
            output_channels_num = _ensure_divisible(out_channels_num * width_multiplier, divisor)
            exp_size = _ensure_divisible(exp_size * width_multiplier, divisor)
            self.features.append(Bottleneck(input_channels_num, exp_size, output_channels_num, kernel_size, stride, use_SE, NL, BN_momentum))
            input_channels_num = output_channels_num

        last_stage_channels_num = _ensure_divisible(exp_size * width_multiplier, divisor)
        self.features.append(tf.keras.Sequential([
            layers.Conv2D(last_stage_channels_num, kernel_size=1, strides=1, padding='same', use_bias=False),
            layers.BatchNormalization(momentum=BN_momentum),
            H_swish()
        ]))

        self.features = tf.keras.Sequential(self.features)

        self.last_stage = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Flatten()
        ])

        self.classifier = tf.keras.Sequential([
            layers.Dropout(dropout),
            layers.Dense(last_channels_num, activation='relu'),
            layers.Dense(classes_num)
        ])

    def call(self, inputs):
        x = self.features(inputs)
        x = self.last_stage(x)
        x = self.classifier(x)
        return x

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, depth=10)
    return image, label

train_dataset = tf.keras.utils.image_dataset_from_directory(
    'C:/Users/Administrator/Desktop/Work/6.Test_File/6PiSaveFile',  
    image_size=(224, 224),  
    batch_size=64,          
    label_mode='int'        
).map(preprocess)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    'C:/Users/Administrator/Desktop/Work/6.Test_File/Test',  
    image_size=(224, 224),  
    batch_size=64,          
    label_mode='int'       
).map(preprocess)

model = MobileNetV3(mode='large', classes_num=10, input_size=224, width_multiplier=1.0)

model.compile(optimizer=Adam(learning_rate=0.001),
              loss=CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_dataset, 
          epochs=50, 
          validation_data=test_dataset, 
          verbose=2)

test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
print(f'Test accuracy: {test_acc:.4f}')
import numpy as np
import matplotlib.pyplot as plt
class_names = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 
               'Class 6', 'Class 7', 'Class 8', 'Class 9', 'Class 10']

def show_predictions(dataset, model, num_images=5):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        preds = model.predict(images)
        preds_labels = np.argmax(preds, axis=-1)
        true_labels = np.argmax(labels, axis=-1)
        
        for i in range(num_images):
            plt.subplot(1, num_images, i + 1)
            plt.imshow(images[i].numpy())
            plt.title(f"Pred: {class_names[preds_labels[i]]}\nTrue: {class_names[true_labels[i]]}")
            plt.axis("off")
    plt.show()

show_predictions(test_dataset, model, num_images=5)
