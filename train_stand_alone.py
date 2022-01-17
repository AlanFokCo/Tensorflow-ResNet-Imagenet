import tensorflow as tf
import numpy as np
from load_data import load_data_with_split_only_index, decode_img
from sklearn import metrics
import argparse
import os

parser = argparse.ArgumentParser(description='Tensorflow2.0 ResNet Classification ImageNet')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--learning-rate', type=float, default=0.001)
parser.add_argument('--train-dir', default=os.path.expanduser('~/imagenet/train'))
parser.add_argument('--test-dir', default=os.path.expanduser('~/imagenet/validation'))

args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.learning_rate

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_visible_devices(gpu, 'GPU')

train_dataset = args.train_dir
test_dataset = args.test_dir

model = tf.keras.Sequential([
    tf.keras.applications.resnet_v2.ResNet50V2(weights=None, input_shape=(224, 224, 3), include_top=False),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1000, activation='softmax')
])

optimizer = tf.optimizers.SGD(learning_rate=learning_rate)
loss = tf.losses.SparseCategoricalCrossentropy()

train_images, train_labels = load_data_with_split_only_index(para=1, index=0, dataset=train_dataset)
test_images, test_labels = load_data_with_split_only_index(para=1, index=0, dataset=test_dataset)

train_loss = tf.keras.metrics.Mean(name='train_loss')

train_length = len(train_labels)
test_length = len(test_labels)


def evaluate_fn(test_model, test_batch_size):
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = 0
    test_count = 0

    for test_idx in range(0, test_length, test_batch_size):
        test_count += 1
        test_data = []
        for test_image in test_images[test_idx:test_idx + test_batch_size]:
            test_data.append(decode_img(test_image, [224, 224]))

        test_data = np.array(test_data)
        test_result = test_model(test_data)
        test_predictions = []
        for pre_item in test_result:
            tmp_arr = list(pre_item)
            test_predictions.append(tmp_arr.index(max(tmp_arr)))

        test_predictions = np.array(test_predictions)
        test_label = tf.cast(test_labels[test_idx:test_idx + test_batch_size], dtype=tf.float32)
        test_loss_value = loss(test_label, test_result)

        test_loss(test_loss_value)
        test_accuracy += metrics.accuracy_score(tf.cast(test_label, dtype=tf.int32), test_predictions)

    return (test_accuracy / test_count) * 100, test_loss.result()


for epoch in range(0, epochs):
    train_accuracy = 0
    train_count = 0
    for idx in range(0, train_length, batch_size):
        train_count += 1
        data = []
        for image in train_images[idx:idx + batch_size]:
            data.append(decode_img(image, [224, 224]))

        data = np.array(data)
        with tf.GradientTape() as tape:
            result = model(data)
            predictions = []
            for pre in result:
                tmp = list(pre)
                predictions.append(tmp.index(max(tmp)))

            predictions = np.array(predictions)
            label = tf.cast(train_labels[idx:idx + batch_size], dtype=tf.float32)
            loss_value = loss(label, result)

        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_loss(loss_value)
        train_accuracy += metrics.accuracy_score(tf.cast(label, dtype=tf.int32), predictions)

    test_accuracy, test_loss = evaluate_fn(model, 32)

    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {(train_accuracy / train_count) * 100}, '
        f'Test Loss: {test_loss}, '
        f'Test Accuracy: {test_accuracy}, '
    )
