
import tensorflow as tf

# Check for TensorFlow GPU access
devices_str = "\n".join([str(num+1) + "- " + str(dev) for num, dev in enumerate(tf.config.list_physical_devices())])
print(f"TensorFlow has access to the following devices:\n{devices_str}")

# See TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

cifar = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar.load_data()
model = tf.keras.applications.ResNet50(
    include_top=True,
    weights=None,
    input_shape=(32, 32, 3),
    classes=100,)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5, batch_size=64)