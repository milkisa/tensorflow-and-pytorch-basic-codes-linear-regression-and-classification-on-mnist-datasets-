
##6s 3ms/step - loss: 0.1396 - accuracy: 0.9476 - val_loss: 0.2621 - val_accuracy: 0.9142
import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('val_accuracy')>0.93):
      print("\nReached 98% accuracy so cancelling training!")
      self.model.stop_training = True
callbacks = myCallback()

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(128, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(128, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Dropout(0.2),
  # tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  # tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
history= model.fit(training_images, training_labels,validation_data=(test_images, test_labels) ,epochs=10,  callbacks=[callbacks])
test_loss = model.evaluate(test_images, test_labels)
import matplotlib.pyplot as plt
accuracy= history.history['accuracy']
valid_accuracy= history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs= range(len(accuracy))
plt.plot(epochs,accuracy, 'red', label= 'accuracy')
plt.plot(epochs,valid_accuracy,'g',label= 'valid_accuracy')
plt.title('training and validation accuracy')
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()