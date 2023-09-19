import tensorflow as tf
from datetime import datetime


from clearml import Task
task = Task.init(project_name='HyperparameterOptimizationTest', 
                 task_name='base_task')

parameters = {
    'hidden_dim': 25,
    'optimizer': "adam",
    'dropout': 0.1
}
task.connect(parameters)


fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(parameters['hidden_dim'], activation='relu'),
    tf.keras.layers.Dropout(parameters['dropout']),
    tf.keras.layers.Dense(10, activation='softmax')
])

#loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=parameters['optimizer'],
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

logdir = "logs/image/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')

#cm_callback = tf.keras.callbacks.LambdaCallback()

model.fit(train_images, 
          train_labels, 
          epochs=10,
          verbose=0,
          callbacks=[tensorboard_callback],
          validation_data=(test_images, test_labels),
)