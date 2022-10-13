import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Union


builder = tfds.builder('mnist')
BATCH_SIZE = 256
NUM_EPOCHS = 12
NUM_CLASSES = 10 

def preprocess(x):
  image = tf.image.convert_image_dtype(x['image'], tf.float32)
  subclass_labels = tf.one_hot(x['label'], builder.info.features['label'].num_classes)
  return image, subclass_labels

def loadMnist():
    mnist_train = tfds.load('mnist', split='train', shuffle_files=False).cache()
    mnist_train = mnist_train.map(preprocess)
    mnist_train = mnist_train.shuffle(builder.info.splits['train'].num_examples)
    mnist_train = mnist_train.batch(BATCH_SIZE, drop_remainder=True)

    mnist_test = tfds.load('mnist', split='test').cache()
    mnist_test = mnist_test.map(preprocess).batch(BATCH_SIZE)
    return mnist_train, mnist_test

def getTeacherModel():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=1))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(128 , activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(NUM_CLASSES))
    return model

def getTeachingAssistantModel():
    model = tf.keras.Sequential()
    # your code start from here for step 2
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Dense(784 , activation='relu'))
    model.add(tf.keras.layers.Dense(784 , activation='relu'))
    model.add(tf.keras.layers.Dense(NUM_CLASSES))
    return model

def getStudentModel():
    model = tf.keras.Sequential()
    # your code start from here for step 2
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Dense(784 , activation='relu'))
    model.add(tf.keras.layers.Dense(784 , activation='relu'))
    model.add(tf.keras.layers.Dense(NUM_CLASSES))
    return model

@tf.function
def compute_num_correct(model, images, labels):
  """Compute number of correctly classified images in a batch.

  Args:
    model: Instance of tf.keras.Model.
    images: Tensor representing a batch of images.
    labels: Tensor representing a batch of labels.

  Returns:
    Number of correctly classified images.
  """
  class_logits = model(images, training=False)
  return tf.reduce_sum(
      tf.cast(tf.math.equal(tf.argmax(class_logits, -1), tf.argmax(labels, -1)),
              tf.float32)), tf.argmax(class_logits, -1), tf.argmax(labels, -1)

@tf.function
def compute_loss(model,images, labels):
    """Compute subclass knowledge distillation teacher loss for given images
        and labels.

    Args:
    images: Tensor representing a batch of images.
    labels: Tensor representing a batch of labels.

    Returns:
    Scalar loss Tensor. 
    """
    subclass_logits = model(images, training=True)
    cross_entropy_loss_value = tf.math.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=subclass_logits,labels=labels))
    return cross_entropy_loss_value

def train_and_evaluate(model,trainingData, testingData):
    """Perform training and evaluation for the teacher model model.

    Args:
    model: Instance of tf.keras.Model.
    compute_loss_fn: A function that computes the training loss given the
        images, and labels.
    """

    # your code start from here for step 4
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    for epoch in range(1, NUM_EPOCHS + 1):
        # Run training.
        print('Epoch {}: '.format(epoch), end='')
        for images, labels in trainingData:
            with tf.GradientTape() as tape:
                loss_value = compute_loss(model,images,labels)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Run evaluation.
        num_correct = 0
        num_total = builder.info.splits['test'].num_examples
        for images, labels in testingData:
            # your code start from here for step 4
            num_correct += compute_num_correct(model,images,labels)[0]
        print("Class_accuracy: " + '{:.2f}%'.format(
            num_correct / num_total * 100))

def distillation_loss(teacher_logits: tf.Tensor, student_logits: tf.Tensor,temperature: Union[float, tf.Tensor]):
    """Compute distillation loss.

    This function computes cross entropy between softened logits and softened
    targets. The resulting loss is scaled by the squared temperature so that
    the gradient magnitude remains approximately constant as the temperature is
    changed. For reference, see Hinton et al., 2014, "Distilling the knowledge in
    a neural network."

    Args:
        teacher_logits: A Tensor of logits provided by the teacher.
        student_logits: A Tensor of logits provided by the student, of the same
        shape as `teacher_logits`.
        temperature: Temperature to use for distillation.

    Returns:
        A scalar Tensor containing the distillation loss.
    """
    # your code start from here for step 3
    soft_targets = tf.nn.softmax(teacher_logits/temperature)

    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            soft_targets, student_logits / temperature)) * temperature ** 2

def compute_student_loss_using_KD(studentModel, teacherModel, images, labels, alpha, temprature):
    """Compute subclass knowledge distillation student loss for given images
        and labels.

    Args:
        images: Tensor representing a batch of images.
        labels: Tensor representing a batch of labels.

    Returns:
        Scalar loss Tensor.
    """
    student_subclass_logits = studentModel(images, training=True)

    # Compute subclass distillation loss between student subclass logits and
    # softened teacher subclass targets probabilities.

    # your code start from here for step 3

    teacher_subclass_logits = teacherModel(images, training=False)
    distillation_loss_value = distillation_loss(teacher_subclass_logits,student_subclass_logits,temprature)

    # Compute cross-entropy loss with hard targets.

    # your code start from here for step 3
    hard_targets = tf.math.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels,student_subclass_logits))
    cross_entropy_loss_value =  hard_targets + alpha * distillation_loss_value

    return cross_entropy_loss_value

def train_and_evaluate_using_KD(studentModel, teacherModel,trainingData, testingData, alpha, temprature):
    """Perform training and evaluation for the teacher model model.

    Args:
    model: Instance of tf.keras.Model.
    compute_loss_fn: A function that computes the training loss given the
        images, and labels.
    """

    # your code start from here for step 4
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    for epoch in range(1, NUM_EPOCHS + 1):
        # Run training.
        print('Epoch {}: '.format(epoch), end='')
        for images, labels in trainingData:
            with tf.GradientTape() as tape:
                loss_value = compute_student_loss_using_KD(studentModel, teacherModel,images,labels, alpha, temprature)
            grads = tape.gradient(loss_value, studentModel.trainable_variables)
            optimizer.apply_gradients(zip(grads, studentModel.trainable_variables))

        # Run evaluation.
        num_correct = 0
        num_total = builder.info.splits['test'].num_examples
        for images, labels in testingData:
            # your code start from here for step 4
            num_correct += compute_num_correct(studentModel,images,labels)[0]
        print("Class_accuracy: " + '{:.2f}%'.format(
            num_correct / num_total * 100))

def testModel(model,testData):
    num_correct = 0
    num_total = builder.info.splits['test'].num_examples

    for images, labels in testData:
        num_correct += compute_num_correct(model,images,labels)[0]
    print("model Testing Accuracy: " + '{:.2f}%'.format(
        (num_correct / num_total) * 100))
    return (num_correct / num_total) * 100
