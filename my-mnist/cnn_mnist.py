import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    # [batch,28,28,1]
    input_layer = tf.reshape(features["x"], [-1,28,28,1])

    # [batch,28,28,32]
    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters=32, kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu
    )
    # [batch, 14,14,32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)

    # [batch, 14,14,64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5,5],
        padding="same", activation=tf.nn.relu
    )
    # [batch, 7,7,64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)

    # [-1, 3136]
    pool2_flat = tf.reshape(pool2, [-1,7*7*64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024)
    # [batch, 1024], dropout only performed if training is True
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    # [batch, 10]
    logits = tf.layers.dense(inputs=dropout, units=10)

    # == eval ==
    predictions = {
        # argmax returns the largest value in axis=1
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # if predict
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # eg. [0,1,0,0,0,0,0,0,0,0], ...
    # onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    # loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # softmax_cross = one hot version of labels used in sparse_softmax_cross
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # if train
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]
        )}
    # if evaluation
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



def main(unused_argv):
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model"
    )

    tensors_to_log = {"probabilities": "softmax_tensor"}
    # print value of tensors every N local steps
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # return input fn that has signature (feed dict of numpy array into model)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True
    )
    mnist_classifier.train(
        input_fn=train_input_fn, steps=20000, hooks=[logging_hook]
    )

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False
    )
    eval_results = mnist_classifier.evaluate(input_fn = eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
    tf.app.run()





