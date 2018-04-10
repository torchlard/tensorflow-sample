import numpy as np
import tensorflow as tf
import iris_data

tf.logging.set_verbosity(tf.logging.INFO)

# def input_evaluation_set():
#     features = {'SepalLength': np.array([6.4,5.0]),
#                 'SepalWidth': np.array([2.8, 2.3]),
#                 'PetalLength': np.array([5.6, 3.3]),
#                 'PetalWidth': np.array([2.2, 1.0])}
#     labels = np.array([2,1])
#     return features, labels


(train_x, train_y), (test_x, test_y) = iris_data.load_data()

my_feature_columns = []
for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))


classifier = tf.estimator.DNNClassifier(
    feature_columns = my_feature_columns,
    hidden_units=[10,10],
    n_classes=3
)

batch_size = 100

classifier.train(
    input_fn = lambda:iris_data.train_input_fn(train_x,train_y, batch_size),
    steps = 200
)

# tensors_to_log = {"probabilities": "softmax_tensor"}
#
# logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

eval_result = classifier.evaluate(
    input_fn = lambda:iris_data.eval_input_fn(test_x, test_y, batch_size)
    # hooks = [logging_hook]
)


print('\nTest set accuracy: {0}\n'.format(**eval_result))
print(eval_result)






