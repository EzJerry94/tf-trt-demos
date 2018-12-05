import tensorflow as tf
from tensorflow.python.framework import graph_util
from hparams import hparams
from models import create_model


if __name__ == '__main__':
    model_folder = 'logs-tacotron'
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path
    print(input_checkpoint)
    output_graph = 'frozen_model.pb'
    output_node_names = ['model/inference/dense/BiasAdd']

    tf.reset_default_graph()
    with tf.variable_scope('datafeeder') as scope:
        inputs = tf.placeholder(tf.int32, [None, None], 'inputs')
        input_lengths = tf.placeholder(tf.int32, [None], 'input_lengths')
    with tf.variable_scope('model') as scope:
        model = create_model('tacotron', hparams)
        model.initialize(inputs, input_lengths)

    graph = tf.get_default_graph()
    saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, input_checkpoint)
        input_graph_def = graph.as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names
        )
        with tf.gfile.GFile(output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        with open('operations.txt', 'w') as wf:
            for op in graph.get_operations():
                wf.write(op.name)
                wf.write('\n')
        print('%d ops in the final graph.' % len(output_graph_def.node))