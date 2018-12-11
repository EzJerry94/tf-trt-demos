import tensorflow as tf
from tensorflow.python.framework import graph_util

if __name__ == '__main__':
    model_folder = 'models'
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path
    print(input_checkpoint)
    output_graph = 'frozen_model.pb'
    output_node_names = ['ArgMax','decoder_targets']

    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session(graph=graph) as sess:
        #sess.run(tf.global_variables_initializer())
        saver.restore(sess, input_checkpoint)
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