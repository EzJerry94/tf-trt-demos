import tensorflow as tf
import time
import numpy as np
import argparse
from hparams import hparams
from text import text_to_sequence


sentences = [
    # From July 8, 2017 New York Times:
    'Scientists at the CERN laboratory say they have discovered a new particle.',
    'There’s a way to measure the acute emotional intelligence that has never gone out of style.',
    'President Trump met with other leaders at the Group of 20 conference.',
    'The Senate\'s bill to repeal and replace the Affordable Care Act is now imperiled.',
    # From Google's Tacotron example page:
    'Generative adversarial network or variational auto-encoder.',
    'The buses aren\'t the problem, they actually provide a solution.',
    'Does the quick brown fox jump over the lazy dog?',
    'Talib Kweli confirmed to AllHipHop that he will be releasing an album in the next year.',
]

def load_graph(frozen_model):
    tf.reset_default_graph()
    # parse the graph_def file
    with tf.gfile.GFile(frozen_model, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        inputs, input_lengths, linear_outputs = tf.import_graph_def(
            graph_def,
            return_elements=['datafeeder/inputs:0',
                             'datafeeder/input_lengths:0',
                             'model/inference/dense/BiasAdd:0']
        )
    print('Generated tensor by frozen graph')
    return graph, inputs, input_lengths, linear_outputs

def main(args):
    graph, inputs, input_lengths, linear_outputs = load_graph('frozen_model.pb')
    with tf.Session(graph=graph) as sess:
        for text in sentences:
            cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
            seq = text_to_sequence(text, cleaner_names)
            np_inputs = [np.asarray(seq, dtype=np.int32)]
            np_input_lengths = np.asarray([len(seq)], dtype=np.int32)
            new_np_inputs = np.tile(np_inputs, (args.batch_size, 1))
            new_np_input_lengths = np.tile(np_input_lengths, (args.batch_size))
            feed_dict = {
                inputs: new_np_inputs,
                input_lengths: new_np_input_lengths
            }
            start_time = time.process_time()
            linear_res = sess.run(linear_outputs, feed_dict=feed_dict)
            stop_time = time.process_time()
            print('{:.2f} milliseconds'.format((stop_time - start_time) * 1000))
            #print(linear_res.shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    args = parser.parse_args()
    main(args)