import tensorflow as tf
import time
import numpy as np
import argparse
from hparams import hparams
from models import create_model
from text import text_to_sequence
from util import audio
from tensorflow.contrib import tensorrt as trt


# parameters
workspace_size_bytes = 1 << 30
trt_gpu_ops = tf.GPUOptions(per_process_gpu_memory_fraction = 0.50)

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

def main(args):
    # load classification graph def
    frozen_model_file = 'frozen_model.pb'
    frozen_graph_def = tf.GraphDef()
    with tf.gfile.Open(frozen_model_file, 'rb') as f:
        data = f.read()
        frozen_graph_def.ParseFromString(data)
    print('Loaded classifier graph def')

    trt_graph_def = trt.create_inference_graph(
        input_graph_def=frozen_graph_def,
        outputs=['model/inference/dense/BiasAdd', 'model/inference/embedding'],
        max_batch_size=args.batch_size,
        max_workspace_size_bytes=workspace_size_bytes,
        precision_mode=args.precision_mode
    )

    # for only 'INT8'
    # trt_graph_def = trt.calib_graph_to_infer_graph(trt_graph_def)
    print('Generated TensorRT graph def')

    # generate tensor with TensorRT graph
    tf.reset_default_graph()
    with tf.Graph().as_default() as graph:
        inputs, input_lengths, linear_outputs = tf.import_graph_def(
            trt_graph_def,
            return_elements=['datafeeder/inputs:0',
                             'datafeeder/input_lengths:0',
                             'model/inference/dense/BiasAdd:0']
        )
    print('Generated tensor by TensorRT graph')

    with tf.Session(graph=graph) as sess:
        for text in sentences:
            cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
            seq = text_to_sequence(text, cleaner_names)
            np_inputs = [np.asarray(seq, dtype=np.int32)]
            np_input_lengths = np.asarray([len(seq)], dtype=np.int32)
            new_np_inputs = np.tile(np_inputs, (5, 1))
            new_np_input_lengths = np.tile(np_input_lengths, (5))
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
    parser.add_argument('--precision_mode', choices=['FP32', 'FP16', 'INT8'], default='FP32')
    args = parser.parse_args()
    main(args)