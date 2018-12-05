import time
import numpy as np
import argparse
import tensorflow as tf
from tensorflow.contrib import tensorrt as trt


def main(args):
    workspace_size_bytes = 1 << 30
    trt_gpu_ops = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)

    # transform images (image -> input vector)
    tf.reset_default_graph()
    g1 = tf.Graph()
    with g1.as_default():
        # create graph
        in_images = tf.placeholder(tf.string, name='in_images')
        decoded_input = tf.image.decode_png(in_images, channels=3)
        float_input = tf.cast(decoded_input, dtype=tf.float32)
        # (224, 224, 3) -> (n, 224, 224, 3)
        rgb_input = tf.expand_dims(float_input, axis=0)
        # for VGG preprocess, reduce means and convert to BGR
        slice_red = tf.slice(rgb_input, [0, 0, 0, 0], [1, 224, 224, 1])
        slice_green = tf.slice(rgb_input, [0, 0, 0, 1], [1, 224, 224, 1])
        slice_blue = tf.slice(rgb_input, [0, 0, 0, 2], [1, 224, 224, 1])
        sub_red = tf.subtract(slice_red, 123.68)
        sub_green = tf.subtract(slice_green, 116.779)
        sub_blue = tf.subtract(slice_blue, 103.939)
        transferred_input = tf.concat([sub_blue, sub_green, sub_red], 3)
        # transform to vectors
        with tf.Session(config=tf.ConfigProto(gpu_options=trt_gpu_ops)) as s1:
            with open('tiger224x224.jpg', 'rb') as f:
                data1 = f.read()
                imglist1 = s1.run([transferred_input], feed_dict={in_images: data1})
                image1 = imglist1[0]
            with open('lion224x224.jpg', 'rb') as f:
                data2 = f.read()
                imglist2 = s1.run([transferred_input], feed_dict={in_images: data2})
                image2 = imglist2[0]
            with open('orangutan224x224.jpg', 'rb') as f:
                data3 = f.read()
                imglist3 = s1.run([transferred_input], feed_dict={in_images: data3})
                image3 = imglist3[0]
    print('Loaded image vectors (tiger, lion, orangutan')

    # When you test batch, please uncomment here. (single prediction is executed by default)
    image1 = np.tile(image1, (args.batch_size, 1, 1, 1))
    image2 = np.tile(image2, (args.batch_size, 1, 1, 1))
    image3 = np.tile(image3, (args.batch_size, 1, 1, 1))

    # load classification graph def
    classifier_model_file = 'resnetV150_frozen.pb'
    classifier_graph_def = tf.GraphDef()
    with tf.gfile.Open(classifier_model_file, 'rb') as f:
        data = f.read()
        classifier_graph_def.ParseFromString(data)
    print('Loaded classifier graph def')

    trt_graph_def = trt.create_inference_graph(
        input_graph_def=classifier_graph_def,
        outputs=['resnet_v1_50/predictions/Reshape_1'],
        max_batch_size=args.batch_size,
        max_workspace_size_bytes=workspace_size_bytes,
        precision_mode=args.precision_mode
    )

    if args.precision_mode == 'INT8':
        trt_graph_def = trt.calib_graph_to_infer_graph(trt_graph_def)
    print('Generated TensorRT graph def')

    # generate tensor with TensorRT graph
    tf.reset_default_graph()
    g2 = tf.Graph()
    with g2.as_default():
        trt_x, trt_y = tf.import_graph_def(
            trt_graph_def,
            return_elements=['input:0', 'resnet_v1_50/predictions/Reshape_1:0']
        )
    print('Generated tensor by TensorRT graph')

    # run classification with TensorRT graph
    with open('imagenet_classes.txt', 'rb') as f:
        labeltext = f.read()
        classes_entries = labeltext.splitlines()

    with tf.Session(graph=g2, config=tf.ConfigProto(gpu_options=trt_gpu_ops)) as s2:
        eval_list = [image1, image2, image3]
        for img in eval_list:
            start_time = time.process_time()
            result = s2.run([trt_y], feed_dict={trt_x: img})
            stop_time = time.process_time()
            # list -> 1 x n ndarray : feature's format is [[1.16643378e-06 3.12126781e-06 3.39836406e-05 ... ]]
            nd_result = result[0]
            # remove row's dimension
            onedim_result = nd_result[0,]
            # set column index to array of possibilities
            indexed_result = enumerate(onedim_result)
            # sort with possibilities
            sorted_result = sorted(indexed_result, key=lambda x: x[1], reverse=True)
            # get the names of top5 possibilities
            print('********************')
            for top in sorted_result[:5]:
                print(classes_entries[top[0]], 'confidence:', top[1])
            print('{:.2f} milliseconds'.format((stop_time - start_time) * 1000))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--precision_mode', choices=['FP32', 'FP16', 'INT8'], default='FP32')
    args = parser.parse_args()
    main(args)