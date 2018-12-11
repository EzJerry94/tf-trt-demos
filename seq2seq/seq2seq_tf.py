import time
import numpy as np
import tensorflow as tf
import argparse

EOS = 1


def random_sequences(length_from, length_to, vocab_lower, vocab_upper, batch_size):
    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)

    while True:
        yield [
            np.random.randint(low=vocab_lower, high=vocab_upper, size=random_length()).tolist()
            for _ in range(batch_size)
        ]

def make_batch(inputs, max_sequence_length=None):
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)
    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)
    return inputs_time_major, sequence_lengths

def load_graph(frozen_model):
    tf.reset_default_graph()
    # parse the graph_def file
    with tf.gfile.GFile(frozen_model, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        encoder_inputs, decoder_inputs, decoder_targets, decoder_prediction = tf.import_graph_def(
            graph_def,
            return_elements=['encoder_inputs:0', 'decoder_inputs:0', 'decoder_targets:0', 'ArgMax:0']
        )
    print('Generated tensor by frozen graph')
    return graph, encoder_inputs, decoder_inputs, decoder_targets, decoder_prediction

def main(args):
    batches = random_sequences(length_from=3, length_to=10, vocab_lower=2, vocab_upper=10, batch_size=args.batch_size)
    frozen_model = 'frozen_model.pb'
    graph, encoder_inputs, decoder_inputs, decoder_targets, decoder_prediction = load_graph(frozen_model)

    with tf.Session(graph=graph) as sess:
        for _ in range(args.roll):
            batch = next(batches)
            encoder_inputs_, _ = make_batch(batch)
            decoder_targets_, _ = make_batch([(sequence) + [EOS] for sequence in batch])
            decoder_inputs_, _ = make_batch([[EOS] + (sequence) for sequence in batch])
            feed_dict = {encoder_inputs: encoder_inputs_,
                        decoder_inputs: decoder_inputs_,
                        decoder_targets: decoder_targets_}
            start_time = time.process_time()
            predict_ = sess.run(decoder_prediction, feed_dict)
            stop_time = time.process_time()
            for i, (inp, pred) in enumerate(zip(feed_dict[encoder_inputs].T, predict_.T)):
                print('input > {}'.format(inp))
                print('predicted > {}'.format(pred))
                if i >= 10:
                    break
            print('{:.2f} milliseconds'.format((stop_time - start_time) * 1000))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--roll', default=3, type=int)
    args = parser.parse_args()
    main(args)