import argparse
import tensorflow as tf
from keras import backend as K
from DeepTau_utilis import LoadModel
from pathlib import Path


def main(src_dir, dst_dir):
    """
    Converts H5 DeepTau Fullmodel to Saved Model with signatures batch_size_N, N being a potence of 2, ranging from 0 to 14.

    Args:
        input: dir of the Model.h5
        dst_dir: dir of the dst_dir saved_model

    Returns: None

    """

    # tf.keras.Model.save(model,'./temp')
    # exit()
    print('STARTING TO CONVERT .h5 DeepTau Model to Saved_Model')
    # Loads the network from h5 file
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=2,
                                      inter_op_parallelism_threads=2,
                                      allow_soft_placement=True,
                                      device_count={'CPU': 1, 'GPU': 0})

    session = tf.compat.v1.Session(config=config)
    K.set_session(session)
    K.set_learning_phase(0)  # set into inference mode

    model_path = str(Path(src_dir))
    model = LoadModel(model_path, False)
    print('Model loaded')

    batch_sizes = [pow(2, potence) for potence in range(0, 14)]
    signature_dictionary = {}
    for i, batch in enumerate(batch_sizes):
        # shapes of all inputs of deeptau (only full mode)
        # these inputs are made by hand, by using the saved_model_cli show --dir DIR --all
        # look the correct order of inputs are marked at the concrete function of call
        input_tau = (batch, 47)  # inp 0
        input_inner_egamma = (batch, 11, 11, 86)  # inp 1
        input_inner_muon = (batch, 11, 11, 64)  # inp 2
        input_inner_hadrons = (batch, 11, 11, 38)  # inp 3
        input_outer_egamma = (batch, 21, 21, 86)  # inp 4
        input_outer_muon = (batch, 21, 21, 64)  # inp 5
        input_outer_hadrons = (batch, 21, 21, 38)  # inp 6

        # get conrecte function of the __call__ function of the network
        # the shapes are fixed in this case for all batch sizes
        signature = tf.function(model.__call__).get_concrete_function((
            [tf.TensorSpec(shape=input_tau, dtype=tf.float32, name='input_0')],
            [tf.TensorSpec(shape=input_inner_egamma, dtype=tf.float32, name='input_1')],
            [tf.TensorSpec(shape=input_inner_muon, dtype=tf.float32, name='input_2')],
            [tf.TensorSpec(shape=input_inner_hadrons, dtype=tf.float32, name='input_3')],
            [tf.TensorSpec(shape=input_outer_egamma, dtype=tf.float32, name='input_4')],
            [tf.TensorSpec(shape=input_outer_muon, dtype=tf.float32, name='input_5')],
            [tf.TensorSpec(shape=input_outer_hadrons, dtype=tf.float32, name='input_6')]))

        # prints all shapes of the arguments of __call__
        # to confirm they are right
        print(signature)

        # save signature in dictionary
        key = ''.join(('batch_size_', str(batch)))
        print(f'Saving Model with servingkey {key}')
        signature_dictionary[key] = signature

    dst_model_path = Path(dst_dir)
    tf.saved_model.save(model, str(dst_model_path), signatures=signature_dictionary)
    print(f'Destionation of saved model: {str(dst_model_path)}')
    print('DONE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deploy keras model.')
    parser.add_argument('--src_dir', required=True, type=str, help='Input Keras model')
    parser.add_argument('--dst_dir', required=True, type=str,
                        help='Destination directory of the saved model file')
    args = parser.parse_args()

    main(args.src_dir, args.dst_dir)
