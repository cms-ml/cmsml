import tensorflow as tf
from pathlib import Path
import cms_ml
import numpy as np


def count_parameter(model, verbose=True):
    number_of_parameter = np.sum(
        [np.prod(np.array(p.shape.as_list())) for p in model.trainable_variables])

    if verbose:
        print(f'Model has {number_of_parameter} parameter')
    return number_of_parameter


def build_toy_model(input_shape: tuple,
                    batch_size: int = None,
                    verbose_summary: bool = False,
                    number_of_layers: int = 25,
                    nodes: int = 128):
    """
      Build toy model that was used to test AOT compilation. The model is build like:

      The number of parameters are calculated by:
      Batchnorm add 2 * Number of Nodes
      Bias adds 1 * Number of Nodes
      Hiddenlayer = HIDDENLAYER * Number of Nodes^2

      IF HIDDENLAYER > 0
    {  (INPUTS+3) * 64    } + HIDDENLAYER * (128^2 + 3*128) + 128*10 + 10
      First Layer

      IF HIDDENLAYER == 0
      (INPUTS+3) * 64 + (64 + 1) * 10

    Args:
        input_shape (tuple): 1D input shape of the network
        batch_size (int, optional): used batch size
        verbose_summary (bool, optional): print summary
        number_of_layers (int, optional): how many hidden layers

    Returns:
        tf.keras.model: build Tensorflow Model

    """
    from tensorflow.keras import layers
    from tensorflow.keras import models

    nn = models.Sequential()

    nn.add(layers.Input(shape=input_shape, batch_size=batch_size, name='input'))
    nn.add(layers.Dense(units=64))
    nn.add(layers.BatchNormalization())
    nn.add(layers.Activation('selu'))
    for i in range(0, number_of_layers):
        nn.add(layers.Dense(units=nodes))
        nn.add(layers.BatchNormalization())
        nn.add(layers.Activation('selu'))
    nn.add(layers.Dense(units=10, activation='softmax', name='output'))

    nn.build()

    if verbose_summary:
        print(nn.summary())
        count_parameter(nn, True)
    return nn


def load_model(path):
    model = tf.saved_model.load(path)
    return model


def save_graph(path, model, variable_constant):
    cms_ml.save_graph(str(Path(path).with_suffix('.pb')), model, variable_constant)


def load_graph(path):
    return cms_ml.load_graph(str(Path(path)))


def main(dst, input_shape=32, number_of_layers=4, save_as_keras_model=False, nodes=256):
    network = build_toy_model(input_shape, None, None, number_of_layers)

    dst = Path(dst)
    create_dir = False

    # check if you want to create the dir
    if not dst.exists:
        print(f'Dir {dst} does not exists, want me to create it? (Y/N)')
        confirmation = input()
        if confirmation.lower() == 'y':
            create_dir = True

    # set dst suffix
    suffix = '_KERAS' if save_as_keras_model else "_TF^"
    dst = dst.with_stem(dst.stem + suffix)

    if create_dir:
        dst.mkdir(parents=True, exist_ok=True)

    if save_as_keras_model:
        tf.keras.models.save_model(network, str(dst))
    else:
        tf.saved_model.save(network, str(dst))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="""Tool to freeze a tf saved model and make it
    useable with the Cpp API of tensorflow in CMSSW""",
                                     prog='freeze')

    parser.add_argument('-i',
                        '--input_shape',
                        help="""Input shape of the network.""",
                        type=int,
                        required=True,
                        nargs='+',
                        dest='input_shape')

    parser.add_argument('-dst',
                        help="""Destination path of the toy model. If depending on type of saving
                        method a suffix is added: _KERAS if a keras model is saved and _TF if a
                        tensorflow core model is saved. If dst path does not exist, it will be created""",
                        type=str,
                        required=True,
                        dest='dst')

    parser.add_argument('--keras',
                        help="""Save the model as Keras model if True. Otherwise as tensorflow model""",
                        type=bool,
                        required=False,
                        dest='save_as_keras_model')

    parser.add_argument('-l',
                        '--number_of_layers',
                        help="""Number of hidden layers""",
                        type=int,
                        required=False,
                        dest='number_of_layers')

    parser.add_argument('-n',
                        '--nodes',
                        help="""Number of nodes in the hidden layer""",
                        type=int,
                        required=False,
                        dest='nodes')

    args = parser.parse_args()
    main(**args.__dict__)
