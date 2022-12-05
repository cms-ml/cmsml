import numpy as np
import tensorflow as tf

def count_parameter(model, verbose=True):
    number_of_parameter = np.sum(
        [np.prod(np.array(p.shape.as_list())) for p in model.trainable_variables])
    if verbose:
        print(f'Model has {number_of_parameter} parameter')
    return number_of_parameter

def create_tensorboard_visualization(graph, tensorboard_dir):
    """Takes *graph* and create a tensorboard visualization at *tensorboard_dir*

    Args:
        graph (str): path to frozen graph
        tensorboard_dir (str): path where the summary writer saves the visualization
    """
    # read graph definition
    with tf.gfile.GFile(graph, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # now build the graph in the memory and visualize it
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        writer = tf.summary.FileWriter(tensorboard_dir, graph)
        writer.close()
