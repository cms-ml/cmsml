import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf

import cms_ml
import utilis


def create_input_array(num_events=1, num_inputs=32, batch_size=1):
    """
    create and returns tf.tensor of shape (batch_size * num_events, num_inputs)
    Args:
        num_events: # of events
        num_inputs: # of inputs
        batch_size: # batch

    Returns: tf.tensor with uniform distributed random numbers of float32 type.
    """
    # mutlplication guarantee that we got exakt num_events number of batches in each dataset
    num_of_batches = num_events * batch_size
    shape = (num_of_batches, num_inputs)
    inp = tf.random.uniform(shape,
                            minval=0,
                            maxval=None,
                            dtype=tf.dtypes.float32,
                            seed=None,
                            name=None)
    return inp


def dataset(numpy_array, batch_size):
    """
    create tf.dataset from input np.array and slice it into batches of size batch_size
    Args:
        numpy_array: np.array
        batch_size: int

    Returns: iterator of dataset that returns batches.
    """
    return tf.data.Dataset.from_tensor_slices(numpy_array).batch(batch_size)


@utilis.measure_time_of_function
def measure_runtime_of_tf1_graph(graph, input, signiture=('input:0', 'output:0')):
    """
    Runs Graph under the convention of TF1 convention with sessions. Returns np.array as Graph output.

    Args:
        graph: computational graph
        input: input data
        signiture: signature of the node, the defaults are input:N , output:N, with N being the N node

    Returns: measured time and result of prediction as numpy array
    """
    session = tf.compat.v1.Session(graph=graph)
    with graph.as_default():
        x = graph.get_tensor_by_name(signiture[0])
        y = graph.get_tensor_by_name(signiture[1])
        out = session.run(y, {x: input})
    return out


@utilis.measure_time_of_function
def measure_runtime_of_tf2_graph(graph, input, jit_compile=False):
    """
    Runs Graph in tensorflow 2, which is simply using the call method of the graph object.
    Returns result as TF.Tensor
    Args:
        graph: Loaded Graph Def
        input: Input.Tensor

    Returns: measured time and result of prediction as tf.tensor
    """
    if jit_compile:
        graph = tf.function(graph, jit_compile=True)
    return graph(input)


@utilis.measure_time_of_function
def run_keras_model(model, input):
    """
    Runs Keras model in Eager mode (without creating a graph) with given dataset for multiple times and measure the time.

    Args:
        model: Keras Model
        input: input tensor

    Returns: measured time and result of prediction
    """
    return model.predict(input)


@tf.function(jit_compile=True)
def measure_runtime(graph_obj, dataset, jit_compile=False):
    """
    Runs a graph with given dataset for multiple times and measure the time.

    Args:
        graph_obj: Graph
        dataset: TF.Dataset
        repeats: int > 0

    Returns: List with computation times per batch event
    """
    runtime_times = []
    for batch in dataset:
        if utilis.using_TF1():
            runtime, _ = measure_runtime_of_tf1_graph(
                graph_obj, batch, signiture=['input:0', 'Identity:0'])
        else:
            runtime, _ = measure_runtime_of_tf2_graph(graph_obj, batch, jit_compile)
        runtime_times.append(runtime)
    return runtime_times


def save_model(model, dir, input_signature='serving_default'):
    """
    Saves Keras Model as SavedModelFormat
    Args:
        model: Keras Model
        dir:  Path to saving directory
        input_signature: String, name under which the Graph is saved.

    Returns: None

    """
    dir = Path(dir)
    print(f'Saving Model under path: {dir}')
    tf.saved_model.save(model, str(dir), input_signature)


def load_model(dir, serving_key='serving_default'):
    """
    Loads models graph saved by tf.saved_model.save and returns the graph
    Args:
        dir: path of saved_model.
        serving_key: string under which the graph is saved, dedfault: serving_default

    Returns: Graph Obj.

    """
    dir = Path(dir)
    loaded = tf.saved_model.load(str(dir))
    inference_func = loaded.signatures[serving_key]
    # there is a bug for TFV < 2.8 where python garbage collector kills the model (weak link)
    inference_func._backref_to_saved_model = loaded
    return inference_func


def main(name='version',
         number_of_inputs=32,
         number_of_datapoints=100,
         batch_sizes=(1),
         eager_or_graph='graph',
         device='cpu', threads=0, jit_compile=False):

    # sets threads, 0 = take as much as you need
    tf.config.threading.set_inter_op_parallelism_threads(threads)
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.compat.v1.disable_eager_execution()
    # set device to run on
    tf_device = utilis.check_device(device)

    # Paths zum speicher
    if name == 'version':
        name = f'TFv_{tf.__version__}'.replace('.', '-')

    DIRPATH = Path(__file__).resolve().parent
    GRAPH_DIR = DIRPATH.joinpath('graphs')
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_FILE = GRAPH_DIR.joinpath(Path(name))

    print('\n Save and Load Graph of the network:')

    neural_network = utilis.build_toy_model(number_of_inputs, None, False)
    if utilis.using_TF1():
        cms_ml.save_graph(path=GRAPH_FILE, variables_to_constants=True,
                          obj=neural_network, output_names=['output'])
        graph = cms_ml.load_graph(GRAPH_FILE)
    else:
        save_model(neural_network, GRAPH_FILE, None)
        graph = load_model(GRAPH_FILE)

    # Saves the Graph and Loads it (to ensure that no eager mode interferes)
    duration_times = defaultdict(dict)  # creates a dict with dicts as defaullt value
    for batch_size in batch_sizes:
        # Prepare Data # iterable dataset
        input_tensor = create_input_array(num_events=number_of_datapoints, num_inputs=number_of_inputs,
                                          batch_size=batch_size)
        d_set = dataset(input_tensor, batch_size)

        # ---- MEASUREMENT STARTS HERE ----

        #jit_scope = tf.contrib.compiler.jit.experimental_jit_scope
        # tf.config.optimizer.set_jit(True) # Enable XLA.
        with tf_device:
            if eager_or_graph == 'eager':
                duration = run_keras_model(model=neural_network, dataset=d_set)
            else:
                # duration is time on event basis in seconds!
                with utilis.jit_scope(compile_ops=jit_compile):
                    duration = measure_runtime(graph_obj=graph, dataset=d_set, jit_compile=jit_compile)
        # tf.config.optimizer.set_jit(False) # Enable XLA.
        # saves times on event basis in dictionary with batch_size as key

        duration_times[str(batch_size)] = duration[100:]

        dur_arr = (np.array(duration) * 1000 / batch_size).mean()
        dur_std = (np.array(duration) * 1000 / batch_size).std()
        print(f'For Batchsize {batch_size} the time is: {dur_arr} +- {dur_std}')

    # saving times dictionary as json file
    name = f'TFv_{tf.__version__}'.replace('.', '-')
    full_name = '_'.join((name, eager_or_graph, device, str(
        number_of_datapoints), 'thread-' + str(threads)))
    performance_path = Path('./performance/').joinpath(full_name).with_suffix('.json')
    print(f'\n Saving performance under the name {full_name}')

    with open(performance_path, 'w') as outfile:
        json.dump(duration_times, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o',
                        '--runtime_dir',
                        help='Save directory of the runtime data',
                        type=str,
                        default='version',
                        dest='runtime_dir')

    parser.add_argument('-b',
                        '--batch_size',
                        help='How many events are sampled',
                        type=int,
                        nargs='+',
                        default=[1])

    parser.add_argument('-ni',
                        '--num_inp',
                        help='Number of Inputs of the Network',
                        type=int,
                        default=32)

    parser.add_argument('--num_events',
                        help='Number of events',
                        type=int,
                        default=100)

    parser.add_argument('--v_diff',
                        help='Verbose the difference between Eager and Graph Mode',
                        type=bool,
                        default=False)

    parser.add_argument('--eager_or_graph',
                        help='Use Eager or Graph mode',
                        type=str,
                        default='graph')

    parser.add_argument('--threads',
                        help='Set the amount of threads for executing the graph',
                        type=int,
                        default=0)

    parser.add_argument('--device',
                        help='Try to use cpu or gpu. cpu is fallback',
                        type=str,
                        default='cpu')

    parser.add_argument('--jit_compile',
                        help='enables jit compile',
                        type=bool,
                        default=False)

    parser.add_argument('--name',
                        help='Name of the performance file',
                        type=str)

    args = parser.parse_args()

    print('Using following arguments:')
    for arg in vars(args):
        print(arg, getattr(args, arg))

    print('\n' * 3)
    print('Start Running script')

    main(name=args.save_dir, number_of_inputs=args.num_inp, number_of_datapoints=args.num_events,
         batch_sizes=args.batch_size, eager_or_graph=args.eager_or_graph, device=args.device, threads=int(args.threads))
