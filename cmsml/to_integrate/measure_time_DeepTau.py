import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf


def set_threads(num_threads: int):
    tf.config.threading.set_inter_op_parallelism_threads(num_threads)
    tf.config.threading.set_intra_op_parallelism_threads(num_threads)


def measure_time_of_function(func):
    """
    Decorator to measure run time of <func>
    """
    from time import perf_counter

    def timed_function(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        end = perf_counter()
        time = end - start
        return time, result
    return timed_function


def return_device(device='CPU', num_device='0'):
    """
    Wrapper to return tf.device. Checks if GPU existst
    Args:
        device: cpu or gpu, Default: CPU

    Returns: tf.device
    """
    device = device.upper()

    if device == 'GPU':
        if len(tf.config.list_physical_devices()) == 0:
            raise ValueError(
                '\n !!! There is no GPU on the working Machine !!! \n')
        device = f'/device:GPU:{num_device}'
    elif device == 'CPU':
        device = f'/device:CPU:{num_device}'

    device = tf.device(device)
    return device


def get_input_names(model):
    """
    Gets the name of the inputs for DeepTau this will return:
        'input_inner_egamma',
        'input_inner_hadrons',
        'input_inner_muon',
        'input_outer_egamma',
        'input_outer_hadrons',
        'input_outer_muon', 'input_tau'


    Args:
        model (tf.model): DeepTau Keras/TF Model

    Returns:
        list(str): All input names in the correct order
    """
    models_input_layer = model.inputs
    names = []

    for input_layer in models_input_layer:
        shape = input_layer.shape.as_list()
        if not shape:
            # skip empty
            continue
        name = input_layer.name
        name_without_indice = name.replace(':0', '')
        names.append(name_without_indice)
    return names


def create_input_dataset(model: 'tf.model', batch_size: int, num_events: int = 1, fill: str = 'zero') -> 'tf.Dataset':
    models_input_layer = model.inputs
    datasets = []

    for input_layer in models_input_layer:
        shape = input_layer.shape.as_list()
        if not shape:
            continue
        samples = batch_size * num_events
        shape[0] = samples
        dtype = input_layer.dtype
        name = input_layer.name
        if fill == 'random':
            node_input_tensor = tf.random.uniform(shape,
                                                  minval=0,
                                                  maxval=None,
                                                  dtype=dtype,
                                                  seed=None,
                                                  name=name)
        elif fill == 'zero':
            node_input_tensor = tf.zeros(shape,
                                         dtype=dtype,
                                         name=name)

        input_dataset = tf.data.Dataset.from_tensor_slices(node_input_tensor)
        datasets.append(input_dataset)
    return tf.data.Dataset.zip(tuple(datasets)).batch(batch_size)


def measure_runtime(graph_obj, dataset, verbose=False):
    """
    Runs a graph with given dataset for multiple times and measure the time.

    Args:
        graph_obj: Graph
        dataset: TF.Dataset
        repeats: int > 0

    Returns: List with computation times per batch event
    """
    runtime_times = []
    names = get_input_names(graph_obj)
    for batch in dataset:

        input = {}
        for inp, n in zip(batch, names):
            input[n] = inp
        runtime, prediction = measure_time_of_function(graph_obj)(**input)
        runtime_times.append(runtime)
        if verbose:
            print(f'Prediction: {prediction}')
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


def main(saved_model_path, name='',
         number_of_datapoints=1,
         batch_sizes=(1,),
         device='CPU', threads=0, warmup=1, runtime_dir='dry', verbose=False):

    set_threads(threads)

    # sets threads, 0 = take as much as you need

    tf_device = return_device(device, 0)

    saved_model_path = Path(saved_model_path)
    duration_times = defaultdict(dict)  # creates a dict with dicts as defaullt value

    for batch_size in batch_sizes:
        signature_key = ''.join(('batch_size_', str(batch_size)))
        graph = load_model(str(saved_model_path),
                           signature_key)

        input_dataset = create_input_dataset(graph,
                                             batch_size,
                                             num_events=number_of_datapoints,
                                             fill='zero')

        with tf_device:
            # duration is time on event basis in seconds!
            duration = measure_runtime(graph_obj=graph,
                                       dataset=input_dataset,
                                       verbose=verbose)

        duration_times[str(batch_size)] = duration[warmup:]

        dur_arr = (np.array(duration) * 1000 / batch_size).mean()
        dur_std = (np.array(duration) * 1000 / batch_size).std()
        print(f'For Batchsize {batch_size} the time is: {dur_arr} +- {dur_std}')

    # save run time in the runtime dir
    if runtime_dir != 'dry':
        version = f'TFv_{tf.__version__}'.replace('.', '-')
        full_name = '_'.join((version, name, device, 'thread-' + str(threads)))
        performance_path = Path(runtime_dir).joinpath(full_name).with_suffix('.json')

        with open(performance_path, 'w') as outfile:
            print(f'\n Saving performance under the name {full_name}')
            json.dump(duration_times, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--saved_model_dir',
                        help='Saved model of DeepTau',
                        type=str,
                        default='dry',
                        dest='src')

    parser.add_argument('-o',
                        '--runtime_dir',
                        help='Save directory of the runtime data',
                        type=str,
                        default='dry',
                        dest='runtime_dir')

    parser.add_argument('-b',
                        '--batch_size',
                        help='How many events are sampled',
                        type=int,
                        nargs='+',
                        default=[1],
                        dest='batch_sizes')

    parser.add_argument('--num_events',
                        help='Number of events',
                        type=int,
                        default=1,
                        dest='number_of_events')

    parser.add_argument('--threads',
                        help='Set the amount of threads for executing the graph',
                        type=int,
                        default=0,
                        dest='threads')

    parser.add_argument('--device',
                        help='Try to use cpu or gpu. cpu is fallback',
                        type=str,
                        default='cpu',
                        dest='device')

    parser.add_argument('--name',
                        type=str,
                        dest='name')

    parser.add_argument('--verbose',
                        action='store_true')

    args = parser.parse_args()
    print('\n' * 3)
    print('Start Running script')

    main(args.src,
         name=args.name,
         number_of_datapoints=args.number_of_events,
         batch_sizes=args.batch_sizes,
         device=args.device,
         threads=args.threads,
         warmup=100,
         verbose=args.verbose)
