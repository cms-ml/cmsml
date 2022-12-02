import cms_ml
import tensorflow as tf
from pathlib import Path


def freeze_graph(src: str, dst: str = '') -> None:
    """Freezes graph of a toy saved model and make it useable by the Cpp tensorflow API.
    The Graphs name is always <dst/frozen_graph>.
    If dst is blank ("") graph is saved in src dir.

    Args:
        src (str): Source dir of tf saved model
        dst (str, optional): Destination path

    Raises:
        ValueError: If src dir does not exist
    """
    src = Path(src)
    if not src.exists():
        raise ValueError(f"Source directory does not exist: \n {str(src)}")

    model = tf.saved_model.load(str(src))
    # extract concrete function
    # works only with toy model, with 1 input node and 32 input features
    concrete_function = tf.function(model.__call__).get_concrete_function(
        tf.TensorSpec(shape=(None, 32), dtype=tf.float32, name='input_0')
    )

    if dst == '':
        dst = src.joinpath('frozen_graph.pb')
    dst_path = Path(dst)

    # using Marcel Riegers (https://github.com/cms-ml/cmsml)
    print(f'Saving at {str(dst)}')
    # TODO: check if works with multiple output nodes
    cms_ml.save_graph(str(dst_path),
                      concrete_function,
                      True,
                      output_names='output_0')
    print('Done')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="""Tool to freeze a tf saved model and make it
    useable with the Cpp API of tensorflow in CMSSW""",
                                     prog='freeze')

    parser.add_argument('-src',
                        help="""Path to the tensorflow saved model that you want to freeze""",
                        type=str,
                        required=True,
                        dest='src')

    parser.add_argument('-dst',
                        help="""Destination path of the created frozen graph.
                        Following special cases are allowed:\n
                        \t - If dst="", this is the default value, the graph in the dir of the
                        source saved model with the name "frozen_graph".\n
                        \t - Anything else is interpreted as directory path""",
                        type=str,
                        required=True,
                        dest='dst')

    args = parser.parse_args()
    print('\n' * 3)
    print('Start Running conversion:')
    freeze_graph(src=args.src,
                 dst=args.dst)
