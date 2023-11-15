# coding: utf-8

"""
Script that reads a tensorflow graph from a model file and ahead-of-time compiles it for selected batch-sizes using XLA.
"""

from __future__ import annotations

import os

from cmsml.util import colored, interruptable_popen
from cmsml.tensorflow.tools import import_tf, load_model


def compile_tf_graph(
    model_path: str,
    output_path: str,
    batch_sizes: tuple[int] = (1,),
    input_serving_key: str = "serving_default",
    output_serving_key: str | None = None,
    compile_prefix: str | None = None,
    compile_class: str | None = None,
) -> None:
    """
    For AOT compilation a static memory layout at runtime is required. This function prepares the given input SavedModel
    to make it ready for AOT compilation

    This function takes the subgraph saved under the *input_serving_key* signature within a given SavedModel, stored in
    *model_path*, and creates a 'ConcreteFunction' with a static shape for given *batch_sizes*. If not
    *input_serving_key* is given the TensorFlow default 'serving_default' is used.

    The resulting static 'ConcreteFunction' is saved as subgraph under a new *output_serving_key* signature in a
    SavedModel stored at *output_path*. If no *output_serving_key* is given the 'ConcreteFunction' are saved with the
    signature "{*input_serving_key*}_bs{*batch_size*}".

    An optional AOT compilation is initiated if *compile_class* and *compile_prefix* are given. In this case
    *compile_prefix* is the file prefix, while *compile_class* is the name of the AOT class within the generated files.
    """
    tf = import_tf()[0]

    # default output_serving key
    if not output_serving_key:
        output_serving_key = input_serving_key + "_bs{}"

    # check compile values
    if compile_prefix and not compile_class:
        raise ValueError("when compile_prefix is set, compile_class must not be empty")
    if compile_class and not compile_prefix:
        raise ValueError("when compile_class is set, compile_prefix must not be empty")

    # get the model object
    model = load_model(model_path)

    # get the tf function
    func = model.signatures[input_serving_key]

    # prepare the output directory
    output_path = os.path.expandvars(os.path.expanduser(str(output_path)))
    if os.path.isfile(output_path):
        raise OSError(f"output_path existing and points to file: {output_path}")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # create concrete functions per batch size
    c_funcs = {}
    for bs in sorted(set(map(int, batch_sizes))):
        # create a fully defined signature, filling leading None's in shapes with the batch size
        specs = {}
        for key, spec in model.signatures["serving_default"].structured_input_signature[1].items():
            # ignore inputs without undefined axes
            if None not in spec.shape:
                continue
            # create new shape and name
            shape = [
                (bs if n is None else n)
                for n in spec.shape
            ]
            # : is the delimiter of ops numering scheme
            name = f"{spec.name.replace(':', '_')}_bs{bs}"
            # store the new spec
            specs[key] = type(spec)(type(spec.shape)(shape), dtype=spec.dtype, name=name)

        # concrete function
        c_funcs[output_serving_key.format(bs)] = tf.function(func).get_concrete_function(**specs)

    # save concrete functions as signatures of the model
    tf.saved_model.save(model, output_path, signatures=c_funcs)
    print(f"saved model at '{colored(output_path, 'magenta')}'")

    # optionally compile
    if compile_prefix and compile_class:
        aot_compile(
            output_path,
            os.path.join(output_path, "aot"),
            compile_prefix,
            compile_class,
            batch_sizes=batch_sizes,
            serving_key=output_serving_key,
        )


def aot_compile(
    model_path: str,
    output_path: str,
    prefix: str,
    class_name: str,
    batch_sizes: tuple[int] = (1,),
    serving_key: str = r"serving_default_bs{}",
) -> None:
    """
    Loads the graph from the SavedModel located at *model_path*, extracts the static graph specified by *serving_key*
    from it, AOT compiles it.

    This process generates header and object files at *output_path*. The *class_name* is used as class name within the
    header access the AOT-compiled network.
    """
    # prepare model path
    model_path = os.path.abspath(os.path.expandvars(os.path.expanduser(str(model_path))))

    # merge output path and prefix, and split them again with prefix being the basename
    output_path = os.path.expandvars(os.path.expanduser(str(output_path)))
    prefix = os.path.expandvars(os.path.expanduser(str(prefix)))
    output_path, prefix = os.path.split(os.path.join(output_path, prefix))

    # prepare the output directory
    if os.path.isfile(output_path):
        raise OSError(f"output_path existing and points to file: {output_path}")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # get the compilation executable
    exe = _which_saved_model_cli()

    # compile for each batch size
    for bs in sorted(set(map(int, batch_sizes))):
        cmd = (
            f"{exe} aot_compile_cpu"
            f" --dir {model_path}"
            f" --signature_def_key {serving_key.format(bs)}"
            f" --output_prefix {prefix.format(bs)}"
            f" --cpp_class {class_name.format(bs)}"
            " --tag_set serve"
        )

        print(f"compiling for batch size {colored(bs, 'magenta')}")
        code = interruptable_popen(cmd, executable="/bin/bash", shell=True, cwd=output_path)[0]
        if code != 0:
            raise Exception(f"aot compilation using {exe} failed with exit code {code}")


def _which_saved_model_cli() -> str:
    """
    Determines the ``saved_model_cli`` executable that is used for the AOT compilation.
    """
    # prefer executable set by CMSML_SAVED_MODEL_CLI
    exe = os.getenv("CMSML_SAVED_MODEL_CLI")
    if exe:
        return exe

    # try usual candidates
    for exe in ["saved_model_cli", "saved_model_cli3"]:
        cmd = f"type {exe} &> /dev/null"
        code = interruptable_popen(cmd, executable="/bin/bash", shell=True)[0]
        if code == 0:
            return exe

    # return default and let subsequent tools potentially fail
    return "saved_model_cli"


def main() -> None:
    from argparse import ArgumentParser

    parser = ArgumentParser(
        prog=f"cmsml_{os.path.splitext(os.path.basename(__file__))[0]}",
        description="ahead-of-time (AOT) compiles TensorFlow graphs for fixed batch sizes with XLA",
    )

    parser.add_argument(
        "model_path",
        help="the path of the model to open",
    )
    parser.add_argument(
        "output_path",
        help="the path where compiled models should be stored",
    )
    parser.add_argument(
        "--batch-sizes",
        "-b",
        default=(1,),
        type=(lambda s: tuple(map(int, s.strip().split(",")))),
        help="comma-separated list of batch sizes to convert the model for; default: 1",
    )
    parser.add_argument(
        "--input-serving-key",
        default="serving_default",
        help="serving key of the model in --src; default: serving_default",
    )
    parser.add_argument(
        "--output-serving-key",
        help=r"serving key pattern for concrete models in --output-path, with {} being replaced by "
        r"the batch size; default: <input_serving_key>__bs{}",
    )
    parser.add_argument(
        "--compile",
        "-c",
        nargs=2,
        help=r"file name prefix and class name of the AOT compiled objects; in both values, {} is "
        "replaced by the batch size; no AOT compilation is triggered when empty; files will be "
        "saved at <output_path>/aot/<prefix>{.h,.o,_metadata.o,_makefile.inc}",
    )

    args = parser.parse_args()

    compile_tf_graph(
        model_path=args.model_path,
        output_path=args.output_path,
        batch_sizes=args.batch_sizes,
        input_serving_key=args.input_serving_key,
        output_serving_key=args.output_serving_key,
        compile_prefix=args.compile and args.compile[0],
        compile_class=args.compile and args.compile[1],
    )


if __name__ == "__main__":
    main()
