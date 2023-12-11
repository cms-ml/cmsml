# coding: utf-8

"""
Script that provides insight on which TensorFlow operations are XLA / AOT compatible and whether a specified graph would
be supported.
"""

from __future__ import annotations

import tabulate

from cmsml.util import colored
from cmsml.tensorflow.tools import load_graph_def
from cmsml.tensorflow.aot import OpsData, get_graph_ops


def check_aot_compatibility(
    model_path: str,
    serving_key: str = "serving_default",
    devices: tuple[str] = ("cpu",),
    table_format: str = "grid",
) -> None:
    """
    Loads model stored in *model_path* and extracts the GraphDef saved under the specified *serving_key*. From this
    GraphDef, all ops for specific *devices* are read and compared to all ops with XLA implementation. The matching
    result is printed given the chosen *table_format* style.
    """
    # open the graph
    graph_def = load_graph_def(model_path, serving_key=serving_key)

    # extract operation names
    op_names = get_graph_ops(graph_def)

    # remove trivial ops
    op_names = [op_name for op_name in op_names if op_name not in ["Placeholder", "NoOp"]]

    # print the op table
    devices, ops = print_op_table(devices, filter_ops=op_names, table_format=table_format)

    # print a final summary per device
    for device in devices:
        failed_ops = [
            op_name
            for op_name in op_names
            if not ops.get(op_name, {}).get(device)
        ]

        msg = f"\n{colored(device, 'magenta')}: "
        if failed_ops:
            msg += colored("not compatible", "red")
            msg += f", {len(failed_ops)} incompatible ops: {', '.join(failed_ops)}"
        else:
            msg += colored("all ops compatible", "green")
        print(msg)


def print_op_table(
    devices: tuple[str],
    filter_ops: list[str] | None = None,
    table_format: str = "grid",
) -> tuple[list[str], OpsData]:
    """
    Reads all ops for specific *devices* and prints a table given *table_format* style. Specific ops can be filtered
    using *filter_ops*.
    """
    # read ops
    ops = OpsData(devices)

    # get parsed devices
    devices = [
        device
        for device in ops.device_ids
        if any(
            op_data.get(device)
            for op_name, op_data in ops.items()
            if not filter_ops or op_name in filter_ops
        )
    ]
    devices = sorted(set(devices), key=devices.index)

    # prepare the table
    headers = ["Operation"] + devices
    content = []
    str_flag = lambda b: "yes" if b else "NO"
    for op_name, op_data in ops.items():
        if filter_ops and op_name not in filter_ops:
            continue

        content.append([
            op_name,
            *(str_flag(bool(op_data.get(device))) for device in devices),
        ])

    # print it
    print(tabulate.tabulate(content, headers=headers, tablefmt=table_format))

    return devices, ops


def main() -> None:
    import os
    import sys
    from argparse import ArgumentParser

    parser = ArgumentParser(
        prog=f"cmsml_{os.path.splitext(os.path.basename(__file__))[0]}",
        description="performs XLA / AOT compatiblity checks on a TensorFlow graph",
    )

    parser.add_argument(
        "model_path",
        nargs="?",
        help="the path of the model to open",
    )
    parser.add_argument(
        "--serving-key",
        "-k",
        default="serving_default",
        help="serving key of the graph in --model-path; default: serving_default",
    )
    parser.add_argument(
        "--table",
        "-t",
        action="store_true",
        help="just print a table showing which operations are XLA / AOT supported for --devices",
    )
    parser.add_argument(
        "--table-format",
        "-f",
        default="grid",
        help="the tabulate format for printed tables; default: grid",
    )
    parser.add_argument(
        "--devices",
        "-d",
        type=(lambda s: tuple(s.strip().split(","))),
        help="comma separated list of devices to check; choices: cpu,gpu,tpu, default: cpu",
    )

    args = parser.parse_args()

    if args.table:
        # print the op table
        print_op_table(
            devices=args.devices,
            table_format=args.table_format,
        )

    elif args.model_path:
        # run the compatibility check
        check_aot_compatibility(
            model_path=args.model_path,
            serving_key=args.serving_key,
            devices=args.devices,
            table_format=args.table_format,
        )

    else:
        print("either '--model-path PATH' or '--table' must be set", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
