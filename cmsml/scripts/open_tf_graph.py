# coding: utf-8

"""
Script that starts a tensorboard process and loads a graph for visualization.
"""


import os
import tempfile
import subprocess
import signal
import shutil
import argparse


def main():
    from cmsml.tensorflow import write_graph_summary

    parser = argparse.ArgumentParser(
        prog="cmsml_open_tf_graph",
        description="Takes a TensorFlow graph that was previously saved to a protobuf file and "
        "opens a tensorboard server to visualize it",
    )

    parser.add_argument(
        "graph_path",
        help="the path of the graph to open",
    )
    parser.add_argument(
        "--log-dir",
        "-l",
        help="the tensorboard logdir, temporary when not set",
    )
    parser.add_argument(
        "--txt",
        "-t",
        action="store_true",
        help="force reading the graph as text",
    )
    parser.add_argument(
        "--binary",
        "-b",
        action="store_true",
        help="force reading the graph as a binary",
    )
    parser.add_argument(
        "--tensorboard-args",
        "-a",
        help="optional arguments to pass to the tensorboard command",
    )

    args = parser.parse_args()

    # prepare the log_dir
    log_dir = args.log_dir
    is_tmp = not args.log_dir
    if is_tmp:
        log_dir = tempfile.mkdtemp()
    elif not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # prepare the as_text flag
    as_text = None
    if args.txt:
        as_text = True
    elif args.binary:
        as_text = False

    # write the summary
    write_graph_summary(args.graph_path, log_dir, as_text=as_text)

    # build the command
    cmd = f"tensorboard --logdir '{log_dir}'"
    if args.tensorboard_args:
        cmd += " " + args.tensorboard_args

    # start the tensorboard process
    print(f"starting tensorboard with command: {cmd}")
    p = subprocess.Popen(cmd, shell=True, executable="/bin/bash", preexec_fn=os.setsid)
    try:
        p.communicate()
    except (Exception, KeyboardInterrupt):
        print("tensorboard terminated")
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)

    # cleanup when log_dir is temporary
    if is_tmp:
        shutil.rmtree(log_dir)


if __name__ == "__main__":
    main()
