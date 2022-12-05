import argparse


def main():
    from cmsml.to_integrate.tool_convert_model_shape_to_static import ModelConverter

    parser = argparse.ArgumentParser(description="""Tool to set batch size dimension to a static
    value and save it in a saved model graph with a keyword of name: batch_size_N, with N being your given batch size""",
                                     prog='Prog')

    parser.add_argument('-src',
                        help="""Path to the tensorflow savedmodel. A serving_key,--key,
                         is necessary if a custom key is used.""",
                        type=str,
                        required=True,
                        dest='src')

    parser.add_argument('-dst',
                        help="""Destination path of the static saved model.
                        Following special cases are allowed:\n
                        \t - If dst="", this is the default value, the graph is saved in the same
                        directory as the source, with the suffix "_static".\n
                        \t - If dst='same' the graph is saved within the src TF_saved_model\n
                        \t - Anything else is just a interpreted as path to the saved_model dir""",
                        type=str,
                        required=True,
                        dest='dst')

    parser.add_argument('--serving_key',
                        help="""A saved model graph is saved with a "key", the signature, in the *.pb file.
                        One *.pb file can habits multiple graphs and the serving_key is the name of the graph.
                        The default is "serving_default". """,
                        type=str,
                        default='serving_default',
                        required=False,
                        dest='serving_key')

    parser.add_argument('-b',
                        '--batch_sizes',
                        help=""" Pass a string with ints using spaces as delimiter. e.g. "1 2 4"
                        will create a model with static batch sizes 1 2 and 4.
                        This sets the suffix of the serving key under which the mode is saved.
                        From our example the model with batch size 1 has the key: batch_size_1.^""",
                        type=str,
                        required=True,
                        default='1',
                        dest='batch_sizes')

    args = parser.parse_args()

    converter = ModelConverter(args.src,
                               args.dst,
                               serving_key=args.serving_key,
                               batch_sizes=args.batch_sizes)

    converter.create_static_signatures()
    converter.save_signatures()


if __name__ == '__main__':
    main()
