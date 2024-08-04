import argparse

import maskexp.model.train_weighted as train_weighted
import maskexp.model.train_ordinal as train_ordinal

MODEL_TYPES = ['raw', 'weighted', 'raw_ordinal', 'weighted_ordinal']


def get_train_func_by_type(type_str):
    assert type_str in MODEL_TYPES
    if type_str == 'weighted':
        return train_weighted.train


def train(args):
    """

    :param args:
    :return:
    """


def main():
    parser = argparse.ArgumentParser(description="======Train Masked Expressiveness======")
    # Required arguments
    parser.add_argument(
        '--name',
        type=str,
        required=True,
        help='Provide name for the model (as well as ckpt file name)'
    )

    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Provide a path to the dataset for training.'
    )

    parser.add_argument(
        '--device',
        type=str,
        required=True,
        choices=['cuda', 'mps', 'cpu'],
        help='The device type to be used for training (e.g., cuda, mps, cpu).'
    )

    # Optional arguments
    parser.add_argument(
        '--train_type',
        type=str,
        choices=MODEL_TYPES,
        default='raw',
        help="choose a type from predefined models. Options: raw, weighted, raw_ordinal, weighted_ordinal"
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to the model checkpoint to resume training from. If not provided, training starts from scratch.'
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
        help='Directory to save checkpoints. If not provided, checkpoints are not saved.'
    )

    args = parser.parse_args()

    # Display the arguments
    print(f"Data Path: {args.data_path}")
    print(f"Device: {args.device}")
    print(f"Train Type: {args.train_type}")
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
    if args.save_dir:
        print(f"Checkpoints will be saved to: {args.save_dir}")

    # Your training logic here
    # train_model(data_path=args.data_path, device=args.device, ...)

    # train_velocitymlm()
    # continue_velocitymlm()


if __name__ == '__main__':
    main()
