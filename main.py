import argparse

import maskexp.model.train_raw as train_raw
import maskexp.model.train_weighted as train_weighted
import maskexp.model.train_rordinal as train_rordinal
import maskexp.model.train_wordinal as train_wordinal

MODEL_TYPES = ['raw', 'weighted', 'raw_ordinal', 'weighted_ordinal']


def get_train_func_by_type(type_str):
    assert type_str in MODEL_TYPES
    if type_str == 'raw':
        return train_raw.train
    elif type_str == 'weighted':
        return train_weighted.train
    elif type_str == 'weighted_ordinal':
        return train_wordinal.train
    elif type_str == 'raw_ordinal':
        return train_rordinal.train

    return train_raw.train


def train(args):
    """

    :param args:
    :return:
    """
    train_fn = get_train_func_by_type(args.train_type)
    train_fn(name=args.name, data_path=args.data_path, device_str=args.device,
             resume_from=args.resume, save_dir=args.save_dir)


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
        default='save',
        help='Directory to save checkpoints. If not provided, checkpoints are not saved.'
    )

    args = parser.parse_args()

    # Display the arguments
    print(f"Model Name: {args.name}")
    print(f"Data Path: {args.data_path}")
    print(f"Device: {args.device}")
    print(f"Train Type: {args.train_type}")
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
    if args.save_dir:
        print(f"Checkpoints will be saved to: {args.save_dir}")

    train(args)


if __name__ == '__main__':
    main()
