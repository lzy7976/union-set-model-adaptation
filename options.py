import argparse


def load_base_options():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--source', type=str, nargs='*', help='source domains: Synscapes, GTA5, SYNTHIA')
    parser.add_argument('--target', type=str, default='Cityscapes', help='target domain: Cityscapes')
    parser.add_argument('--label_setting', type=str, choices=['f', 'p', 'n'], help='f=fully-overlapping, p=partly-overlapping, n=non-overlapping')
    parser.add_argument('--num_classes', type=int, default=19)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--gpu_id', type=int, default=0)
    return parser


def load_train_options():
    base_parser = load_base_options()
    parser = argparse.ArgumentParser(parents=[base_parser])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=2.5e-4)
    parser.add_argument('--num_steps', type=int, default=80000)
    parser.add_argument('--power', type=float, default=0.9)
    parser.add_argument('--save_freq', type=int, default=5000)
    parser.add_argument('--print_freq', type=int, default=1000)
    parser.add_argument('--test_freq', type=int, default=5000)
    parser.add_argument('--restore_path', type=str)
    parser.add_argument('--pseudo_label_path', type=str)
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints')
    return parser.parse_args()


def load_test_options():
    base_parser = load_base_options()
    parser = argparse.ArgumentParser(parents=[base_parser])
    parser.add_argument('--stage', type=int)
    parser.add_argument('--restore_path', type=str)
    return parser.parse_args()


def load_psl_options():
    base_parser = load_base_options()
    parser = argparse.ArgumentParser(parents=[base_parser])
    parser.add_argument('--stage', type=int)
    parser.add_argument('--restore_path', type=str)
    parser.add_argument('--pseudo_label_path', type=str)
    return parser.parse_args()