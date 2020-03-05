import sys
import argparse
import os
import torch


def list_directories(path):
    return [directory for directory in os.listdir(path) if os.path.isdir(os.path.join(path, directory))]


# Gets model version as latest, bump or version specified as integer
def get_version(args):
    if args.version is None:
        print('Please define --version latest xor --version bump xor --version N')
        exit(1)

    name_prefix = _extract_prefix(args)

    existing_models = list_directories('ecg_data/ecg/runs')
    models_with_name = [
        model for model in existing_models if model.startswith(name_prefix)]
    if len(models_with_name) == 0:
        return 1

    model_numbers = [int(model.split('.')[-1]) for model in models_with_name]

    if not (args.version == 'latest' or args.version == 'bump'):
        try:
            value = int(args.version)
            print("Using explicit version")
            return value
        except ValueError:
            print("Couldn't parse version as a number: {}".format(args.version))
            exit(1)
    else:
        if args.version == 'latest':
            print("Using latest version")
            return max(model_numbers)
        elif args.version == 'bump':
            print("Creating newer version")
            return max(model_numbers) + 1


def _extract_prefix(args):
    return '{}_{}_op{}_{}_lr{}_loss{}_cb{}_sampler{}_'.format(
        args.dataset, args.model_name, args.optimizer, args.scheduler, args.lr,
        args.criterion, args.class_balance, args.sampler)


def extract_name(args):
    return _extract_prefix(args) + args.version


def parse_args():
    parser = argparse.ArgumentParser(description='')

    #  experiment settings
    # name of dataset used in the experiment, e.g. gtsrd
    parser.add_argument('--dataset', default='ecg', type=str,
                        help='name of dataset to train upon')
    parser.add_argument('--test', dest='test', action='store_true', default=False,
                        help='To only run inference on test set')
    parser.add_argument('--verbose',  action='store_true', default=False,
                        help='Print all validation predicitons to console')

    # main folder for data storage
    parser.add_argument('--root-dir', type=str, default='ecg_data')

    # model settings
    parser.add_argument('--model-name', type=str,
                        help='type of model to be used. Particular instance of a given architecture, e.g. cnn1d_3')
    parser.add_argument('--version', type=str, default='bump',
                        help='Chosse between [latest] [bump] or number [N]')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='which checkpoint to resume from possible values ["latest", "best", epoch]')
    parser.add_argument('--pretrained', action='store_true',
                        default=False, help='use pre-trained model')

    # data settings
    parser.add_argument('--num-classes', default=5, type=int)
    # number of workers for the dataloader
    parser.add_argument('-j', '--workers', type=int, default=1)

    # training settings
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--step', type=int, default=5,
                        help='frequency of updating learning rate, given in epochs')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 70)')
    parser.add_argument('--optimizer', default='adam', type=str,
                        help='name of the optimizer')
    parser.add_argument('--scheduler', default='StepLR', type=str,
                        help='name of the learning rate scheduler')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='sgd momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--lr-decay', default=0.995, type=float,
                        metavar='lrd', help='learning rate decay (default: 0.995)')
    parser.add_argument('--criterion', default='nll', type=str,
                        help='criterion to optimize')
    parser.add_argument('--class-balance', default=None, type=str, metavar='cb',
                        help='class balancing (loss augmenting) scheme from ["equal", "importance"], cannot be used with sampler')
    parser.add_argument('--sampler', default=None, type=str, metavar='sampling',
                        help='sampling scheme from ["equal", "importance"], cannot be used with class-balance')

    # misc settings
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--short-run', action='store_true',
                        default=False, help='running only over few mini-batches for debugging purposes')
    parser.add_argument('--disable-cuda', action='store_true', default=False,
                        help='disables CUDA training / using only CPU')
    parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', default=False,
                        help='Use tensorboard to track and plot')

    args = parser.parse_args()

    # update args
    args.version = get_version(args)
    args.name = extract_name(args)
    args.data_dir = '{}/{}'.format(args.root_dir, args.dataset)
    args.log_dir = '{}/runs/{}/'.format(args.data_dir, args.name)
    args.res_dir = '%s/runs/%s/res' % (args.data_dir, args.name)
    args.out_pred_dir = '%s/runs/%s/pred' % (args.data_dir, args.name)

    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    args.device = 'cuda' if args.cuda else 'cpu'

    assert args.data_dir is not None
    assert args.num_classes > 0
    assert not (args.sampler and args.class_balance)

    if args.verbose:
        print(' '.join(sys.argv))
        print(args)

    return args
