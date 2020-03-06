from models.resnet import resnet
from models.squeezenet import squeezenet
from models.lenet import lenet
from models.cnn1d import cnn1d
from models.resnet1d import resnet1d
from models.resnet1d_v2 import resnet1d_v2
from models.lstm import lstm


def get_model(args):
    arch = get_arch_from_model(args.model_name)
    print('Fetching model %s - %s ' % (arch, args.model_name))

    model_generator = get_generator(arch)
    model = model_generator(args.model_name,
                            num_classes=args.num_classes,
                            masking=args.masking,
                            lstm_window=args.lstm_window)

    return model


def get_generator(arch):
    return {
        'resnet': resnet,
        'squeezenet': squeezenet,
        'lenet': lenet,
        'cnn1d': cnn1d,
        'resnet1d': resnet1d,
        'resnet1d_v2': resnet1d_v2,
        'lstm': lstm
    }[arch]


def get_arch_from_model(model_name):
    return {
        'cnn1d_3': 'cnn1d',
        'lenet5': 'lenet',
        'lstm_1': 'lstm',
        'resnet18': 'resnet',
        'resnet34': 'resnet',
        'resnet50': 'resnet',
        'resnet1d_v2_18': 'resnet1d_v2',
        'resnet1d_v2_10': 'resnet1d_v2',
        'resnet1d_18': 'resnet1d',
        'resnet1d_18': 'resnet1d',
        'squeezenet1_0': 'squeezenet',
        'squeezenet1_1': 'squeezenet'
    }[model_name]
