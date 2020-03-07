from models.cnn1d import cnn1d
from models.resnet1d import resnet1d
from models.resnet1d_v2 import resnet1d_v2
from models.lstm import lstm


def get_model(args):
    arch = get_arch_from_model(args.model_name)
    print('Fetching model %s - %s ' % (arch, args.model_name))

    model_generator = get_generator(arch)
    model = model_generator(
        args.model_name, num_classes=args.num_classes, masking=args.masking)

    return model


def get_generator(arch):
    return {
        'cnn1d': cnn1d,
        'resnet1d': resnet1d,
        'resnet1d_v2': resnet1d_v2,
        'lstm': lstm
    }[arch]


models_to_arch = {
    'cnn1d_3': 'cnn1d',
    'lstm_1': 'lstm',
    'lstm_2': 'lstm',
    'lstm_3': 'lstm',
    'resnet1d_v2_18': 'resnet1d_v2',
    'resnet1d_v2_10': 'resnet1d_v2',
    'resnet1d_18': 'resnet1d',
    'resnet1d_18': 'resnet1d',
}


def get_arch_from_model(model_name):
    return models_to_arch[model_name]
