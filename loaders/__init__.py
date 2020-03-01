from loaders.ecg_loader import ECGLoader


def get_loader(args):
    """get_loader

    :param name:
    """
    return {
        'ecg': ECGLoader,
    }[args.dataset]
