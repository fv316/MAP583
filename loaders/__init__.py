from loaders.ecg_loader import ECGLoader, ECGLoader_bin


def get_loader(args):
    """get_loader

    :param name:
    """
    return {
        'ecg': ECGLoader,
        'ecg_bin': ECGLoader_bin,
    }[args.dataset]
