"""Module used to tune hyperparamets of our models."""

from ray import tune

from args import parse_args
from commander import run_from_args
from models import models_to_arch
import random

def create_args(config):
    args = [
        "--root-dir", "/home/szymonpajzert/MAP583/ecg_data",
        "--tensorboard", "--version", "bump"]

    for arg_name, arg_value in config.items():
        if arg_value is not None:
            args.extend(["--{}".format(arg_name), str(arg_value)])

    # TODO: fix root-dir
    return parse_args(args)


def tuning(config):
    args = create_args(config)
    result = run_from_args(args)

    tune.track.log(mean_average_precision=result["mAP"].val,
                   acc_class=result['acc_class'].val,
                   acc_val=result['acc1'].val)


if __name__ == "__main__":
    resnet_args = {
        "kernel-size":  tune.grid_search([3, 5, 7]),
        "adaptive-size": tune.grid_search([1, 2, 3])
    }

    params = {
        "cnn1d_3": {
            "conv1-size": tune.grid_search([5, 8, 10, 20]),
            "conv2-size": tune.grid_search([10, 20, 32, 40]),
            "conv-kernel-size": tune.grid_search([5, 7, 10]),
        },
        "lstm": {
            'input-dim': tune.grid_search([10, 50, 100, 187]),
            'hidden-dim': tune.grid_search([30, 50, 100, 200]),
            'layer-dim': tune.grid_search([1, 2, 3, 4]),
        },
        'resnet1d_v2_18': resnet_args,
        'resnet1d_v2_10': resnet_args
    }

    fields = set()
    for p in params.values():
        fields.update(p.keys())

    for model_name, model_params in params.items():
        search_space = {
            "lr": tune.grid_search([
                # 1e-08,
                # 1e-05,
                1e-03,
                1e-02,
                1e-01,
            ]),
            'optimizer': tune.grid_search([
                'sgd', 'adam'
            ])
        }
        search_space["model-name"] = tune.grid_search([model_name])
        for field in fields:
            search_space[field] = model_params.get(field, tune.grid_search([-1]))
        
        analysis = tune.run(tuning,
                            config=search_space,
                            resources_per_trial={
                                "cpu": 1,
                            })

        print("Best configuration for model {}: ".format(model_name), analysis.get_best_config(
            metric="mean_average_precision"))
