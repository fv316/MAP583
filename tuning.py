"""Module used to tune hyperparamets of our models."""

from ray import tune

from args import parse_args
from commander import run_from_args
from models import models_to_arch


def create_args(config):
    args = [
        "--root-dir", "/home/svp/Programming/MAP583/ecg_data",
        "--tensorboard", "--version", "bump"]

    for arg_name, arg_value in config.items():
        args.extend(["--{}".format(arg_name), str(arg_value)])

    # TODO: fix root-dir
    return parse_args(args)


def tuning(config):
    args = create_args(config)
    result = run_from_args(args)
    tune.track.log(mean_average_precision=result["mAP"])
    # TODO: use average class accuracy


if __name__ == "__main__":
    search_space = {
        "lr": tune.grid_search([
            # 1e-08,
            # 1e-05,
            1e-03,
            # 1e-02,
            1e-01,
        ]),
        "model-name": tune.grid_search(
            list(models_to_arch.keys()))
    }

    analysis = tune.run(tuning,
                        config=search_space)

    print("Best configuration: ", analysis.get_best_config(
        metric="mean_average_precision"))
