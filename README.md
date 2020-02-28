# pytorch-ecg
This is the project directory for Projet group 4: Healthcare (ECG & cardiology) for the [DL course @ école polytechnique](https://mlelarge.github.io/dataflowr-web/). For this project the MIT-BIH Arrhythmia Dataset is used, it is available on Kaggle: [ECG Heartbeat Categorization Dataset](https://www.kaggle.com/shayanfazeli/heartbeat).

This directory has been adapted from the github project [dldiy-gtsrb](https://github.com/abursuc/dldiy-gtsrb). Visit the link to see how a directry could be structured for easier testing of new architectures and parameters, tracking of results and improving of the models. View the [Directives list](https://github.com/fv316/MAP583/blob/master/Francisco/Directives.txt) for next steps and grading instructions.


## Data
The zipped data can download from [here](https://drive.google.com/file/d/17Rd4YpGwssSpk4xZAT5AyYskjvs95dAY/view?usp=sharing). Note that on running the `commander.py` script for the first time, the data is automatically downloaded, unzipped, and saved for future use.


## Project structure

The project is structured as following:

```bash
.
├── loaders
|   └── dataset selector
|   └── ecg_loader.py # loading and pre-processing ecg data
|   └── gdd.py # loading and unzipping data from google drive
├── models
|   └── architecture selector
|   └── lenet.py # classical CNN
|   └── resnet.py # relatively recent CNN 
|   └── squeezenet.py # highly compressed CNN
|   └── cnn1d.py # classical 1 dimensional CNN
|   └── resnet1d.py # 1 dimensional CNN with skip connections
|   └── resnet1d_v2.py # deeper version of resnet1d
├── toolbox
|   └── optimizer and losses selectors
|   └── logger.py  # keeping track of most results during training and storage to static .html file
|   └── metrics.py # code snippets for computing scores and main values to track
|   └── plotter.py # snippets for plotting and saving plots to disk
|   └── utils.py   # various utility functions
├── commander.py # main file from the project serving for calling all necessary functions for training and testing
├── args.py # parsing all command line arguments for experiments
├── trainer.py # pipelines for training, validation and testing
```


## Launching
Experiments can be launched by calling `commander.py` and a set of input arguments to customize the experiments. You can find the list of available arguments in `args.py` and some default values. Note that not all parameters are mandatory for launching and most of them will be assigned their default value if the user does not modify them.

Here is a typical launch command and some comments:

- `python commander.py --dataset ecg --name ecg_resnet1d_10_optadam_lr1e-3_lrReducePlateau0.5_bsz128 --epochs 20 --num-classes 5 --root-dir ecg_data --arch resnet1d --model-name resnet1d_10 --batch-size 128 --lr 0.001 --scheduler ReduceLROnPlateau --optimizer adam --lr-decay 0.5 --step 15 --workers 4 --tensorboard`
  + this experiment is on the _ecg_ dataset which can be found in `--root-dir/ecg` trained over _resnet1d_10_. It optimizes with _adam_ with initial learning rate (`--lr`) of `1e-3` which is decayed by half whenever the `--scheduler` _ReduceLRonPlateau_ does not see an improvement in the validation accuracy for more than `--step` epochs. Input sequences are of size 187. In addition it saves intermediate results to `--tensorboard`.
  + when debugging you can add the `--short-run` argument which performs training and validation epochs of 10 mini-batches. This allows testing your entire pipeline before launching an experiment
  + if you want to resume a previously paused experiment you can use the `--resume` flag which can continue the training from _best_, _latest_ or a specifically designated epoch.
  + if you want to use your model only for evaluation on the test set, add the `--test` flag. 

Here are some more typical launch commands:

- `python commander.py --dataset ecg --name ecg_cnn1d_3_optadam_lr1e-3_lrReducePlateau0.5_bsz128 --epochs 20 --num-classes 5 --root-dir ecg_data --arch cnn1d --model-name cnn1d_3 --batch-size 128 --lr 0.001 --scheduler ReduceLROnPlateau --optimizer adam --tensorboard`

- `python commander.py --dataset ecg --name ecg_resnet1d_10_optadam_lr1e-3_lrReducePlateau0.5_bsz128 --epochs 20 --num-classes 5 --root-dir ecg_data --arch resnet1d --model-name resnet1d_10 --batch-size 128 --lr 0.001 --scheduler ReduceLROnPlateau --optimizer sgd --tensorboard`


## Output
For each experiment a folder with the same name is created in the folder `root-dir/ecg/runs`
 This folder contains the following items:

```bash
.
├── checkpoints (\*.pth.tar) # models and logs are saved every epoch in .tar files. Non-modulo 5 epochs are then deleted.
├── best model (\*.pth.tar) # the currently best model for the experiment is saved separately
├── config.json  # experiment hyperparameters
├── logger.json  # scores and metrics from all training epochs (loss, learning rate, accuracy,etc.)
├── res  # predictions for each sample from the validation set for every epoch
├── tensorboard  # experiment values saved in tensorboard format
 ```

### Tensorboard
In order the visualize metrics and results in tensorboard you need to launch it separately (from the project directory): `tensorboard --logdir ecg_data/ecg/runs`. You can then access tensorboard in our browser at [localhost:6006](localhost:6006). If you have performed multiple experiments, tensorboard will aggregate them in the same dashboard. To check if any tensorboard runs exist run `tensorboard --inspect --logdir ecg_data/ecg/runs`


## Results


