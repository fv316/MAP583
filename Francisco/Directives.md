# project tasks and directives

## TODO in order of importance
- Variable length RNN inputs. Use masking e.g. Word-2-Vector class example. 
- Use downsampling of normal class to achieve better performance.
- Create a RNN model using GRU or LSTM.
- Visualize the ROC curve and F1 scores.
- Try to use data augmentation and visualization.
- Research using generative adversarial networks.
- Apply new loss functions.


## already done
- Create a simple 1D-CNN model.
- Make ECG data loader.
- Transform project directory for the ECG data.

## current results
### 1D-CNN
Train - Loss: 0.1189 Acc: 0.9692
Test - Loss: 0.1372 Acc: 0.9664

## example cli commands
- python commander.py --dataset ecg --name ecg_trial1.5_bsz128 --epochs 2 --num-classes 5 --root-dir ecg_data --arch cnn1d --model-name cnn1d_3 --batch-size 128 --tensorboard

- python commander.py --dataset ecg --name ecg_trial1.5_bsz128 --epochs 2 --num-classes 5 --root-dir ecg_data --arch cnn1d --model-name cnn1d_3 --batch-size 128 --tensorboard --short-run

## directives for presentation
- chaque equipe a 10 minutes de presentations en incluant les questions. Vous assistez a toutes les presentations des autres projets et pourrez poser des questions. Toute l'equipe doit etre presente pour la presentation.

- il n'y a pas de rapport a rendre mais le code doit etre accessible sur un github dont vous nous enverrez l'adresse par mail. Ne pas mettre les datasets sur le github mais toute la pipeline doit etre disponible (telechargement des donnees, pretraitement...)

- pour votre presentation, il faut imperativement:

+ presenter brievement la tache de learning.

+ presenter le dataset utilise avec qq statistiques.

+ decrire l'architecture du reseau ou des reseaux en donnant les references explicites.

+ donner qq resultats obtenus et les comparer a une baseline, aux resultats dans les papiers... Donner des resultats quantitatifs (avec les metriques propres a votre tache) et qualitatifs. Si les resultats vous semblent mauvais, donner des exemples qui vous permettent de voir les causes possibles et donc les ameliorations si vous aviez plus de temps...

+ prevoir qq minutes pour les questions

A priori, le plus simple est pour chaque equipe de presenter avec son laptop, il faut une prise VGA ou HDMI.

