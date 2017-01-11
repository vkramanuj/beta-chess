# Deep RL for Chess

## Structure

*models/* Contains policy and evaluation models. Policy and eval models both come in a fully connected version with embeddings and a convolutional version. The CNN model is better for "obvious" positions while the fully connected version seems to have better accuracy.

*datasets/* Contains datasets. You can add your own! Place a pgn formatted filed in the datasets/pgn directory and train using `python train --dataset NAME` for file `NAME.pgn`. The datasets/epds directory contain Extended Position Description files for each of these datasets. And the datasets/csvs directory contains a csv version. 

*logs/* Each folder is named by date and time it was created. In each folder are subfolders *checkpoints/* and *summaries/* which contain model checkpoints and logging summaries respectively. Use the command `tensorboard --logdir=FOLDER` where `FOLDER` is your date and time to view these logs and checkpoints in a nice format.

*train_eval.py* Trains the evaluation network. Use `python train_eval.py --help` to get a list of flags.

*train_policy.py* Trains the policy network. Use `python train_policy.py --help` to get a list of flags.

Will add a script to play against a trained model soonish.

## Requirements

- tensorflow (clearly)
- tflearn
- python-chess
- numpy
- pandas
- tqdm