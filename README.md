# Code for the paper: "Hierarchical Average Reward Option Critic"

This repository was implemented using Python 2.7.12 and extends the option critic policy gradient repository by https://github.com/mattriemer.

**Tested Environments:** The code implements a generic interface to the OpenAI Gym for Atari style games. The --env flag of each start script can be used to control the environment. "FlappyBird-v0" implements our Multi-task Flappy Bird experiments. We have also tested our code and reported experiments on a number of Atari environments: ["Alien-v0", "Amidar-v0", "Asterix-v0", "Asteroids-v0", "Berzerk-v0", "Boxing-v0", "DoubleDunk-v0", "Frostbite-v0", "IceHockey-v0", "Jamesbond-v0", "Krull-v0", "Seaquest-v0", "SpaceInvaders-v0", "Tennis-v0", "Tutankham-v0"].

## Basic setup:

As a first step to get up and running, clone this git repository and navigate into the root directory of your local version of the repository. To get started, please install the requirements inside your environment.

If you don't have an environment, we recommend that you create one (using [conda](http://anaconda.org)). The following instructions will guide you:

Install `conda` and type

```conda create --name optioncritic python=2.7```

This will create a conda environment (an isolated workplace) in which we can install the right versions of the software. Then, activate the environment:

```source activate optioncritic```

or

```conda activate optioncritic```

Within the `optioncritic` environment, install PyTorch, Cython, and the PyGame Learning Environment using as follows:

```conda install pytorch=0.4.0 -c pytorch```

```conda install cython```

```git clone https://github.com/ntasfi/PyGame-Learning-Environment.git```

```cd PyGame-Learning-Environment/```

```pip install --user -e .```

```cd ..```

and then install the rest of the requirements using the following command:

```pip install --user -r requirements.txt```


## Running standard option-critic with an actor-critic style policy over options:

Flappy Bird:

```PYTHONPATH='.' python2.7 start_scripts/oc.py --env FlappyBird-v0 --workers 16 --save-model-dir ocsave/ --log-dir ocsave/ --options 8 --delib 0.1```

Boxing:

```PYTHONPATH='.' python2.7 start_scripts/oc.py --env Boxing-v0 --workers 16 --save-model-dir ocsave/ --log-dir ocsave/ --options 8 --delib 0.3```

## Running the proposed option-critic policy gradient theorem:

Flappy Bird:

```PYTHONPATH='.' python2.7 start_scripts/ocpg.py --env FlappyBird-v0 --workers 16 --save-model-dir ocpgsave/ --log-dir ocpgsave/ --options 8 --delib 0.1```

Boxing:

```PYTHONPATH='.' python2.7 start_scripts/ocpg.py --env Boxing-v0 --workers 16 --save-model-dir ocpgsave/ --log-dir ocpgsave/ --options 8 --delib 0.3```
