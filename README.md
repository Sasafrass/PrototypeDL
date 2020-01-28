# FACT-AI
Private Github repository for the course Fairness, Accountability, Confidentiality and Transparency in AI at the University of Amsterdam. By Albert Harkema, Anna Langedijk, Christiaan van der Vlist, and Hinrik Snær

# Based on
https://github.com/OscarcarLi/PrototypeDL

# Instructions
First, activate the correct environment (more about this below):
```
conda env create -f environment_prototype.yml
source activate prototype 
```

Then, run the code either from the IPython notebook, or by running `run.py`: 
```
python run.py [--seed <int>] [--dir <directory name>] [--hier]
```

# Environment
I based this environment on the environment provided by the DL course and added jupyter for easy ipython notebooks.

## Setup the environment
```
conda env create -f environment_prototype.yml
conda activate prototype
```

## Changing the environment if you need more packages
```
conda install -n prototype <your-packages>
conda env export > environment_prototype.yml
```

## Other things about the environment
I hate it! Lisa doesnt have all the up to date packages! So I removed the ipython notebook dependencies and just include the training for this guy.
If you want to use it on lisa, use `environment_prototype2.yml` instead.
This includes an older version of `pillow`, see https://github.com/python-pillow/Pillow/issues/4130. 

