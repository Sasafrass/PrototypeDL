# FACT-AI
Private Github repository for the course Fairness, Accountability, Confidentiality and Transparency in AI at the University of Amsterdam. By Albert Harkema, Anna Langedijk, Christiaan van der Vlist, and Hinrik Snær

# Based on
https://github.com/OscarcarLi/PrototypeDL

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