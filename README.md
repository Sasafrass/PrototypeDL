# FACT-AI: Towards Hierarchical Explanation
Private Github repository for the course Fairness, Accountability, Confidentiality and Transparency in AI at the University of Amsterdam. 

## Authors
Albert Harkema () (albert.harkema@student.uva.nl)
Anna Langedijk (12297402) (annalangedijk@gmail.com)
Christiaan van der Vlist () (christiaan.vandervlist@student.uva.nl)
Hinrik Snær (12675326) (hinriksnaer@gmail.com)

## Based on
Our implementation is based on the tensorflow code in https://github.com/OscarcarLi/PrototypeDL.
It extends the original implementation by using hierarchical prototypes.

# Instructions
First, (create and then) activate the correct environment:
```
[conda env create -f environment_prototype.yml]
source activate prototype 
```

Then, run the code either from the IPython notebook, or by running `run.py`: 
```
python run.py [--seed <int>] [--dir <directory name>] [--hier]
```
This will run the code with default parameters/seed for reproduction.

# Environment
I based this environment on the environment provided by the DL course and added jupyter for easy IPython notebooks.

## Setup the environment
```
conda env create -f environment_prototype.yml
conda activate prototype
```

## Help! if environment does not work on LISA:
I removed the ipython notebook dependencies in a second environment `environment_prototype2.yml`. 
If there is trouble using Lisa, use `environment_prototype2.yml` instead.
This includes an older version of `pillow`, see https://github.com/python-pillow/Pillow/issues/4130. 

