# Machine Learning Photovoltaics

Supplementary information for **Effect of increasing the descriptor set on machine learning prediction of small-molecule-based organic solar cells**, by _Z-W Zhao, M del Cueto, Y Geng and A Troisi_.

The **MLPhotovoltaics.py** program allows to predict the power conversion efficiency (and other target properties) with different machine learning (ML) algorithms, given a set of descriptors of several donor/acceptor pairs of organic solar cells, compiled in **database.csv**. The code allows to use different ML algorithms, descriptors, and optimize all corresponding hyperparameters. More information is provided in main text of the article and the electronic supporting information.

---

## Getting Started

### Prerequisites

The following packages are used:

- re
- sys
- ast
- time
- math
- numpy
- scipy
- pandas
- sklearn
- functools
- matplotlib

### Usage

All input parameters are specified in file: _inputMLPhotovoltaics.inp_. Input options in this file are separated in different groups:

- Parallelization: only relevant when trying to use differential evolution algorithm to optimize hyperparameters
- Verbose options: allows some flexibility for how much information to print to standard output and log file
- Data base options: allows to select how many donor/acceptor pairs are used, as well as which descriptors are considered
- Machine Learning Algorithm options: allows to select what ML algorithm is used (whether kNN, KRR or SVR), as well as cross validation method, hyperparameters etc.

To execute program, make sure that you have all necessary python packages installed, and that all necessary files are present: the database (_database.csv_), input file (_inputMLPhotovoltaics.inp_) and program (_MLPhotovoltaics.py_). Finally, simply run:

```
./MLPhotovoltaics.py
```

The provided input file  in the folder Fig6a already has the corresponding options to reproduce Figure 6A of the article: 5D<sub>ph</sub>+2D<sub>fp</sub> descriptors, with kNN (k=3), for hyperparameter values already optimized.

---

### Contributors

Zhi-Wen Zhao and Marcos del Cueto
