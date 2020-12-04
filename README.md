# Machine Learning Photovoltaics

Supplementary information for [Effect of increasing the descriptor set on machine learning prediction of small-molecule-based organic solar cells](https://doi.org/10.1021/acs.chemmater.0c02325), by _Z-W Zhao, M del Cueto, Y Geng and A Troisi_.

The **MLPhotovoltaics.py** program allows to predict the power conversion efficiency (and other target properties) with different machine learning (ML) algorithms, given a set of descriptors of several donor/acceptor pairs of organic solar cells, compiled in **database.csv**. The code allows to use different ML algorithms, descriptors, and optimize all corresponding hyperparameters. More information is provided in main text of the article and the electronic supporting information.

---

## Getting Started

### Prerequisites

The necessary packages (with the tested versions with Python 3.8.5) are specified in the file requirements.txt. These packages can be installed with pip:
```
pip3 install -r requirements.txt
```

### Usage

All input parameters are specified in file: _inputMLPhotovoltaics.inp_. Input options in this file are separated in different groups:

- Parallelization: only relevant when trying to use differential evolution algorithm to optimize hyperparameters
- Verbose options: allows some flexibility for how much information to print to standard output and log file
- Data base options: allows to select how many donor/acceptor pairs are used, as well as which descriptors are considered
- Learning Curve options: allows to print learning curves with different CV methods, to visualize model learning performance
- Output prediction csv: allows to print the actual and predicted target properties values of the test points
- Machine Learning Algorithm options: allows to select what ML algorithm is used (whether kNN, KRR or SVR), as well as cross validation method, hyperparameters etc.

To execute program, make sure that you have all necessary python packages installed, and that all necessary files are present: the database (**database.csv**), input file (**inputMLPhotovoltaics.inp**) and program (**MLPhotovoltaics.py**). Finally, simply run:

```
python MLPhotovoltaics.py
```

### Example inputs
The provided input files in folders Fig6a, Fig6b and Fig6c contain the corresponding options to reproduce all three panels of Figure 6 of the article: 5D<sub>ph</sub>+2D<sub>fp</sub>, 7-D-D<sub>ph</sub>+2D<sub>fp</sub> and 9D<sub>ph(5basic+2D+2A)</sub>+2D<sub>fp</sub> descriptors respectively, with kNN (k=3) and hyperparameter values already optimized.

---

### Authors

Zhi-Wen Zhao, [Marcos del Cueto](https://github.com/marcosdelcueto), Y Geng and A Troisi
