# LEOPARD

**Missing view comp<ins>l</ins>etion for multi-tim<ins>e</ins>point <ins>o</ins>mics data via re<ins>p</ins>resentation disent<ins>a</ins>nglement and tempo<ins>r</ins>al knowle<ins>d</ins>ge transfer**

Longitudinal multi-view omics data offer unique insights into the temporal dynamics of individual-level physiology, which provides opportunities to advance personalized healthcare. However, the common occurrence of incomplete views makes extrapolation tasks difficult, and there is a lack of tailored methods for this critical issue. Here, we introduce LEOPARD, an innovative approach specifically designed to complete missing views in multi-timepoint omics data. By disentangling longitudinal omics data into content and temporal representations, LEOPARD transfers temporal knowledge to omics-specific content, thereby completing missing views. Compared to conventional imputation methods, LEOPARD yields the most robust results across three benchmark datasets. LEOPARD-imputed data also achieve the best agreement with the observed data in our analyses for age-associated metabolites detection and chronic kidney disease prediction. Our work takes the first step toward a generalized treatment of missing views in longitudinal omics data, enabling comprehensive exploration of temporal dynamics and providing valuable insights into personalized healthcare.

*Thank you for checking our LEOPARD!*

*Any questions regarding LEOPARD please drop an email to the zookeeper Siyu Han (siyu.han@tum.de) or post it to [issues](https://github.com/HAN-Siyu/LEOPARD/issues).*


## Habitat

Specific environment settings are required to run LEOPARD. The following packages are used in our stduy:

- python: 3.79
- numpy: 1.23.5
- pandas: 1.4.4
- scikit-learn: 1.0.2
- pytorch: 1.11.10
- pytorch_lightning: 1.6.4
- tensorboard: 2.10.0
- cuda (if use GPU): 11.3


## How to Train Your LEOPARD

The architecture of LEOPARD is fully customizable and supports data of two views. LEOPARD is better to run in an interactive editors (Jupyter Notebook, PyCharm, Spyder, etc). Instruction for how to train a LEOPARD is provided in the juper-notebook file "manual.ipynb".


## Script Files

- manual.ipynb: a brief instruction (jupyter-notebook, with Python kernel) for training LEOPARD
- plot.ipynb: a jupyter-notebook (with R kernel) for reproducing the plots
- example.py: examples in Python to reproduce LEOPARD's imputation results of the MGH COVID dataset
- src/: scripts for building LEOPARD
  - data.py: dataset preparation
  - layers.py: basic layers used to build LEOPARD
  - model.py: class of LEOPARD architecture
  - train.py: LightningModule of LEOPARD training
  - utils.py: some utility functions for data processing
- data/: data for reproducing our results and figures
  - MGH_COVID: benchmark dataset constructed from the [MGH COVID study](http://dx.doi.org/10.17632/nf853r8xsj)
  - MGH_COVID_imputed: imputation results of the test set of the MGH COVID dataset, obtained from LEOPARD, mice, missForest, and cGAN (under obsNum = 0, 25, 50, and 100)
  - plotData: data used to generate the plots in our study


## Cite This Work

If you use the code or data in this repository, please cite:

```bibtex
@article{han2023missing,
  title={Missing view completion for multi-timepoints omics data via representation disentanglement and temporal knowledge transfer},
  author={Han, S and Yu, S and Shi, M and Harada, M and Ge, J and Lin, J and Prehn, C and Petrera, A and Li, Y and Sam, F and others},
  journal={biorxiv preprint doi:10.1101/2023.09.26.559302},
  year={2023}
}
```


## Our Big Cat Zoo

- [TIGER](https://github.com/HAN-Siyu/TIGER): technical variation elimination for metabolomics data using ensemble learning architecture
- [LION](https://github.com/HAN-Siyu/LION): an integrated R package for effective prediction of lncRNA/ncRNA–protein interaction
- LEOPARD (this work): missing view completion for multi-timepoint omics data via representation disentanglement and temporal knowledge transfer
