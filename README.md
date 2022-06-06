# MICCAI 2019 ABCD Neurocognitive Prediction Challenge

Team: Purdue150  
Submission Date: Mar 24, 2019

If you use this codes in your analysis, please cite:

 - Zou, Y., Jang, I., Reese, T.G., Yao, J., Zhu, W., Rispoli, J.V. (2019). Cortical and Subcortical Contributions to Predicting Intelligence Using 3D ConvNets. In: Pohl, K., Thompson, W., Adeli, E., Linguraru, M. (eds) Adolescent Brain Cognitive Development Neurocognitive Prediction. ABCD-NP 2019. Lecture Notes in Computer Science, vol 11791. Springer, Cham. [[link](https://doi.org/10.1007/978-3-030-31901-4_21)]

## Overview

1. The submission includes the following major codes: 
 - `datagenerator_ROI.py`
 - `training.py`
 - `testing_ROI_part1.py`
 - `testing_ROI_part2.py` 
 - `evaluation.R`

2. The directory that includes these codes should have two directories named: `trainvaliddata` and `testdata`. The two directories includes all T1 data.

3. Major dependencies: 
 - Tensorflow 1.12
 - Python 3.6
 - Keras 2.2.4
 - nibabel 2.3.3
 - R

4. Output: `test_results.csv`

## Example running command in order:

### 1) Training

```
 python training.py X1 13 3
 python training.py X1 71 3
 python training.py X1 77 3
```

### 2) Testing

```
 python testing_ROI_part1.py X1 13 3
 python testing_ROI_part2.py X1 13 3
 python testing_ROI_part1.py X1 71 3
 python testing_ROI_part2.py X1 71 3
 python testing_ROI_part1.py X1 77 3
 python testing_ROI_part2.py X1 77 3
```
### 3) Evaluation

```
 Rscript evaluation.R
```

## All dependencies
```
# Name                    Version                   Build  Channel
attrs                     19.1.0                    <pip>
backcall                  0.1.0                     <pip>
bleach                    3.1.0                     <pip>
bz2file                   0.98                      <pip>
ca-certificates           2019.1.23                     0  
certifi                   2019.3.9                 py36_0  
decorator                 4.4.0                     <pip>
defusedxml                0.5.0                     <pip>
entrypoints               0.3                       <pip>
ipykernel                 5.1.0                     <pip>
ipython                   7.4.0                     <pip>
ipython-genutils          0.2.0                     <pip>
ipywidgets                7.4.2                     <pip>
jedi                      0.13.3                    <pip>
Jinja2                    2.10                      <pip>
jsonschema                3.0.1                     <pip>
jupyter                   1.0.0                     <pip>
jupyter-client            5.2.4                     <pip>
jupyter-console           6.0.0                     <pip>
jupyter-core              4.4.0                     <pip>
Keras                     2.2.4                     <pip>
libedit                   3.1.20181209         hc058e9b_0  
libffi                    3.2.1                hd88cf55_4  
libgcc-ng                 8.2.0                hdf63c60_1  
libstdcxx-ng              8.2.0                hdf63c60_1  
MarkupSafe                1.1.1                     <pip>
mistune                   0.8.4                     <pip>
nbconvert                 5.4.1                     <pip>
nbformat                  4.4.0                     <pip>
ncurses                   6.1                  he6710b0_1  
nibabel                   2.3.3                     <pip>
notebook                  5.7.6                     <pip>
openssl                   1.1.1b               h7b6447c_1  
pandas                    0.24.2                    <pip>
pandocfilters             1.4.2                     <pip>
parso                     0.3.4                     <pip>
pexpect                   4.6.0                     <pip>
pickleshare               0.7.5                     <pip>
pip                       19.0.3                   py36_0  
prometheus-client         0.6.0                     <pip>
prompt-toolkit            2.0.9                     <pip>
ptyprocess                0.6.0                     <pip>
Pygments                  2.3.1                     <pip>
pyrsistent                0.14.11                   <pip>
python                    3.6.8                h0371630_0  
python-dateutil           2.8.0                     <pip>
pytz                      2018.9                    <pip>
PyYAML                    5.1                       <pip>
pyzmq                     18.0.1                    <pip>
qtconsole                 4.4.3                     <pip>
readline                  7.0                  h7b6447c_5  
scikit-learn              0.20.3                    <pip>
scipy                     1.2.1                     <pip>
Send2Trash                1.5.0                     <pip>
setuptools                40.8.0                   py36_0  
sklearn                   0.0                       <pip>
sqlite                    3.27.2               h7b6447c_0  
terminado                 0.8.1                     <pip>
testpath                  0.4.2                     <pip>
tk                        8.6.8                hbc83047_0  
tornado                   6.0.1                     <pip>
traitlets                 4.3.2                     <pip>
wcwidth                   0.1.7                     <pip>
webencodings              0.5.1                     <pip>
wheel                     0.33.1                   py36_0  
widgetsnbextension        3.4.2                     <pip>
xz                        5.2.4                h14c3975_4  
zlib                      1.2.11               h7b6447c_3
```
