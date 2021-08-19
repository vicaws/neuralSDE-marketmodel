# neuralSDE-marketmodel

Python modules and jupyter notebook examples for the paper 
[Arbitrage-free Neural-SDE Market Models](https://arxiv.org/abs/2105.11053).

## Code

### Installation of pre-requisites

It is recommended to create a new environment and install pre-requisite
packages. All packages in requirements.txt are compatible with Python 3.6.9.

>```
>pip install -r requirements.txt
>```

In addition, in this repository we also include our proprietary python package 
[arbitragerepair](https://github.com/vicaws/arbitragerepair) for calculating 
static arbitrage boundaries. To know more about the methodology, please refer to
our paper 
[Detecting and Repairing Arbitrage in Traded Option Prices](https://www.tandfonline.com/eprint/YQKWWNED73HSPVGC5ZBE/full?target=10.1080/1350486X.2020.1846573).

### Usage

The following notebook/script examples show common usage of the code:

1. [Paper](notebook/paper.ipynb): this notebook loads the offline-trained 
   checkpoint files for all the neural network models that are used in the 
   published paper.

2. [Factor decoding](notebook/factors.ipynb): this notebook shows a few exploratory examples of decoding 
   factors from call option prices simulated from a Heston-SLV model.

3. [Model training](main.py): this python script gives the complete flow of 
   codes from decoding factors to training models and simulating option prices.

## Citation
>```
>@misc{nsdemm2021,
>    author = {Samuel N. Cohen, Christoph Reisinger, Sheng Wang},  
>    title = {neuralSDE-marketmodel},
>    year = {2021},
>    howpublished = {\url{https://github.com/vicaws/neuralSDE-marketmodel}},
>    note = {commit XXXX}
>}
>```
