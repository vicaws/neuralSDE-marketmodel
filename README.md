# neuralSDE-marketmodel

[![DOI](https://zenodo.org/badge/397935869.svg)](https://zenodo.org/badge/latestdoi/397935869)

Python modules and jupyter notebook examples for the following papers: 
1. [Arbitrage-free Neural-SDE Market Models](https://arxiv.org/abs/2105.11053).
2. [Estimating Risks of Option Books using Neural-SDE Market Models](https://arxiv.org/abs/2202.07148).
3. [Hedging Option Books using Neural-SDE Market Models](https://arxiv.org/abs/2205.15991) (code to be added).
 

## Code

### Installation of pre-requisites

It is recommended to create a new environment and install pre-requisite
packages.

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

1. [Paper I](notebook/paper.ipynb): this notebook loads the offline-trained 
   checkpoint files for all the neural network models that are used in the 
   published paper [Arbitrage-free Neural-SDE Market Models](https://arxiv.org/abs/2105.11053).
2. [Paper II](notebook/optionmetrics_VaR.ipynb): this notebook loads the 
   processed historical price data for EUROSTOXX 50 and DAX index options, the 
   pre-decoded factors, and offline-trained checkpoint files for some of the 
   neural network models that are used in the published paper
   [Estimating Risks of Option Books using Neural-SDE Market Models](https://arxiv.org/abs/2202.07148).

3. [Factor decoding](notebook/factors.ipynb): this notebook shows a few exploratory examples of decoding 
   factors from call option prices simulated from a Heston-SLV model.

4. [Model training](main.py): this python script gives the complete flow of 
   codes from decoding factors to training models and simulating option prices 
   (from synthetic call option prices simulated from a Heston-SLV model).

## Citation
>```
>@misc{nsdemm2022,
>    author = {Samuel N. Cohen and Christoph Reisinger and Sheng Wang},  
>    title = {neuralSDE-marketmodel},
>    publisher = {GitHub},
>    journal = {GitHub repository},
>    year = {2022},
>    howpublished = {\url{https://github.com/vicaws/neuralSDE-marketmodel}},
>    note = {DOI: 10.5281/zenodo.5337522}
>}
>```
