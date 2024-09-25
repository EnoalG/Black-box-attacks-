# Black-Box Classifier Attack Comparison

This repository contains scripts to compare the performance of three black-box attacks on classifiers: **SurFree**, **CGBA**, and **GeoDA**. The comparison is conducted on two datasets: **ImageNet** and **MNIST**.
The goal is to evaluate how the gradint estimation can improve the performance of the attacks. 

## Overview

In the field of adversarial machine learning, black-box attacks are used to generate adversarial examples without direct access to the internal model parameters. This project aims to evaluate and compare three black-box attacks in terms of their effectiveness and performance.  

## Attacks

- **SurFree**
- **CGBA**
- **GeoDA**

## Requirements

To run the code, you'll need to install the required package using:

```bash
pip install -r requirements.txt
```
# Getting started

Two scripts are provided to run the experiments on different datasets:

1. **ImageNet**: Run the following command to execute the experiment on ImageNet:

```bash
python script_imagenet.py
```
   This script generates adversarial examples for a pre-trained ImageNet classifier and saves the results in .npy format.

2. **MNIST**: Run the following command for MNIST:
```bash
python script_imagenet.py
```
   This script generates adversarial examples for the MNIST_gpu classifier and saves the results in .npy format.



