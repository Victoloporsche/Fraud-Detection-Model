# Fraud-Detection-Model

This repository provides an implementation to detecting fraudlent Bussiness Transaction from a wide range of operations.

#Installation

This implementation is written with Python version 3.6 with the listed packages in the requirements.txt file

1) Clone this repository with git clone https://github.com/Victoloporsche/Fraud-Detection-Model.git
2) With Virtual Environent, use : 
    a) pip install virtualenv
    b) cd path-to-the-cloned-repository
    c) virtualenv myenv
    d) source myenv/bin/activate
    e) pip install -r requirements.txt
3) With Conda Environment, use:
  a) cd path-to-the-cloned-repository
  b) conda create --name myenv
  c) source activate myenv
  d) pip install -r requirements.txt

# Running the Implementation:

1) data_information.py : This module provides a detailed information about the dataset
2) preprocess_data.py: This module preprocesses the data 
3) preprocess_data_strategy.py: This module implements different strategies to preprocess the data
4) optimization.py: This module searches for the best hyperparameters with different classification models
5) main.py: This modules brings all the modules from 1-4 above together.
6) example.ipynb: This uses the modules created to detect fraudlent business transactions.

More modifications and commits would be made to this repository from time to time. Kindly reach out if you have any questions or improvements.
