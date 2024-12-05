# Randomised Signature Timeseries Generation for OMR project 2

## How to get started
First instal the required. Recommendation is to install it in an virtual environment.
First install pytorch via:

    pip install torch

Now proceed to install the rest of the packages:

    pip install numpy seaborn signatory scipy yfinance

For now I only changed the CONDITIONAL code, so run the main.py  in this folder and check if no errors occur.

## Where to configure the code?
All configuration is done in the config.py file. For now the settings are the same as in what they used in the paper (which is different from the default!). To reduce runtime (but at the cost of worse results) for e.g. testing , change GRADIENT_STEPS to a smaller number (250 takes ~10 sec). 

## Adding other examples?
In data.py new data can be generated which can be used in the code. Should be quite straight forward to implement our own examples. 

