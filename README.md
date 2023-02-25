# CGCNN-DomainAdaptation
This repo contains code and artefacts for domain adaptation-based CGCNN for predictions of experimental band gaps from crystal lattice. 

![image](https://github.com/hasmasood/DomainAdaptation-CGCNN/blob/master/images/cgcnn.jpg)

The original concept of CGCNN is leveraged from https://github.com/txie-93/cgcnn, with further modifications made to support domain adaptation.

## Technology stack
[Windows Subsytem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install)

[Amazon Web Services (AWS)](https://aws.amazon.com/)

[Visual Studio Code](https://code.visualstudio.com/)

## Development environment
The binaries and packages used for this project are included in environment.yml file
```
conda env create -f environment.yml
```
## Contents
Foundational pieces of the project are in form of jupyter notebooks (documentation is provided where necessary), along with additional python scripts helping the execution. The workflow can be traced under a logical sequence.
### [Data wrangling](https://github.com/hasmasood/DomainAdaptation-CGCNN/blob/main/01-Data%20preparation/DataPrep.ipynb)
A number of processing steps are applied to transfom data into required schema as an ETL pipeline. The raw data is hosted on S3 bucket, so ```boto3``` needs to be imported in the dev environment. Material crystals and DFT band gaps were retrieved from Materials project database with ``` pymatgen ``` API query. The processed data to be used by machine learning model is loaded back to S3 bucket, and JSON dumps were stored locally.
### [EDA](https://github.com/hasmasood/DomainAdaptation-CGCNN/blob/main/01-Data%20preparation/DataPlots.ipynb)
Exploratory analysis to get high-level insights on distribution and types of materials included in datasets.