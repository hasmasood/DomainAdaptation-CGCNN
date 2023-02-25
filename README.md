# DomainAdaptation-CGCNN
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
### [Data wrangling](https://github.com/hasmasood/DomainAdaptation-CGCNN/blob/master/01-Data%20preparation/DataPrep.ipynb)
A number of processing steps are applied to transfom data into required schema as an ETL pipeline. The raw data is hosted on S3 bucket, so ```boto3``` needs to be imported in the dev environment. Material crystals and DFT band gaps were retrieved from Materials project database with ``` pymatgen ``` API query. The processed data to be used by machine learning model is loaded back to S3 bucket, and JSON dumps were stored locally.
### [EDA](https://github.com/hasmasood/DomainAdaptation-CGCNN/blob/master/01-Data%20preparation/DataPlots.ipynb)
Exploratory analysis to get high-level insights on distribution and types of materials included in datasets.
### [Model development](https://github.com/hasmasood/DomainAdaptation-CGCNN/blob/master/02-MLdev/Protocol.ipynb)
The base model was built on CGCNN framework on DS1 (circa 20,000 samples with inaccurate labels sourced from community database of Materials Project). The base model was then then modified by domain adaptation with DS2 (about 500 samples having accurate labels). An additional model was developed as a control, which was trained from scratch on DS2 with no knowledge transfer from the base model. Performance evaluation was conducted using DS3, which has no overlap with other two datasets used in model development. Layer freezing and fine-tuning tests were also performed during the domain adaptation process, but not included in the notebook. The files [cgcnn_train_bg.py](https://github.com/hasmasood/DomainAdaptation-CGCNN/blob/master/02-MLdev/cgcnn_train_bg.py) and [cgcnn_validation_bg.py](https://github.com/hasmasood/DomainAdaptation-CGCNN/blob/master/02-MLdev/cgcnn_validation_bg.py) contain training and validation scripts for CGCNN. 
### [Insights](https://github.com/hasmasood/DomainAdaptation-CGCNN/blob/master/03-Insights/Insights.ipynb)
The final convolution layer with learnt coefficients from previous steps was projected on a 2-D plane to capture trends and patterns. 