# Pharmaceutical Sales Prediction

**Table of content**

- [Pharmaceutical Sales Prediction](#pharmaceutical-sales-prediction)
  - [Overview](#overview)
  - [Requirements](#requirements)
  - [Install](#install)
  - [File](#file)
  - [Data](#data)
  - [Features](#features)
  - [Models](#models)
  - [Metrics](#metrics)
  - [Pages](#pages)
  - [Notebooks](#notebooks)
  - [Scripts](#scripts)
  - [Test](#test)

## Overview

Rossmann operates over 3,000 drug stores in 7 European countries. Currently, Rossmann store managers are tasked with predicting their daily sales for up to six weeks in advance. Store sales are influenced by many factors, including promotions, competition, school and state holidays, seasonality, and locality. With thousands of individual managers predicting sales based on their unique circumstances, the accuracy of results can be quite varied.

## Requirements
Python 3.5 and above, Pip and MYSQL
## Install
```
git clone https://github.com/eandualem/PharmaceuticalSalesPrediction
cd PharmaceuticalSalesPrediction
pip install -r requirements.txt
```
## File
  - train.csv: historical data including Sales and Customers
  - test.csv: historical data excluding Sales  and Customers
  - sample_submission.csv: a sample submission file in the correct format
  - store.csv: supplemental information about the stores

## Data
  - that represents a(Store, Date) duple within the test set
  - Store: a unique Id for each store
  - Sales: the turnover for any given day(this is what you are predicting)
  - Customers: the number of customers on a given day
  - Open: an indicator for whether the store was open: 0 = closed, 1 = open
  - StateHoliday: indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None
  - SchoolHoliday: indicates if the(Store, Date) was affected by the closure of public schools
  - StoreType: differentiates between 4 different store models: a, b, c, d
  - Assortment: describes an assortment level: a = basic, b = extra, c = extended. Read more about assortment here
  - CompetitionDistance: distance in meters to the nearest competitor store
  - CompetitionOpenSince[Month / Year]: gives the approximate year and month of the time the nearest competitor was opened
  - Promo: indicates whether a store is running a promo on that day
  - Promo2: Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
  - Promo2Since[Year / Week]: describes the year and calendar week when the store started participating in Promo2
  - PromoInterval: describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store

## Features
- Generated features are stored inside features folder

## Models
- Models are stored inside models folder

## Metrics
- Models evaluation metrics are stored inside metrics folder

## Pages
- Pages in streamlit app are stored inside pages folder

## Notebooks
  - `1.0 preprocessing.ipynb`: clean data by filling null values and removing outliers
  - `2.0 exploration.ipynb`: data exploration with lots of insights
  - `3.0 prediction.ipynb`: A machine learning approach to implement multi variable sales prediction and Time Series Sales Prediction using deep learning

## Scripts
  - `file_handler`: helper class for reading and writing to files
  - `create_features`: helper class for reading and writing to files
  - `train_model`: this class is used for training all of the machine learning model
  - `evaluate_model`: class for calculates evaluation metrics for a give model using actual data
  - `time_series_model`: this class is used for training deep learning model
  - `app.py`: Streamlit's home page

## Test
  - Holds test files for `test_df_cleaner.py` and `test_file_handler.py`.
