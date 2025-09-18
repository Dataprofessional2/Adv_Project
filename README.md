# AirBnb Price Optimization (Started 17 Sept 2025)
This repository contains projects and case studies related to the Google Data Analytics Certification. It includes data cleaning, analysis, visualization, and insights using different tools. The goal is to apply real-world data analytics techniques to solve business problems and derive actionable insights.

## How Airbnb Seattle Optimizes Pricing for Maximum Revenue
## Table of Contents

- Introduction
- Overview of Airbnb Seattle Dataset
- Business Objective
- Data Cleaning & Preparation
- Data Loading and Inspection
- Handling Missing Values
- Data Cleaning and Formatting
- Exploratory Data Analysis (EDA)
- Price Distribution and Outliers
- Location and Pricing Trends
- Seasonality and Booking Patterns
- Feature Engineering
- Price per Guest
- Proximity to Downtown
- Host Experience Metrics
- Temporal Features
- Modeling Approach
- Baseline Linear Regression
- Advanced Models: Random Forest and XGBoost
- Model Evaluation
- Error Metrics: MAE and RMSE
- Feature Importance Analysis
- Insights and Recommendations
- Pricing Strategies for Hosts
- Balancing Occupancy and Revenue
-Identifying High-Impact Features
- Conclusion






## Project Overview: AirBnb Price Optimization

### Introduction
In this project, we explore how Airbnb hosts in Seattle can optimize pricing strategies to improve revenue and occupancy. Using the open Airbnb Seattle dataset, we analyze patterns in pricing across different property types, locations, and seasons. The goal is to build a predictive model that recommends optimal prices by leveraging available listing attributes and booking trends.


### Overview of [AirBnb Seattle Dataset](https://www.kaggle.com/datasets/airbnb/seattle)
The dataset includes comprehensive information on Airbnb listings in Seattle, such as property features, pricing, availability, and customer reviews. Key files include:
- listings.csv: details on accommodations, amenities, host characteristics, and pricing
- calendar.csv: daily booking availability and prices
- reviews.csv: customer feedback and ratings
  
These datasets allow us to explore how various factors influence pricing decisions and how they can be optimized.


## Business Objective
The primary objective is to predict the ideal listing price for Airbnb properties in Seattle based on features such as location, room type, capacity, and host behavior. By identifying patterns and relationships within the data, we aim to help hosts set competitive yet profitable prices while ensuring higher occupancy rates.



## Data Cleaning & Preparation

#### Data Loading and Inspection
- The first step involves loading the datasets using Python libraries such as pandas and inspecting their structure. We check the shape, column types, and summary statistics to understand the data and identify areas that need cleaning.

#### Handling Missing Values
- Missing data is common in real-world datasets. We explore missing values across features like bathrooms, reviews, or price and decide on imputation techniques or row removal based on their impact on the model.

#### Data Cleaning and Formatting
- We format columns such as price by removing currency symbols, convert dates into datetime formats, and ensure consistency in categorical variables. This step prepares the data for efficient modeling.


```python
# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

![An Image](https://github.com/Dataprofessional2/Adv_Project/blob/main/Screenshot%202025-09-18%20133700.png)


```python
calendar_data=pd.read_csv('D://airbnb_project//datasets//calendar.csv')
calendar_data
```

![An Image](https://github.com/Dataprofessional2/Adv_Project/blob/main/calendar_data.png)


```python
listings_data=pd.read_csv('D://airbnb_project//datasets//listings.csv')
listings_data
reviews_data=pd.read_csv('D://airbnb_project//datasets//reviews.csv')
reviews_data
```

![An Image](https://github.com/Dataprofessional2/Adv_Project/blob/main/Complete_Data.png)
