# AirBnb Price Optimization (Started 17 Sept 2025)
This repository contains projects and case studies based on real world business problems. It includes data cleaning, analysis, visualization, and insights using different tools. The goal is to apply real-world data analytics techniques to solve business problems and derive actionable insights.

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
# Inspecting Data
calendar_data.head(5)
calendar_data.info()
```

![An Image](https://github.com/Dataprofessional2/Adv_Project/blob/main/Inspecting_Data.png)




```python
# Calendar DataSet Cleaning
# Drop duplicates
calendar_data = calendar_data.drop_duplicates()

# Clean and convert price column
calendar_data['price'] = (
    calendar_data['price']
    .replace('[\$,]', '', regex=True)  # remove $ and ,
    .astype(float))

# Keep only available days
calendar_data = calendar_data[calendar_data['available'] == 't']

# Fill missing prices with mean
mean_price = calendar_data['price'].mean()
calendar_data['price'] = calendar_data['price'].fillna(mean_price)

# Convert date column to datetime
calendar_data['date'] = pd.to_datetime(calendar_data['date'])

# Extract useful time features
calendar_data['month'] = calendar_data['date'].dt.month
calendar_data['weekday'] = calendar_data['date'].dt.day_name()

# Check results
print(calendar_data.info())
print("Missing prices after cleaning:", calendar_data['price'].isna().sum())
```
![An Image](https://github.com/Dataprofessional2/Adv_Project/blob/main/Calendar_DataSet_Cleaning.png)





# Listings DataSet Cleansing 
```python
# Importing & Inspecting Listings DataSet
listings_data=pd.read_csv('D://airbnb_project//datasets//listings.csv')
listings_data.head(5)
```

![An Image](https://github.com/Dataprofessional2/Adv_Project/blob/main/Importing_Inspecting_Listings_Data.png)
![Cols_Image](https://github.com/Dataprofessional2/Adv_Project/blob/main/Cols_In_Listings_Data.png)





```python
# Selecting only the useful columns into a new dataframe
new_listings_columns = [
    'id', 'host_id', 'host_is_superhost', 
    'neighbourhood_cleansed', 'city', 'state', 'zipcode', 
    'latitude', 'longitude',
    'property_type', 'room_type', 'accommodates', 'bathrooms', 
    'bedrooms', 'beds', 'bed_type', 'amenities',
    'price', 'weekly_price', 'monthly_price', 'cleaning_fee', 'extra_people',
    'minimum_nights', 'maximum_nights', 
    'availability_30', 'availability_60', 'availability_90', 'availability_365',
    'number_of_reviews', 'reviews_per_month', 
    'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 
    'review_scores_checkin', 'review_scores_communication', 
    'review_scores_location', 'review_scores_value',
    'instant_bookable', 'cancellation_policy'
]

# Make a new dataframe
listings_clean = listings_data[new_listings_columns].copy()

# Quick check
listings_clean.info()
```

![Cols Retention](https://github.com/Dataprofessional2/Adv_Project/blob/main/Retaining_Columns.png)

```python
listings_clean.info()
```
![Image](https://github.com/Dataprofessional2/Adv_Project/blob/main/DataSet%20Overview.png)
![Image](https://github.com/Dataprofessional2/Adv_Project/blob/main/Complete%20Overview.png)


### Listings DataSet Cleaning
```python
# Basic Cleaning

listings_clean = listings_clean.drop_duplicates()

# Handle missing values
listings_clean['bathrooms'] = listings_clean['bathrooms'].fillna(listings_clean['bathrooms'].median())
listings_clean['bedrooms'] = listings_clean['bedrooms'].fillna(listings_clean['bedrooms'].median())
listings_clean['beds'] = listings_clean['beds'].fillna(listings_clean['beds'].median())
listings_clean['review_scores_rating'] = listings_clean['review_scores_rating'].fillna(listings_clean['review_scores_rating'].median())
listings_clean['reviews_per_month'] = listings_clean['reviews_per_month'].fillna(0)  # no reviews = 0

# Convert categorical yes/no to binary
binary_map = {'t': 1, 'f': 0, 'yes': 1, 'no': 0}
listings_clean['host_is_superhost'] = listings_clean['host_is_superhost'].map(binary_map)
listings_clean['instant_bookable'] = listings_clean['instant_bookable'].map(binary_map)


# Convert price-related columns

price_columns = ['price', 'weekly_price', 'monthly_price', 'cleaning_fee', 'extra_people']
for col in price_columns:
    if col in listings_clean.columns:
        listings_clean[col] = listings_clean[col].replace('[\$,]', '', regex=True).astype(float)


# Outlier Removal (Price)
# -----------------------------
q1 = listings_clean['price'].quantile(0.25)
q3 = listings_clean['price'].quantile(0.75)
iqr_listings = q3 - q1

lower_limit_listings = q1 - 1.5 * iqr_listings
upper_limit_listings = q3 + 1.5 * iqr_listings

print("Lower limit for price:", lower_limit_listings)
print("Upper limit for price:", upper_limit_listings)

print("Before:", listings_clean.shape)

listings_clean = listings_clean[
    (listings_clean['price'] >= lower_limit_listings) &
    (listings_clean['price'] <= upper_limit_listings)
]

print("After :", listings_clean.shape)

# Quick Check
print(listings_clean.head())
```
![Image](https://github.com/Dataprofessional2/Adv_Project/blob/main/DataSet_Cleaning.png)
















## Exploratory Data Analysis on Calendar DataSet to look for basic trends

```python
#Average Daily Price Over Time in $
calendar_data.groupby('date')['price'].mean().plot(figsize=(12,6), title="Average Daily Price Over Time in $")
```
![Image](https://github.com/Dataprofessional2/Adv_Project/blob/main/Basic_EDA.png)

```python
# Group by weekday and calculate mean price
calendar_data['month'] = calendar_data['date'].dt.month
calendar_data.groupby('month')['price'].mean()
```
![Image](https://github.com/Dataprofessional2/Adv_Project/blob/main/Basic_eda2.png)

```python
import matplotlib.pyplot as plt
# Group by weekday and calculate mean price
mean_price_mos = calendar_data.groupby('month')['price'].mean()
# Bar plot
plt.bar(mean_price_mos.index, mean_price_mos.values)
plt.xlabel("Month")
plt.ylabel("Average Price")
plt.title("Average Price by Month in $")
plt.ylim(0)   # y-axis starts from 0
plt.show()
```
![Image](https://github.com/Dataprofessional2/Adv_Project/blob/main/basic_eda3.png)


```python
# Group by weekday and calculate mean price
calendar_data['weekday'] = calendar_data['date'].dt.day_name()
calendar_data.groupby('weekday')['price'].mean().sort_values()
```
![Image](https://github.com/Dataprofessional2/Adv_Project/blob/main/eda4.png)

```python
import matplotlib.pyplot as plt
# Group by weekday and calculate mean price
meanofprice = calendar_data.groupby('weekday')['price'].mean() # Bar plot
plt.bar(meanofprice.index, meanofprice.values)
plt.xlabel("Weekdays")
plt.ylabel("Average Price in $")
plt.title("Average Price by Weekday")
plt.ylim(0) 
plt.show()
```
![Image](https://github.com/Dataprofessional2/Adv_Project/blob/main/eda5.png)








### Exploratory Data Analysis (EDA) on Listings Data 
- Problem 1: What are the features/facilities/ammenities of a property that affect its price?

Our first sub-problem was to focus on the physical features and facilities of the property itself. We wanted to see if there were any common features among the highly priced listings. We mainly focused on the listing's room type, the property type, number of bedrooms and common ammenities.
```python
#Count of Listings by Room Type
plt.figure(figsize=(12,6))
sns.countplot(data=listings_clean, x='room_type', order=listings_clean['room_type'].value_counts().index)
plt.title('Count of Listings by Room Type')
plt.xlabel('Room Type')
plt.ylabel('Number of Listings')
plt.show()
```
![image](https://github.com/Dataprofessional2/Adv_Project/blob/main/viz1.png)

```python
#Average Price by Room Type
import seaborn as sns
plt.figure(figsize=(8,5))
sns.barplot(data=listings_clean, x='room_type', y='price', 
            order=listings_clean.groupby('room_type')['price'].mean().sort_values(ascending=False).index)
plt.title('Average Price by Room Type')
plt.xlabel('Room Type')
plt.ylabel('Average Price ($)')
plt.show()
```
![image](https://github.com/Dataprofessional2/Adv_Project/blob/main/viz2.png)

```python
top_property_types = listings_clean['property_type'].value_counts().nlargest(10).index
plt.figure(figsize=(12,6))
sns.barplot(data=listings_clean[listings_clean['property_type'].isin(top_property_types)], 
            x='property_type', y='price', order=top_property_types)
plt.title('Average Price of Top 10 Property Types')
plt.xlabel('Property Type')
plt.ylabel('Average Price ($)')
plt.xticks(rotation=45)
plt.show()
```
![image](https://github.com/Dataprofessional2/Adv_Project/blob/main/viz3.png)


```python
import matplotlib.pyplot as plt
import seaborn as sns
import ast

# 5. Top Amenities (Without Counter)

# Convert amenities string to list
amenities_list = listings_clean['amenities'].dropna().apply(lambda x: ast.literal_eval(x) if x.startswith('[') else x.strip('{}').split(','))

# Flatten the list of amenities
all_amenities = []
for sublist in amenities_list:
    for amenity in sublist:
        all_amenities.append(amenity.strip().replace('"','').replace("'",""))

# Create a DataFrame from the list
amenities_df = pd.DataFrame(all_amenities, columns=['Amenity'])

# Count occurrences of each amenity
top_amenities_df = amenities_df['Amenity'].value_counts().head(20).reset_index()
top_amenities_df.columns = ['Amenity', 'Count']

# Plot top 20 amenities
plt.figure(figsize=(12,6))
sns.barplot(data=top_amenities_df, x='Count', y='Amenity')
plt.title('Top 20 Amenities in Listings')
plt.xlabel('Number of Listings')
plt.ylabel('Amenity')
plt.show()
```
![image](https://github.com/Dataprofessional2/Adv_Project/blob/main/viz4.png)
