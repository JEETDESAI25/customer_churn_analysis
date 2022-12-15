# Customer Churn Prediction

### Contributors

* [Hetanshee Shah](https://github.com/hetanshee)
* [Jeet Desai](https://github.com/JEETDESAI25)
* [Mustafa Zaki](https://github.com/mustafazaki98)



## What is Constumer Churn

One of the most important parts of running any business is to understand the value of your customers. And in order to survive, or even thrive in your business is to identify customers who are not hesitant to leave your business and turn towards your competitor. Customer churn model aims to identify this and generate a binary value - indicating whether a customer will churn or not. In this paper customer information from an Iranian Telecommunication company was used to predict if a customer will churn or not. The key steps used to implement this is: data preprocessing - where we clean the data from various impurities, model construction - building 3 models that will be used to predict if a customer will churn or not, and lastly, analysis of the data was done in order to mitigate the customer churn rate.

## Data Description

https://archive.ics.uci.edu/ml/datasets/Iranian+Churn+Dataset?TB_iframe=true&width=370.8&height=658.8

Data is about churning telecom customers based on the below features-

| Feature Name           |     Type       | Description                                   | 
| -----------------------|----------------| ----------------------------------------------|
| Call Failures          |  Categorical   | number of call failures.                      |
| Complains              |  Numerical     | binary (0: No complaint, 1: complaint)        |
| Call Failures          |  Categorical   | number of call failures                       |
| Subscription Length    |  Numerical     | total months of subscription                  |
|  Charge Amount         |  Categorical   | 0: lowest amount, 9: highest amount           |
| Seconds of Use         |  Numerical     | total seconds of calls                        |
| Frequency of use       |  Numerical     | total number of calls                         |
| Frequency of SMS       |  Numerical     | total number of text messages                 |
| Distinct Called Numbers|  Numerical     | total number of distinct phone calls          |
| Tariff Plan            |  Categorical   | binary (1: Pay as you go, 2: contractual)     |
| AgeGroup               |  Categorical   | 1: younger age, 5: older age                  |
| Status                 |  Categorical   | binary (1: active, 2: non-active)             |
| Customer Value         |  Numerical     | calculated value of customer                  |
| Churn                  |  Categorical   | binary (1: churn, 0: non-churn) - Class label |

## Project Life Cycle
The following Approaches were used to execute the project Life cycle:

(1) **Data Pre-Processing**
There may be several impurities in the raw data. Data preprocessing aims to get rid of all these impuritties__

 Steps involved in Data prepocessing:
 * Removing Garbage values
 * Removing Null Values
 * Removing Outliers

 (2) **Feature Selection**
 * We used SelectKBest feature selection technique to select the top features to train different multi-classification model

(4) **Model Construction**
* Support Vector Classifier 
* Decision Tree
* Logistic Regression

 (3) **Performance Metrics**
 * To evaluate the performance or quality of the model, different metrics are used, and these metrics are known as performance metrics or evaluation metrics.


