# expense-profile-predictor
expense-profile predictor using machine learning.

# Data
Data is purely artificial.

# Data-prep
In order to prep the data for classification, I first have to prepare the data. Since it is purely artificial and somewhat randomly generated, I had to first group them together and then later use the grouped data as input to a classifier. In order to do that, I performed the following steps:

1. Normalize the data
2. Use KMeans @k=10, iter=15

# Creating the Classifier
For the classifier, I split the KMeans results for training and testing and then trained it without much data prep aside from splitting the data. I used Random Forest algorithm for the Classifier.

# Technologies used
1. R for analyzing data and experimenting
2. Apache Spark (pyspark) for data prep and creating the actual models and their packages
3. IBM Watson Machine Learning for productizing the models to the cloud
