# expense-profile-clustering
clustering different expense profiles.

# Data
Data is purely artificial.

# Data-prep
In order to prep the data for classification, I first have to prepare the data. Since it is purely artificial and somewhat randomly generated, I had to first group them together and then later use the grouped data as input to a classifier. In order to do that, I performed the following steps:

1. Normalize the data
2. Use KMeans @k=10, iter=15

# Technologies used
1. R for analyzing data and experimenting
2. Apache Spark (pyspark) for data prep and creating the actual models and their package
