#clustering

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
import matplotlib.pyplot as plt

sc = SparkContext('local')
spark = SparkSession(sc)


# Loads data.
path = "clusteringds.csv"
df = spark.read.csv(path)
dataset = df.select(df._c0.cast('int'), df._c1.cast('float'),df._c2.cast('float'),df._c3.cast('float'),df._c4.cast('float'),df._c5.cast('float'),df._c6.cast('float'),df._c7.cast('float'),df._c8.cast('float'),df._c9.cast('float'))

# Trains a k-means model.
kmeans = KMeans().setK(3).setSeed(43).setFeaturesCol("features").setPredictionCol("prediction").setMaxIter(15).setInitSteps(3).setInitMode("random")

assembler = VectorAssembler(
    inputCols=["_c1","_c2","_c3","_c4","_c5","_c6","_c7","_c8","_c9"],
    outputCol="features")

#model = kmeans.fit(dataset)
pipeline = Pipeline().setStages([assembler, kmeans])
model = pipeline.fit(dataset)


# Make predictions
predictions = model.transform(dataset)

# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()   

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

x = predictions.select("_c0").rdd.flatMap(lambda x: x).collect()
y = predictions.select("_c0").rdd.flatMap(lambda x: x).collect()
cs = predictions.select("prediction").rdd.flatMap(lambda x: x).collect()



plt.scatter(x,y,c=cs)
plt.show()
#y
#z
#plt.scatter