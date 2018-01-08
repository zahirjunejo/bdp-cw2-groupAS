from pyspark import SparkConf, SparkContext
from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array
from math import sqrt
import xml.etree.ElementTree as ET
import operator

conf = SparkConf().setAppName("My App")

sc = SparkContext(conf=conf)

def converPostToPair(post):
    tuple = ET.fromstring(post)
    key = tuple.get('OwnerUserId')
    favoriteCount = float(tuple.get('FavoriteCount', '0')) #default favorite value is 0.
    viewCount = float(tuple.get('ViewCount', '0')) #default view count value is 0.
    value = (float(1), favoriteCount, viewCount) # (post occurance count, fav. count)
    return (key, value)

def createUserPair(user):
    tuple = ET.fromstring(user)
    key = tuple.get('Id')
    upvotes = float(tuple.get('UpVotes'))
    reputation = float(tuple.get('Reputation'))
    value = (upvotes, reputation)
    return (key, value)

#Get the user data
user = sc.textFile('/data/stackoverflow/Users').filter(lambda x : 'Id' in x);
userPair = user.filter(lambda x : all(ord(c) < 128 for c in x)).map(createUserPair)

#Get the post data
posts = sc.textFile('/data/stackoverflow/Posts').filter(lambda x : 'Id' in x)
postPair = posts.filter(lambda x : all(ord(c) < 128 for c in x)).map(converPostToPair).reduceByKey(lambda x,y : tuple(map(operator.add, x, y)))

joinedData = userPair.join(postPair)

# Gives out : (total upvotes of user / total posts of user, reputation, total favoriteCount, total fav. count / total posts, total view count / total posts)
finalData = joinedData.map(lambda x : (x[1][0][0] / x[1][1][0], x[1][0][1], x[1][1][1], x[1][1][1] / x[1][1][0], x[1][1][2] / x[1][1][0]))

clusters = KMeans.train(finalData, k, maxIterations=100, initializationMode="random")

def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

errorList = []

for k in range(3, 20):
    clusters = KMeans.train(finalData, k, maxIterations=100, initializationMode="random")
    WSSSE = finalData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
    errorList.append((k, WSSSE))

Logger = sc._jvm.org.apache.log4j.Logger
myLogger = Logger.getLogger(__name__)
