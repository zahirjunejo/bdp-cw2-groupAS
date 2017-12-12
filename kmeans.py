from pyspark import SparkConf, SparkContext
import testpoints

conf = (SparkConf()
         .setAppName("KMEAN"))
sc = SparkContext(conf = conf)

Logger= sc._jvm.org.apache.log4j.Logger
logger = Logger.getLogger(__name__)

K = 3
MAX_ITERATIONS=100

points = sc.parallelize(testpoints.makePoints())

curr_centroids = points.takeSample(False,K)

# all points are assigned to index 0 center intially
curr_points_assignment = points.map(lambda a: (0,a)).cache()

def sum_points(a,b):
    #logger.info("\n\n!!!!!\n a: {0} \n!!!!!!!! b: {1} \n".format(a,b))
    return [a + b for  (a,b) in zip(a,b)]

for i in range (1,MAX_ITERATIONS):
    #E step
    curr_points_assignment.map(lambda a : (min_center(a[1], curr_centroids), a[0]))

    logger.info("\n\n!!!!!\n {0} \n!!!!!!!!\n".format(curr_points_assignment.collect()))
    #M step
    # mean is not associative, hence we need to run the counts seperately and the sum
    # for each cluster using the aggregatebyK functionality of spark
    #init_vals = (0,[0 for i in range(K)])

    #curr_centroids = curr_points_assignment.aggregateByKey(init_vals, \
     #   lambda a,b: (a[0]+1,sum_points(a[1],b[1])), \
     #   lambda a,b: (a[0]+b[0],sum_points(a[1],b[1]))).collect()
        #.map(lambda a : a[1]/a[0]).collect()
    logger.info("\n\n!!!!!\n {0} \n!!!!!!!!\n".format(curr_centroids))


def distance_metric(x1,x2):
    # manhattan, dim unknown
    dim_pairs = zip(x1,x2)
    sum = 0
    for (dx1,dx2) in dim_pairs:
        sum += abs(dx1 - dx2)
    return sum;

def min_center(p, curr_centroids):
    logger.info("\n\n!!!!!\n p: {0} \n!!!!!!!! curr centroid: {1} \n".format(p,curr_centroids))
    
    min_dist= float("inf")
    min_index = -1
    for i,c in enumerate(curr_centroids):
        temp_dist = distance_metric(p,c) 
        if temp_dist< min_dist:
            min_index = i
            min_dist
    return min_index