from pyspark import SparkConf, SparkContext
import testpoints
import itertools

conf = (SparkConf()
         .setAppName("KMEAN"))
sc = SparkContext(conf = conf)

Logger= sc._jvm.org.apache.log4j.Logger
logger = Logger.getLogger(__name__)


K = 3
MAX_ITERATIONS=100
THRESHOLD=0.01

# HELPERS
def distance_metric(x1,x2):
    # manhattan, dim unknown
    dim_pairs = zip(x1,x2)
    sum = 0
    for (dx1,dx2) in dim_pairs:
        sum += abs(dx1 - dx2)
    return sum;

def min_center(p, curr_centroids):

    min_dist= float("inf")
    min_index = -1
    for i,c in enumerate(curr_centroids):
        temp_dist = distance_metric(p,c) 
        if temp_dist< min_dist:
            min_index = i
            min_dist = temp_dist
    return min_index

def sum_points(a,b):
    return [a + b for  (a,b) in zip(a,b)]

def div_vector(count_sum_centroids):
    # caution this is a python map..
    return map(lambda centroid_val: float(centroid_val)/ float(count_sum_centroids[0]),count_sum_centroids[1])

def extract_centers(centers):
    centroid_arr = [None]*K
    for elem in centers:
        centroid_arr[elem[0]] = elem[1]
    return centroid_arr

def below_threshold(c1s,c2s):
    return all([ distance_metric(c1,c2) < K for c1,c2 in zip(c1s,c2s)])

# MAIN CODE:
points = sc.parallelize(testpoints.makePoints())

curr_centroids = points.takeSample(False,K)

# all points are assigned to index 0 center intially
curr_points_assignment = points.map(lambda a: (0,a))

# for debug/illustration purposes we store the history of centroids

all_centroids = []

for i in range (1,MAX_ITERATIONS):
    #E step
    Estep = curr_points_assignment.map(lambda a : (min_center(a[1],curr_centroids), a[1]))
    #M step
    #   mean is not associative, hence we need to run the counts seperately and the sum
    # for each cluster using the aggregatebyK functionality of spark
    init_accum = (0,[0 for i in range(K)])

    # first lambda is inter-parition func, partial sums and counts
    # second lambda merge partial sums and counts.

    sum_counts_centroids = Estep.aggregateByKey(init_accum, \
        lambda accum, new_elem: (accum[0]+1, sum_points(accum[1],new_elem)), \
        lambda accum1,accum2: (accum1[0]+accum2[0], sum_points(accum1[1],accum2[1])))

    curr_centroids = extract_centers(sum_counts_centroids.map(lambda count_sum_centroids : \
                                                              (count_sum_centroids[0] , \
                                                div_vector(count_sum_centroids[1]))).collect())


    if(len(all_centroids) > 1 and below_threshold(curr_centroids,all_centroids[-1])):
        logger.info("bailing early due to close center {0}".format(curr_centroids))
        break
    all_centroids.append(curr_centroids)

logger.info("\n\n!!!!!\n {0} \n!!!!!!!!\n".format(all_centroids))
