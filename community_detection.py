from graphframes import GraphFrame
from graphframes.lib import AggregateMessages as AM
from pyspark.sql import SparkSession
from graphframes import GraphFrame
from graphframes.lib import AggregateMessages as AM
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, functions as sqlfunctions, types
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import lit, col

spark = SparkSession.builder.appName('community-detection-problem').getOrCreate()

vertices = spark.createDataFrame([('1',), 
                                  ('2',),
                                  ('3',),
                                  ('4',), 
                                  ('5',),
                                  ('6',),
                                  ('7',)],
                                  ['id'])

edges = spark.createDataFrame([('1', '2'), 
                                ('2', '1'),
                                ('1', '3'),
                                ('3', '1'),
                                ('2', '3'), 
                                ('3', '2'),
                                ('1', '4'),
                                ('4', '1'),
                                ('4', '5'),
                                ('5', '4'),
                                ('5', '6'),
                                ('6', '5'),
                                ('5', '7'),
                                ('7', '5'),
                                ('6', '7'), 
                                ('7', '6')],
                                ['src', 'dst'])

# create a community data type
def new_community(community, id):
    return {"id": id, "community": community}
community_type = types.StructType([types.StructField("id", types.StringType()), types.StructField("community", types.IntegerType())])
new_community_udf = F.udf(new_community, community_type)

# initial community of each node is the node itself
vertices = vertices.withColumn("init_community", vertices["id"].cast(IntegerType()))
vertices = vertices.withColumn("community", new_community_udf(vertices["init_community"], vertices["id"])).drop("init_community")

# display input graph
cached_vertices = AM.getCachedDataFrame(vertices)
g = GraphFrame(cached_vertices, edges)
new_vertices = g.vertices.join(g.degrees, on="id", how="left_outer")
cached_vertices = AM.getCachedDataFrame(new_vertices)
g = GraphFrame(cached_vertices, edges)
g.vertices.show()

# get the smallest community by value, keep previous id for one neighbor case
def get_min(current_community, new_community, degree):
    if (degree > 2):
        return {"id": current_community.id, "community": new_community.community} 
    return {"id": current_community.id, "community": current_community.community} if(current_community.community < new_community.community) else {"id": current_community.id, "community": new_community.community} 
get_min_udf = F.udf(get_min, community_type)

# check if community has changed 
def check_changes(oldcommunity, community):
    return False if oldcommunity == community else True
check_changes_udf = F.udf(check_changes, types.BooleanType())

# get the smallest most common community
def get_new_community(communities):
    communities.sort(key=lambda x: x[1])
    new_community = communities[0]
    new_community_count = 0
    max_count = 0
    max_count_community = new_community
    for community in communities:
        if community.community == new_community.community:
            new_community_count+=1
        else:
            new_community = community
            new_community_count = 1
        if new_community_count > max_count:
                max_count = new_community_count
                max_count_community = new_community
    return {"id": max_count_community.id, "community": max_count_community.community}
get_new_community_udf = F.udf(get_new_community, community_type)

# get community value
def get_community(community):
    return community.community

def print_solution(vertices, edges):
    print("------------ Community Detection Solution ------------")
    vertices = vertices.withColumn("community", get_community(vertices["community"])).drop("community_changed").drop("degree")
    cached_new_vertices = AM.getCachedDataFrame(vertices)
    g = GraphFrame(cached_new_vertices, edges)
    g.vertices.show()

it_count = 1
while(1):
    # maximum iterations = 40
    if it_count > 40: 
        print_solution(new_vertices, g.edges)
        break

    aggregates = g.aggregateMessages(F.collect_set(AM.msg).alias("agg"), sendToDst=AM.src["community"])
    res = aggregates.withColumn("new_community", get_new_community_udf("agg")).drop("agg")
    new_vertices = g.vertices.join(res, on="id", how="left_outer") \
                    .withColumnRenamed("community", "old_community") \
                    .withColumn("community", get_min_udf(F.col("old_community"), F.col("new_community"), F.col("degree"))) \
                    .withColumn("community_changed", check_changes_udf(F.col("old_community"), F.col("community"))) \
                    .drop("new_community").drop("old_community")

    # id, community, community_changed, degree
    cached_new_vertices = AM.getCachedDataFrame(new_vertices)
    g = GraphFrame(cached_new_vertices, g.edges)
    g.vertices.show()
    it_count += 1

    # if at least one vertice has acquired a new community, continue 
    if(bool(new_vertices.filter(new_vertices.community_changed.contains(True)).collect())):
        continue

    print_solution(new_vertices, g.edges)
    break