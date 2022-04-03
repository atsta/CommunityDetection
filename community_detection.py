from graphframes import GraphFrame
from graphframes.lib import AggregateMessages as AM
from pyspark.sql import SparkSession
from graphframes import GraphFrame
from graphframes.lib import AggregateMessages as AM
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, functions as sqlfunctions, types
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import lit

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

# UDF for creating a community data type.
def new_community(community, id):
    return {"id": id, "community": community}
community_type = types.StructType([types.StructField("id", types.StringType()), types.StructField("community", types.IntegerType())])
new_community_udf = F.udf(new_community, community_type)

#initial community of each node is the node itself
vertices = vertices.withColumn("init_community", vertices["id"].cast(IntegerType()))
vertices = vertices.withColumn("community", new_community_udf(vertices["init_community"], vertices["id"])).drop("init_community")

cached_vertices = AM.getCachedDataFrame(vertices)

#display input graph
g = GraphFrame(cached_vertices, edges)
g.vertices.show()
g.edges.show()
g.degrees.show()

#get the smallest between two integers
def get_min(oldcommunity, newcommunity):
    return oldcommunity if(oldcommunity < newcommunity) else newcommunity
get_min_udf = F.udf(get_min, community_type)

#get the smallest most common community
def get_new_community(communities):
    communities_df = spark.createDataFrame(communities, ["id", "community"])
    for community in communities_df:
        print(community)
    communities_df.orderBy("community")
    new_community = communities_df[0]
    new_community_count = 0
    max_count = 0
    max_count_community = new_community
    for community in communities_df:
        if community.community == new_community.community:
            new_community_count+=1
        else:
            new_community = community
            new_community_count = 1
        if new_community_count > max_count:
                max_count = new_community_count
                max_count_community = new_community
    print("-----------")
    return {"id": max_count_community.id, "rating": max_count_community.community}
get_new_community_udf = F.udf(get_new_community, community_type)

superstep = 1
while(1):
    superstep = superstep + 1

    aggregates = g.aggregateMessages(F.collect_set(AM.msg).alias("agg"), sendToDst=AM.src["community"])
    res = aggregates.withColumn("new_community", get_new_community_udf("agg")).drop("agg")
    new_vertices = g.vertices.join(res, on="id", how="left_outer") \
                    .withColumnRenamed("community", "old_community") \
                    .withColumn("previous_community", F.col("old_community")) \
                    .withColumn("community", get_min_udf(F.col("old_community"), F.col("new_community"))) \
                    .drop("new_community").drop("old_community")

    cached_new_vertices = AM.getCachedDataFrame(new_vertices)
    g = GraphFrame(cached_new_vertices, g.edges)
    g.vertices.show()

    if superstep > 40: 
        break
