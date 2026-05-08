from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lower, regexp_replace, split, explode, length


# ============================================================
# BASIC SETTINGS
# ============================================================

DATASET_FILE = "twitter_dataset.csv"


# ============================================================
# SPARK SESSION
# ============================================================

spark = SparkSession.builder \
    .appName("Social Media Sentiment Basic Analysis") \
    .getOrCreate()


# ============================================================
# TASK 1: LOAD DATASET
# ============================================================

print("\n")
print("=" * 70)
print("TASK 1: LOAD DATASET")
print("=" * 70)

df = spark.read.csv(
    DATASET_FILE,
    header=False,
    inferSchema=True,
    quote='"',
    escape='"',
    multiLine=False
)

# Twitter Kaggle Sentiment140 dataset columns
df = df.toDF("target", "id", "date", "flag", "user", "text")

# Sentiment category create karna
df = df.withColumn(
    "category",
    when(col("target") == 0, "Negative")
    .when(col("target") == 4, "Positive")
    .otherwise("Neutral")
)

total_posts = df.count()

print("Dataset loaded successfully.")
print("CSV file name:", DATASET_FILE)
print("Total posts in dataset:", total_posts)

print("\nSample records:")
df.select("target", "category", "user", "text").show(10, truncate=False)


# ============================================================
# TASK 2: COUNT POSTS PER CATEGORY
# ============================================================

print("\n")
print("=" * 70)
print("TASK 2: COUNT POSTS PER CATEGORY")
print("=" * 70)

posts_per_category = df.groupBy("category").count().orderBy("category")

print("Number of posts in each sentiment category:")
posts_per_category.show(truncate=False)


# ============================================================
# TASK 3: FILTER POSITIVE POSTS
# ============================================================

print("\n")
print("=" * 70)
print("TASK 3A: FILTER POSITIVE POSTS")
print("=" * 70)

positive_posts = df.filter(col("category") == "Positive")
positive_count = positive_posts.count()

print("Total positive posts:", positive_count)
print("\nSample positive posts:")
positive_posts.select("category", "text").show(10, truncate=False)


# ============================================================
# TASK 3: FILTER NEGATIVE POSTS
# ============================================================

print("\n")
print("=" * 70)
print("TASK 3B: FILTER NEGATIVE POSTS")
print("=" * 70)

negative_posts = df.filter(col("category") == "Negative")
negative_count = negative_posts.count()

print("Total negative posts:", negative_count)
print("\nSample negative posts:")
negative_posts.select("category", "text").show(10, truncate=False)


# ============================================================
# TASK 4: FIND MOST FREQUENT KEYWORDS
# ============================================================

print("\n")
print("=" * 70)
print("TASK 4: FIND MOST FREQUENT KEYWORDS")
print("=" * 70)

# Text cleaning
clean_df = df.withColumn("clean_text", lower(col("text")))

# URLs remove
clean_df = clean_df.withColumn(
    "clean_text",
    regexp_replace(col("clean_text"), "http\\S+|www\\S+", "")
)

# Mentions remove, example: @username
clean_df = clean_df.withColumn(
    "clean_text",
    regexp_replace(col("clean_text"), "@\\w+", "")
)

# Hashtag symbol remove
clean_df = clean_df.withColumn(
    "clean_text",
    regexp_replace(col("clean_text"), "#", "")
)

# Special characters remove
clean_df = clean_df.withColumn(
    "clean_text",
    regexp_replace(col("clean_text"), "[^a-zA-Z\\s]", "")
)

# Text ko words me convert karna
words_df = clean_df.select(
    explode(split(col("clean_text"), "\\s+")).alias("word")
)

# Common stopwords remove karna
stopwords = [
    "the", "a", "an", "and", "or", "is", "are", "am",
    "i", "you", "he", "she", "it", "we", "they",
    "to", "of", "in", "on", "for", "with", "this",
    "that", "was", "were", "be", "been", "have",
    "has", "had", "my", "your", "me", "so", "but",
    "not", "at", "as", "if", "out", "up", "just",
    "do", "dont", "im", "its", "rt", "get", "got",
    "can", "will", "all", "now", "too", "from"
]

filtered_words = words_df.filter(
    (length(col("word")) > 2) &
    (~col("word").isin(stopwords))
)

most_frequent_keywords = filtered_words.groupBy("word") \
    .count() \
    .orderBy(col("count").desc())

print("Top 20 most frequent keywords:")
most_frequent_keywords.show(20, truncate=False)


# ============================================================
# DELIVERABLE: SUMMARY OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("DELIVERABLE: SUMMARY OUTPUT")
print("=" * 70)

print("Project Name: Social Media Sentiment Dataset Basic Analysis")
print("Tool Used: PySpark")
print("Dataset Used: Twitter Dataset Kaggle")
print("CSV File:", DATASET_FILE)

print("\nSummary:")
print("Total posts:", total_posts)
print("Positive posts:", positive_count)
print("Negative posts:", negative_count)

neutral_count = total_posts - positive_count - negative_count
print("Neutral posts:", neutral_count)

print("\nPosts per category:")
posts_per_category.show(truncate=False)

print("\nTop 20 keywords:")
most_frequent_keywords.show(20, truncate=False)


# ============================================================
# SAVE OUTPUT FILES
# ============================================================

print("\n")
print("=" * 70)
print("SAVING OUTPUT FILES")
print("=" * 70)

posts_per_category.coalesce(1).write.mode("overwrite").csv(
    "output/posts_per_category",
    header=True
)

positive_posts.select("category", "text").coalesce(1).write.mode("overwrite").csv(
    "output/positive_posts",
    header=True
)

negative_posts.select("category", "text").coalesce(1).write.mode("overwrite").csv(
    "output/negative_posts",
    header=True
)

most_frequent_keywords.limit(20).coalesce(1).write.mode("overwrite").csv(
    "output/top_keywords",
    header=True
)

print("Output files saved successfully in output folder.")

print("\n")
print("=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)


# ============================================================
# STOP SPARK
# ============================================================

spark.stop()