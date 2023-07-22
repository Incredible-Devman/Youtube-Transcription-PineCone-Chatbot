import pinecone
import pandas as pd
import openai
import os

from dotenv import load_dotenv
load_dotenv()

pinecone.init(api_key=os.getenv("pinecone_key"), environment=os.getenv("pinecone_env"))
print(os.getenv("pinecone_key"), os.getenv("pinecone_env"))
# pinecone.create_index(
#     "youtube-app", 
#     dimension=1536, 
#     metric="cosine", 
#     pod_type="p1"
# )

print(pinecone.list_indexes())
index = pinecone.Index("youtube-app")
print(index.describe_index_stats())

df = pd.read_csv("history.csv")
print(df.head())
# openai.api_key = ""


def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )

    return response['data'][0]['embedding']


def addData(index,url, title,context):
    my_id = index.describe_index_stats()['total_vector_count']

    chunkInfo = (str(my_id),
                 get_embedding(context),
                 {'video_url': url, 'title':title,'context':context})

    index.upsert(vectors=[chunkInfo])


for indexx, row in df.iterrows():
    addData(index,row["url"],row["title"],row["content"])


print("Completed")