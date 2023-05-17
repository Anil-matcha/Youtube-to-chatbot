from langchain.document_loaders import YoutubeLoader
from langchain.indexes import VectorstoreIndexCreator
loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=O5xeyoRL95U&ab_channel=LexFridman", add_video_info=False)
index = VectorstoreIndexCreator().from_loaders([loader])
query = "What did the president say about Ketanji Brown Jackson"
index.query(query)