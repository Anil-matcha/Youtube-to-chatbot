from langchain.document_loaders import YoutubeLoader
import scrapetube
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain


channel_id = "UC03sxjXYe4mSLqr5etxOXGA"
videos = scrapetube.get_channel(channel_id)
pages = []
for v in videos:
    videoId = v['videoId']
    url = "https://www.youtube.com/watch?v="+videoId
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
    pages += loader.load_and_split()    
embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(pages, embeddings).as_retriever()    
query = "What is Vadoo ?"
docs = docsearch.get_relevant_documents(query)
chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
output = chain.run(input_documents=docs, question=query)