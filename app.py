import os
from apikeys import openai_api_key, pinecone_api_key

os.environ['OPENAI_API_KEY'] = openai_api_key
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

os.environ['PINECONE_API_KEY'] = pinecone_api_key
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
PINECONE_ENV = 'us-west4-gcp-free'

embed_model = 'text-embedding-ada-002'
 

import openai, langchain, pinecone
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter

data_file = open('../text gpt app/wonderfulwizardoz.txt', 'r', encoding='utf8')

file_content = data_file.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000,
    chunk_overlap = 0,
    length_function = len
)

book_content = text_splitter.create_documents([file_content])

index_name = 'testsearchbook'

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

pinecone.init(
    api_key = PINECONE_API_KEY,
    environment=PINECONE_ENV
)

book_docsearch = Pinecone.from_texts([t.page_content for t in book_content], embeddings, index_name=index_name)

from langchain.chains.question_answering import load_qa_chain

llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

query = 'Who is Dorothy'

doc = book_docsearch.similarity_search(query)

chain = load_qa_chain(llm=llm, chain_type='stuff')

response = chain.run(input_documents = doc, question = query)

print(response)





