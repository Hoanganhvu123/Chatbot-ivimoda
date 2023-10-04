import pickle
import os

from langchain.document_loaders import BSHTMLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

from langchain.schema import Document
from pydantic import BaseModel

from itertools import chain
import os

os.environ["OPENAI_API_KEY"] = "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"

from langchain.document_loaders import BSHTMLLoader

urls = [
  "/content/Đầm lụa lệch vai _ IVY moda.html",
  "/content/Đầm lụa lệch vai _ IVY moda.html",
  "/content/Váy_Đầm Nữ Đẹp Cao Cấp Hàng Hiệu - Áo Đầm Sang Trọng _ IVY moda.html"
]

loader = BSHTMLLoader("./templates/Váy_Đầm Nữ Đẹp Cao Cấp Hàng Hiệu - Áo Đầm Sang Trọng _ IVY moda.html")
docs1 = loader.load()
docs1

#Tạo 1 dữ liệu mới gắn thêm đường link của ảnh

class Document(BaseModel):
  page_content: str
  metadata: dict

  @classmethod
  def from_dict(cls, doc_dict):
    return cls(**doc_dict)

images = {
  'Đầm lụa lệch vai Vàng mustard': 'https://pubcdn.ivymoda.com/files/product/thumab/400/2022/09/29/4ff370c71ce09af5f4368bb301e2be94.jpg',
  'Đấm lụa lệch vai Hồng tím': 'https://pubcdn.ivymoda.com/files/product/thumab/1400/2022/09/14/5624820345fdb53d885650ec307b540b.jpg'
}

docs3 = []

for key, value in images.items():
  doc_dict = {
    'page_content': key,
    'metadata': {'img_url': value}
  }

  doc = Document.from_dict(doc_dict)

  docs3.append(doc)

#Gộp 2 dữ liệu với nhau 
docs = chain(docs1, docs3)

#Chia nhỏ dữ liệu 
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(separator='\n',
                                      chunk_size=1000,
                                      chunk_overlap=200)


docs = text_splitter.split_documents(docs)

import pickle
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings


embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
with open("vectorstore.pkl", "wb") as f:
    pickle.dump(vectorstore, f)
def load_retriever():
    with open("vectorstore.pkl", "rb") as f:
        vectorstore = pickle.load(f)
    retriever = VectorStoreRetriever(vectorstore=vectorstore)
    return retriever


