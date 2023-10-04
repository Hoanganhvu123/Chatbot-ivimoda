"""Main entrypoint for the app."""
import logging
import pickle
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from langchain.vectorstores import VectorStore

from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from query_data import get_chain
from schemas import ChatResponse
from langchain.vectorstores import FAISS, Chroma
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks.manager import get_openai_callback
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse 
from callback import StreamingLLMCallbackHandler

app = FastAPI()
templates = Jinja2Templates(directory="templates")
vectorstore: Optional[VectorStore] = None
gmo_retriever = None

os.environ["OPENAI_API_KEY"] = "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"

@app.on_event("startup")
async def startup_event():
    # logging.info("loading vectorstore")
    # if not Path("vectorstore.pkl").exists():
    #     raise ValueError("vectorstore.pkl does not exist, please run ingest.py first")
    # with open("vectorstore.pkl", "rb") as f:
    #     global vectorstore
    #     vectorstore = pickle.load(f)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    gmoDB = Chroma(persist_directory=os.path.join('db', 'gmo'), embedding_function=embeddings)
    global gmo_retriever, vectorstore
    gmo_retriever = gmoDB.as_retriever(search_kwargs={"k": 2})
    vectorstore = gmo_retriever


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history = []
    qa_chain = get_chain(vectorstore, question_handler, stream_handler)
    # Use the below line instead of the above line to enable tracing
    # Ensure `langchain-server` is running
    # qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)
    #prompt_ext = """. Language: Vietnamese. Require: You must List the results with each line separately. Please write an introduction for your answer"""
    prompt_ext = """"""
    prompt_suffix = """.You must List the reply with each line separately. Please write an introduction for your answer.Trả lời bằng tiếng Việt hoặc dịch câu trả lời sang Tiếng Việt"""
    flag = 0
    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())
            query = question + prompt_ext + prompt_suffix
            # prompt_ext = ""
            
            
    
            with get_openai_callback() as cb:
                result = await qa_chain.acall(
                    {"question": query, "chat_history": chat_history}
                )
                print(result)
                # Lấy các giá trị từ callback   
                token_used = cb.total_tokens   
                prompt_tokens = cb.prompt_tokens
                completion_tokens = cb.completion_tokens
                total_cost = cb.total_cost
                
                
                
                # In ra các giá trị
                print("token_used " + str(token_used))
                print("prompt_tokens " + str(prompt_tokens))
                print("completion_tokens " + str(completion_tokens))
                print("total_cost " + str(total_cost))
            # print(result)
           
           
           
           
           
           
            chat_history.append((query, result["answer"]))# 

            end_resp = ChatResponse(sender="bot", message="", type="end")
            # end_resp = ChatResponse(sender="bot", message=result_data, type="end")
            
    
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="192.168.1.3", port=9000)
