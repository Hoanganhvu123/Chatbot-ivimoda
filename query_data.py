"""Create a ChatVectorDBChain for question/answering."""
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT,
                                                     QA_PROMPT)
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.vectorstores.base import VectorStore
from langchain.chat_models import ChatOpenAI


def get_chain(
    vectorstore: VectorStore, question_handler, stream_handler, tracing: bool = False
) -> ConversationalRetrievalChain:
    """Create a ConversationalRetrievalChain for question/answering."""
    # Construct a ConversationalRetrievalChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    manager = AsyncCallbackManager([])
    question_manager = AsyncCallbackManager([question_handler])
    stream_manager = AsyncCallbackManager([stream_handler])
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
        question_manager.add_handler(tracer)
        stream_manager.add_handler(tracer)

    question_gen_llm = OpenAI(
        # model_name="text-curie-001",
        model_name="gpt-3.5-turbo",
        # model_name="gpt-3.5-turbo",
        # model_name="text-babbage-001",
        # model_name="text-davinci-003",
        # model_name="curie",
        temperature=0.0,
        verbose=True,
        callback_manager=question_manager,
        max_tokens=800
    )
    streaming_llm = OpenAI(
        # model_name="text-curie-001",
        model_name="gpt-3.5-turbo",
        # model_name="text-babbage-001",
        # model_name="text-davinci-003",
        # model_name="curie",
        streaming=True, 
        callback_manager=stream_manager,
        verbose=True,
        temperature=0.0,
        max_tokens=1000
    )

    # model = ChatOpenAI(
    #     model_name='gpt-3.5-turbo',
    #     # model_name="text-babbage-001",
    #     # model_name="text-davinci-002",
    #     # model_name="curie",
    #     streaming=True,
    #     callback_manager=stream_manager,
    #     verbose=True,
    #     temperature=0.1,
    #     max_tokens=1300
    # )

    question_generator = LLMChain(
        # llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT, callback_manager=manager
        llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT, callback_manager=manager
    )
    doc_chain = load_qa_chain(
        # streaming_llm, chain_type="stuff", prompt=QA_PROMPT, callback_manager=manager
        # streaming_llm, chain_type="refine", callback_manager=manager
        streaming_llm, chain_type="stuff", callback_manager=manager
    )
    #.from_llm(OpenAI(streaming=True, callbacks=[MyCustomHandler()], temperature=0.1, max_tokens=-1), gmo_retriever, return_source_documents=True, max_tokens_limit=4000)
    # qa = ConversationalRetrievalChain.from_llm(model, vectorstore, return_source_documents=True, max_tokens_limit=1300, callback_manager=manager, chain_type="refine")
    qa = ConversationalRetrievalChain(
        retriever=vectorstore,
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        callback_manager=manager,
        return_source_documents=True,
        max_tokens_limit=800
    )
    return qa
