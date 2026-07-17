import logging # logging details, observability
from fastapi import FastAPI # 把 RAG 包装成 web API
# from 一个包 import 一个特定的函数, 不引入整个包
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv # dotenv: 使python能够读取API KEY
import uuid #生成 unique id
import os # operating system
import datetime

from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGQueryResult, RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc
load_dotenv()# load the environment variables inside of this .env file

# creating some of our clients

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger('uvicorn'),
    is_production=False, # development mode, not online
    serializer=inngest.PydanticSerializer()
    # 任务之间要传数据(比如把 "这个 PDF 的 ID" 传给下一步),数据得序列化成能存能传的格式(通常是 JSON)。默认 serializer 只认 dict、str、int 这种普通类型;PydanticSerializer 让它能直接吃 Pydantic
    # 模型——你定义一个 class Document(BaseModel): ...,可以整个对象丢给 Inngest,它自动转成 JSON 传过去,下一步再自动还原成对象。省事、还带类型检查。

)
#定义一个交给Inngest的后台任务

@inngest_client.create_function( # 说明这个function是交给inngest的一个function
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def  rag_ingest_pdf(ctx: inngest.Context): # async表示这是一个异步函数, 能一边等慢活、一边不占着资源
    # here we define 2 steps, so two functions inside of this rag_ingest_pdf function

    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source:id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads=[{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()

@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGQueryResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question =  ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content=(
        "Use the following context to answer the question.\n\n"
        f"Context: \n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    adapter = ai.openai.Adapter(
        auth_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )

    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You answer questions using only the provided content."},
                {"role": "user", "content": user_content}
            ]
        }
    )

    answer = res["choices"][0]["message"]["content"].strip()
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}
# cuz it's ingest function, we have the ability of having observability
# create server

app = FastAPI()

# set up Endpoints

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])

# database client: allow us to create the database, load data, search data