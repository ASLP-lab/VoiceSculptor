from pymilvus import MilvusClient, Collection
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
app = FastAPI()

class RagQuery(BaseModel):
    text: str

# TODO: 替换为对应的embedding模型
qwen3emb = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

# TODO: 替换为对应的Milvus数据库文件
client = MilvusClient("milvus_audio_final_demo.db")
client.load_collection("audio_collection")
collection_info = client.describe_collection("audio_collection")

def get_embedding(text: str):
    output = qwen3emb.encode([text])
    return output[0].tolist()

@app.post("/rag/search")
def rag_search(req: RagQuery):
    text = req.text
    query_emb = get_embedding(text)
    results = client.search(
        collection_name="audio_collection",
        data=[query_emb],
        limit=1,
        search_params={"metric_type": "IP", "params": {}},
        output_fields=["text"]
    )
    if not results[0]:  # 检查是否有结果
        print("⚠️  未找到匹配项")
        return None
    else:
        hit = results[0][0]
        print(f"✅ 找到匹配项，相似度: {hit['distance']:.4f}")
        print("Retrieved text:")
        print(hit["entity"]["text"])
        return hit["entity"]["text"]
