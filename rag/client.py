import requests

# TODO: 下面替换为部署RAG服务地址
RAG_URL = "http://127.0.0.1:8000/rag/search"  

def rag_infer(text: str, timeout=2.0):
    try:
        resp = requests.post(
            RAG_URL,
            json={"text": text},
            timeout=timeout
        )
        resp.raise_for_status()
        data = resp.json()
        return data 
    except Exception as e:
        print(f"[RAG] 请求失败: {e}")
        return None

print(rag_infer("以均匀从容的节奏进行专业客观的新闻播报。"))