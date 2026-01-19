import os
import torch
from vllm import LLM
from tqdm import tqdm
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

# TODO:这里填写对应的文本embedding模型路径
MODEL_PATH = ""

def load_model_and_get_dim(model_path, gpu_num=8):
    """加载模型并自动检测输出维度，支持多卡"""
    print(f"Loading Qwen3-Embedding model on {gpu_num} GPUs...")
    
    llm = LLM(
        model=model_path,
        task="embed",
        tensor_parallel_size=gpu_num,
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
    )
    test_emb = llm.embed(["测试"])
    dim = len(test_emb[0].outputs.embedding)
    return llm, dim

model, VECTOR_DIM = load_model_and_get_dim(MODEL_PATH)

def get_embedding(text: str):
    """生成文本embedding"""
    if not text or not isinstance(text, str):
        raise ValueError(f"无效文本输入: {text}")
    output = model.embed([text])
    return  torch.tensor(output[0].outputs.embedding).tolist()

DB_PATH = "milvus_audio_demo.db"
COLLECTION_NAME = "audio_collection"

class AudioRAGMilvus:
    def __init__(self, db_path, collection_name):
        self.client = MilvusClient(db_path)
        self.collection_name = collection_name
        self.vector_dim = None
        
    def create_collection(self, dim):
        """创建集合（如果已存在则删除重建）"""
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
        
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=36, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="utt", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
        ]
        schema = CollectionSchema(fields, "Audio RAG Collection")
        
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            consistency_level="Bounded"
        )
        self.client.create_index(
            collection_name=self.collection_name,
            index_params=[{"field_name":"vector","index_name":"vector_index", "metric_type": "IP", "params": {"nlist": 1024}}] #"index_type": "FLAT",
        )
        self.vector_dim = dim
        
    def insert_batch(self, data_list, batch_size=1000):
        """分批插入数据，避免RPC超限"""
        if not data_list:
            print("数据列表为空")
            return
        
        total = len(data_list)
        print(f"准备插入 {total} 条记录，每批次 {batch_size} 条...")
        
        self.client.load_collection(self.collection_name)
        
        for i in tqdm(range(0, total, batch_size), desc="Milvus插入进度"):
            batch = data_list[i:i + batch_size]
            for record in batch:
                if len(record["vector"]) != self.vector_dim:
                    raise ValueError(f"向量维度不匹配: {len(record['vector'])} != {self.vector_dim}")
            
            try:
                self.client.insert(self.collection_name, batch)
            except Exception as e:
                print(f"批次 {i//batch_size + 1} 失败: {e}")
                continue
        
        self.client.flush(self.collection_name)
        stats = self.client.get_collection_stats(self.collection_name)
        print(f"插入完成！集合当前记录数: {stats['row_count']}")
        
    def search(self, query_text, top_k=1):
        """搜索最匹配的音频"""
        query_emb = get_embedding(query_text)
        
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_emb],
            limit=top_k,
            output_fields=["text", "utt"],
            search_params={"metric_type": "IP", "params": {"nprobe": 16}}
        )
        if not results[0]:
            print("⚠️  未找到匹配结果")
            return None
        hit = results[0][0]
        return {
            "score": hit["distance"],
            "text": hit["entity"]["text"],
            "utt": hit["entity"]["utt"]
        }


def load_audio_metadata(scp_path):
    """加载音频元数据，验证文件存在性"""
    utt2texts = {}
    utt2wav_files = {}
    missing_files = []
    
    print(f"加载SCP文件: {scp_path}")
    
    with open(scp_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="解析SCP"), 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                parts = line.split('\t')
                if len(parts)!= 4 and len(parts) !=3:
                    print(f"⚠️  跳过格式错误的行 {line_num}: {line[:50]}...")
                    continue
                
                utt, wav_path, text  = parts[0], parts[1], parts[2]
                instruct = text.split('<|endofprompt|>')[0].strip()

                utt2wav_files[wav_path] = utt
                utt2texts[wav_path] = instruct
                
            except Exception as e:
                print(f"❌ 处理行 {line_num} 失败: {e}")
                continue
    
    if missing_files:
        print(f"⚠️  发现 {len(missing_files)} 个缺失的音频文件（仅显示前5个）:")
        for f in missing_files[:5]:
            print(f"  - {f}")
    
    print(f"✅ 成功加载 {len(utt2texts)} 条有效记录")
    return utt2texts, utt2wav_files

if __name__ == "__main__":
    # 初始化Milvus
    milvus = AudioRAGMilvus(DB_PATH, COLLECTION_NAME)
    milvus.create_collection(VECTOR_DIM)
    wav_path2texts = {}
    wav_path2utt = {}

    # 加载数据
    # TODO: 这里填写需要存入数据库的文本文件 （组织格式： utt文件名 \t wav_path音频路径 \t text指令文本<endofprompt>目标文本）
    SCP_PATH = ''
    utt2texts, utt2wav_files = load_audio_metadata(SCP_PATH)
    
    # 构建插入数据（生成器模式，节省内存）
    def build_data_generator(utt2texts, utt2wav_files):
        """数据生成器，避免内存爆炸"""
        for wav_path, txt in utt2texts.items():
            yield {
                "vector": get_embedding(txt),
                "utt": utt2wav_files[wav_path],
                "text": txt,
            }
    
  
    print("构建数据列表...")
    data_list = []
    
    for record in tqdm(build_data_generator(utt2texts, utt2wav_files), 
                       total=len(utt2texts), desc="生成Embedding"):
        data_list.append(record)
    
    milvus.insert_batch(data_list, batch_size=1000)

    print("\n" + "="*50)
    print("测试检索功能")
    query_text = "以温暖而忧郁的语调娓娓道来"
    
    result = milvus.search(query_text, top_k=3)
    
    if result:
        print(f"  最匹配结果 (相似度: {result['score']:.4f}):")
        print(f"  文本: {result['text']}...")