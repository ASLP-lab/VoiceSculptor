import html
import io
import os
import sys
import queue
import wave
from argparse import ArgumentParser
from pathlib import Path
import traceback
import time
import tempfile
import subprocess
import soundfile as sf
import gradio as gr
import numpy as np
import pyrootutils
from loguru import logger
from typing import Tuple, Dict, Any, List
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import threading
from funasr_onnx import Paraformer
import requests
import torch
from cosyvoice.utils.file_utils import load_wav
from transformers import AutoTokenizer, AutoModelForCausalLM
from modeling_xcodec2 import XCodec2Model
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer
RAG_URL = "http://192.168.0.59:8000/rag/search"  # TODO:替换为对应的rag请求地址

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

asr_model = None
cosyvoice = None
cosyvoice_base = None

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from i18n import i18n
from text.chn_text_norm.text import Text as ChnNormedText



HEADER_MD = f"""# Instruct TTS

{i18n("A instruct text-to-speech model based on LLaSA and CosyVoice2 developed by ASLP.")}  

"""

TEXTBOX_PLACEHOLDER = i18n("Put your text here.")
SPACE_IMPORTED = False

model = None
codec_model = None
tokenizer = None
device = None
def get_embedding(qwen3emb, text: str):
    output = qwen3emb.encode([text])
    return output[0].tolist()

def build_html_error_message(error):
    return f"""
    <div style="color: red; 
    font-weight: bold;">
        {html.escape(str(error))}
    </div>
    """

def normalize_text(user_input, use_normalization):
    if use_normalization:
        return ChnNormedText(raw_text=user_input).normalize()
    else:
        return user_input

def normalize_text_final(user_input):
    return ChnNormedText(raw_text=user_input).normalize()
    

def normalize_me(user_input):
    return user_input
def extract_speech_ids(speech_tokens_str):
 
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            num_str = token_str[4:-2]

            num = int(num_str)
            speech_ids.append(num)
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids
    
def inference_one(
    model,
    codec_model,
    device,
    tokenizer, 
    refined_text,
    instruct_text,
    control_tags
):
    import torch
    from vllm import SamplingParams, TokensPrompt, SamplingParams
    logger.info(f"text: {refined_text}")
    logger.info(f"instruct_text: {instruct_text}")
    
    refined_text = normalize_text_final(refined_text)

    # 二选一：llm前端 or rag
    # instruct_text = process_file(instruct_text)
    instruct_text = normalize_text_final(instruct_text)

    if len(refined_text) < 5:
        raise ValueError("输入文本长度不能少于5个字符")
    if len(refined_text) > 200:
        raise ValueError("输入文本长度不能超过200个字符")
    
    logger.info(f"refined_text: {refined_text}")
    logger.info(f"refine_instruct_text: {instruct_text}")
    
    try:
        with torch.no_grad():
            target_text = instruct_text + '<|endofprompt|>' + control_tags + refined_text
            formatted_text = f"<|TEXT_UNDERSTANDING_START|>{target_text}<|TEXT_UNDERSTANDING_END|>"
            chat = [
                {"role": "user", "content": "Convert the text to speech:" + formatted_text},
                {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"}
            ]

            input_ids = tokenizer.apply_chat_template(
                chat, 
                tokenize=True, 
                return_tensors='pt', 
                continue_final_message=True
            )
            input_ids = input_ids.to(device)
            speech_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
            prompt_ids = input_ids.squeeze(0).tolist()
            prompt = TokensPrompt(prompt_token_ids=prompt_ids)
            # Generate the speech autoregressively
            start_time = time.time()
            sampling_params = SamplingParams(
                temperature=0.9,
                top_p=0.95,
                top_k=15,
                max_tokens=2048,
                repetition_penalty=1.05,
                stop_token_ids=[speech_end_id]
                )
            outputs = model.generate(
                prompts=prompt,
                sampling_params=sampling_params
            )
            # Extract the speech tokens
            generated_ids = outputs[0].outputs[0].token_ids[:-1]

            speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)   

            # Convert  token <|s_23456|> to int 23456 
            speech_tokens = extract_speech_ids(speech_tokens)

            speech_tokens = torch.tensor(speech_tokens).to(device).unsqueeze(0).unsqueeze(0)
            
            # Decode the speech tokens to speech waveform
            gen_wav = codec_model.decode_code(speech_tokens) 
            end_time = time.time()
            cost_time = end_time - start_time
            gen_wav = gen_wav.squeeze(0).squeeze(0).cpu().numpy()
            duration = len(gen_wav) / 16000
            rtf = cost_time / duration
            logger.info(f"rtf is {rtf:.3f}s")
            safe_text = instruct_text[:10]
            save_path = f"./outputs/{int(time.time())}_{safe_text}.wav"
            sf.write(save_path, gen_wav, 16000)
            logger.info(f"音频已保存 → {save_path}")

            return 16000, gen_wav
    except Exception as e:
        logger.error(f"发生错误: {e}", exc_info=True)
        logger.error("错误详细信息:\n" + traceback.format_exc())
        return None, "error"
    

def get_asr(wav):
    global asr_model
    result = asr_model(wav.squeeze().cpu().numpy())
    return result[0]['preds'][0]
    
def inference_cs2(refined_text: str, instruct_text: str, ref_wav: str = None, user_instruct_tag: bool = False, cs2_if_rag_text: bool=True):
    logger.info(f"text: {refined_text}")
    logger.info(f"instruct_text: {instruct_text}")
    global cosyvoice
    global cosyvoice_base

    try:
        if not user_instruct_tag:
            use_instruct = False
            with torch.no_grad():
                prompt_speech_16k_path = ref_wav
                prompt_speech_16k = load_wav(prompt_speech_16k_path, 16000)
            ret = []
            prompt_text = get_asr(prompt_speech_16k)
            logger.info(f"prompt_text: {prompt_text}")
            for i, j in enumerate(cosyvoice_base.inference_zero_shot(refined_text, prompt_text, prompt_speech_16k)):
                ret.append(j['tts_speech'])

            ret_speech = torch.cat(ret, dim=-1) 
            return (cosyvoice_base.sample_rate, ret_speech.squeeze().cpu().numpy()), None
        else:
            use_instruct = True
            # 是否启用检索增强
            if cs2_if_rag_text:
                instruct_text = rag_infer(instruct_text)
            with torch.no_grad():
                prompt_speech_16k_path = ref_wav
                prompt_speech_16k = load_wav(prompt_speech_16k_path, 16000)
            ret = []
            for i, j in enumerate(cosyvoice.inference_instruct2(refined_text, instruct_text, prompt_speech_16k, use_instruct, stream=False, icl_tag=False, spk_rate_tag=False)):
                ret.append(j['tts_speech'])

            ret_speech = torch.cat(ret, dim=-1)
            return (cosyvoice.sample_rate, ret_speech.squeeze().cpu().numpy()), None

    except Exception as e:
        logger.error(f"发生错误: {e}", exc_info=True)
        logger.error("错误详细信息:\n" + traceback.format_exc())
        return None, f"error:{e}"

def _pre_init_cuda_env(cuda_id: str):
    """必须在任何 cuda import 前调用"""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
    os.environ["VLLM_USE_V1"] = "0"

def worker_process(
    worker_id: int, 
    device: str,
    model_path: str, 
    codec_path: str,
    task_queue: multiprocessing.Queue, 
    result_queue: multiprocessing.Queue
):
    try:
        cuda_id = device.split(":")[-1]
        _pre_init_cuda_env(cuda_id)

        os.environ['TRANSFORMERS_CACHE'] = f'/tmp/transformers_cache_w{worker_id}'
        os.environ['HF_HOME'] = f'/tmp/hf_home_w{worker_id}'

        torch.manual_seed(42 + worker_id)
        np.random.seed(42 + worker_id)
        logger.info(f"[Worker {worker_id}] 加载模型到 {device}...")

        device = 'cuda:0'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = LLM(model=str(model_path), gpu_memory_utilization=0.90, max_model_len=2048, enable_prefix_caching=True, dtype='auto',
            quantization=None, enforce_eager=False, kv_cache_dtype='auto')
        
        codec_model = XCodec2Model.from_pretrained(codec_path).eval().to(device)
        
        logger.info(f"[Worker {worker_id}] 模型加载完成")
        
        while True:
            try:
                task = task_queue.get(timeout=0.1)
                
                if task is None:
                    logger.info(f"[Worker {worker_id}] 收到终止信号")
                    break
                    
                task_id = task['task_id']
                refined_text = task['refined_text']
                instruct_text = task['instruct_text']
                control_tags = task['control_tags']
                
                logger.info(f"[Worker {worker_id}] 处理任务 {task_id}")
                
                try:
                    sr, wav = inference_one(
                        model, codec_model, device, tokenizer, 
                        refined_text, instruct_text, control_tags
                    )
                    if sr is None:       
                        result = None
                        error  = wav    
                    else:                 
                        wav_bytes = wav.astype(np.float32).tobytes()
                        result = (sr, wav_bytes)
                        error  = "no error"
                    
                except Exception as e:
                    result = None
                    error = f"推理错误: {str(e)}"
                    logger.error(f"[Worker {worker_id}] 任务 {task_id} 失败: {e}")
                
                result_queue.put({
                    'task_id': task_id,
                    'worker_id': worker_id,
                    'result': result,
                    'error': error
                })
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[Worker {worker_id}] 未知错误: {e}", exc_info=True)
                continue
                
    except Exception as e:
        import traceback
        print(traceback.print_exc())
        logger.error(f"[Worker {worker_id}] 初始化失败: {e}", exc_info=True)
        result_queue.put(None)


class ParallelInferenceManager:
    def __init__(self, model_path: str, codec_path: str, devices: List[str]):
        self.devices = devices
        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.workers: List[multiprocessing.Process] = []
        self.results_cache: Dict[int, Dict[str, Any]] = {}
        self.next_task_id = 0
        self.lock = threading.Lock()
        
        self._start_workers(model_path, codec_path)
        
        self.running = True
        self.collect_thread = threading.Thread(target=self._collect_results, daemon=True)
        self.collect_thread.start()
        
        logger.info(f"ParallelInferenceManager 初始化完成，{len(devices)}个工作进程")
    
    def _start_workers(self, model_path: str, codec_path: str):
        for worker_id, device in enumerate(self.devices):
            p = multiprocessing.Process(
                target=worker_process,
                args=(worker_id, device, model_path, codec_path,
                      self.task_queue, self.result_queue),
                daemon=True 
            )
            p.start()
            self.workers.append(p)
            logger.info(f"工作进程 {worker_id} (PID: {p.pid}) 已启动")

    def _collect_results(self):
        while self.running:
            try:
                result = self.result_queue.get(timeout=0.1)
                if result is None: 
                    logger.error("工作进程初始化失败信号")
                    continue
                
                task_id = result['task_id']
                with self.lock:
                    self.results_cache[task_id] = result
                logger.debug(f"收集到任务 {task_id} 的结果")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"结果收集线程错误: {e}", exc_info=True)
    
    def submit_task(self, refined_text: str, instruct_text: str, control_tags: str) -> int:
        """提交任务，返回task_id"""
        with self.lock:
            task_id = self.next_task_id
            self.next_task_id += 1
        
        self.task_queue.put({
            'task_id': task_id,
            'refined_text': refined_text,
            'instruct_text': instruct_text,
            'control_tags': control_tags
        })
        logger.info(f"提交任务 {task_id}")
        return task_id
    
    def get_results(self, task_ids: List[int], timeout: float = 30.0) -> Tuple:
        """
        等待并收集多个任务的结果
        返回: (audio1, error1, audio2, error2, ...)
        """
        results = {}
        start_time = time.time()
        
        while len(results) < len(task_ids) and (time.time() - start_time) < timeout:
            with self.lock:
                for task_id in task_ids:
                    if task_id in self.results_cache:
                        results[task_id] = self.results_cache.pop(task_id)
            
            if len(results) < len(task_ids):
                time.sleep(0.05)
        
        if len(results) < len(task_ids):
            missing = set(task_ids) - set(results.keys())
            logger.warning(f"任务超时: {missing}")
            for task_id in missing:
                results[task_id] = {'result': None, 'error': 'Timeout'}
        
        ordered_results = []
        for task_id in sorted(task_ids):
            result_data = results[task_id]
            if result_data['result'] is not None:
                sr, wav_bytes = result_data['result']
                wav_array = np.frombuffer(wav_bytes, dtype=np.float32)
                ordered_results.append((sr, wav_array))
            else:
                ordered_results.append(None)
            ordered_results.append(result_data['error'])
        
        return tuple(ordered_results)
    
    def shutdown(self):
        logger.info("正在关闭ParallelInferenceManager...")
        self.running = False
        
        for _ in self.workers:
            self.task_queue.put(None)
        
        for i, p in enumerate(self.workers):
            p.join(timeout=10)
            if p.is_alive():
                logger.warning(f"工作进程 {i} 未正常退出，强制终止")
                p.terminate()
        
        logger.info("ParallelInferenceManager 已关闭")
        
def check_audio_validity(wav_data):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        temp_wav.write(wav_data)
        temp_wav.close()
        
        try:
            ffmpeg_command = [
                'ffmpeg', '-v', 'error', '-i', temp_wav.name, '-f', 'null', '-'
            ]
            subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info(f"音频文件 {temp_wav.name} 验证成功。")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"音频文件 {temp_wav.name} 无效。FFmpeg 错误: {e.stderr.decode()}")
            return False
        finally:
            if os.path.exists(temp_wav.name):
                os.remove(temp_wav.name)
                

n_audios = 1
global_audio_list = []
global_error_list = []



def wav_chunk_header(sample_rate=48000, bit_depth=16, channels=1):
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(bit_depth // 8)
        wav_file.setframerate(sample_rate)

    wav_header_bytes = buffer.getvalue()
    buffer.close()
    return wav_header_bytes



def build_app(llasa_1b, codec_path):
    
    devices = ["cuda:1", "cuda:2", "cuda:3"]
    inference_manager = ParallelInferenceManager(llasa_1b, codec_path, devices)
    
    import atexit
    atexit.register(inference_manager.shutdown)
    
    def build_control_tags(
        age, gender, pitch, pitch_var, volume, speed, emo
    ):
        tag_map = {
            "小孩": "<|小孩|>",
            "青年": "<|青年|>",
            "中年": "<|中年|>",
            "老年": "<|老年|>",
            "男性": "<|男性|>",
            "女性": "<|女性|>",
            "音调很高": "<|音调很高|>",
            "音调较高": "<|音调较高|>",
            "音调中等": "<|音调中等|>",
            "音调较低": "<|音调较低|>",
            "音调很低": "<|音调很低|>",
            "音调变化很强": "<|音调变化很强|>",
            "音调变化较强": "<|音调变化较强|>",
            "音调变化一般": "<|音调变化一般|>",
            "音调变化较弱": "<|音调变化较弱|>",
            "音调变化很弱": "<|音调变化很弱|>",
            "音量很大": "<|音量很大|>",
            "音量较大": "<|音量较大|>",
            "音量中等": "<|音量中等|>",
            "音量较小": "<|音量较小|>",
            "音量很小": "<|音量很小|>",
            "语速很快": "<|语速很快|>",
            "语速较快": "<|语速较快|>",
            "语速中等": "<|语速中等|>",
            "语速较慢": "<|语速较慢|>",
            "语速很慢": "<|语速很慢|>",
            "开心": '<|开心|>',
            "生气": '<|生气|>',
            "难过": '<|难过|>',
            "惊讶": '<|惊讶|>',
            "厌恶": '<|厌恶|>',
            "害怕": '<|害怕|>',
            "其他":'<|其他|>'
        }

        tags = []
        for v in [gender, age,  speed, volume, pitch, pitch_var, emo]:
            if v != "不指定":
                tags.append(tag_map[v])

        return "".join(tags)

    def inference_parallel(
        refined_text,
        instruct_text,
        if_rag_text,
        age, gender, pitch, pitch_var, volume, speed, emo
    ):
        control_tags = build_control_tags(
            age, gender, pitch, pitch_var, volume, speed, emo
        )
        if if_rag_text:
            instruct_text = rag_infer(instruct_text)
        task_ids = [
            inference_manager.submit_task(refined_text, instruct_text, control_tags)
            for _ in range(3)
        ]
        results = inference_manager.get_results(task_ids, timeout=60.0)
        return results[0], results[1], results[2], results[3], results[4], results[5]
    
    from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
    from cosyvoice.utils.file_utils import load_wav

    global cosyvoice
    global cosyvoice_base
    global asr_model
    cosyvoice = CosyVoice2('./ckpt/CosyVoice2-0.5B', './ckpt/CosyVoice2-0.5B/llm.pt', './ckpt/CosyVoice2-0.5B/cosyvoice2.yaml',  use_instruct=True, load_jit=False, load_trt=False, load_vllm=False, fp16=False)
    cosyvoice_base = CosyVoice2('./ckpt/CosyVoice2-0.5B', './ckpt/CosyVoice2-0.5B/llm.pt', './ckpt/CosyVoice2-0.5B/cosyvoice2.yaml',  use_instruct=False, load_jit=False, load_trt=False, load_vllm=False, fp16=False)
    model_dir = "./asr/Paraformer-large"
    asr_model = Paraformer(model_dir, batch_size=1, quantize=True)
    
    logger.info("✅ 所有模型加载完成")

    # Template definitions (keep as is)
    INSTRUCT_TEMPLATES = {
        "自定义": "",
        "幼儿园女教师-温柔甜美": "这是一位幼儿园女教师，用甜美明亮的嗓音，以极慢且富有耐心的语速，带着温柔鼓励的情感，用标准普通话给小朋友讲睡前故事，音量轻柔适中，咬字格外清晰。",
        "小女孩-尖锐清脆": "一位7岁的小女孩，用天真高亢的童声，以不稳定的快节奏，充满兴奋和炫耀地背诵乘法口诀，音调忽高忽低，带着儿童特有的尖锐清脆。",
        "老奶奶-沙哑低沉": "一位慈祥的老奶奶，用沙哑低沉的嗓音，以极慢而温暖的语速讲述民间传说，音量微弱但清晰，带着怀旧和神秘的情感。",
        "诗歌朗诵-雄浑有力": "一位男性现代诗朗诵者，用深沉磁性的低音，以顿挫有力的节奏演绎艾青诗歌，音量洪亮，情感激昂澎湃。",
        "童话风格-甜美夸张": "这是一位女性童话旁白朗诵者，用甜美夸张的童声，以跳跃变化的语速讲述《安徒生童话》，音调偏高，充满奇幻色彩。",
        "评书风格-抑扬顿挫": "这是一位男性评书表演者，用传统说唱腔调，以变速节奏和韵律感极强的语速讲述江湖故事，音量时高时低，充满江湖气。",
        "新闻风格-平静专业": "这是一位女性新闻主播，用标准普通话以清晰明亮的中高音，以平稳专业的语速播报时事新闻，音量洪亮，情感客观中立。",
        "相声风格-夸张幽默": "这是一位男性相声表演者，用夸张幽默的嗓音，以时快时慢的节奏抖包袱，音调起伏大，充满喜感和节奏感。",
        "游戏直播-亢奋激昂": "这是一位男性游戏解说，用亢奋激昂的嗓音，以极快且情绪化的语速直播电竞比赛，音量突然爆发，充满悬念和热血。",
        "悬疑小说-低沉神秘": "一位男性悬疑小说演播者，用低沉神秘的嗓音，以时快时慢的变速节奏营造紧张氛围，音量忽高忽低，充满悬念感。",
        "戏剧表演-夸张戏剧": "这是一位男性戏剧表演者，用夸张戏剧化的嗓音，以忽高忽低的音调和时快时慢的语速表演独白，充满张力。",
        "法治节目-庄严庄重": "这是一位男性法治节目主持人，用严肃庄重的嗓音，以平稳有力的语速讲述案件，音量适中，体现法律的威严。",
        "纪录片旁白-低沉磁性": "这是一位男性纪录片旁白，用深沉磁性的嗓音，以缓慢而富有画面感的语速讲述自然奇观，音量适中，充满敬畏和诗意。",
        "广告配音-沧桑浑厚": "这是一位男性白酒品牌广告配音，用沧桑浑厚的嗓音，以缓慢而豪迈的语速，音量洪亮，传递历史底蕴和男人情怀。",
        "冥想引导师-空灵悠长": "一位女性冥想引导师，用空灵悠长的气声，以极慢而飘渺的语速，配合环境音效，音量轻柔，营造禅意空间。",
        "ASMR-气声耳语": "一位女性ASMR主播，用气声耳语，以极慢而细腻的语速，配合唇舌音，音量极轻，营造极度放松的氛围。",
    }  

    TEXT_REQUIREMENTS = {
        "自定义": "",
        "幼儿园女教师-温柔甜美": "月亮婆婆升上天空啦，星星宝宝都困啦。小白兔躺在床上，盖好小被子，闭上眼睛。兔妈妈轻轻地唱着摇篮曲：睡吧睡吧，我亲爱的宝贝。",
        "小女孩-尖锐清脆": "一一得一！一二得二！一三得三！我会背乘法口诀啦！老师今天表扬我啦！妈妈说我最棒！",
        "老奶奶-沙哑低沉": "很久很久以前，在山的那边，住着一只会说话的狐狸。它常常在月圆之夜，变成美丽的姑娘，来到村子里。",
        "诗歌朗诵-雄浑有力": "为什么我的眼里常含泪水？因为我对这土地爱得深沉。这土地，这河流，这吹刮着的暴风。",
        "童话风格-甜美夸张": "在一个很冷很冷的夜晚，小女孩擦亮了一根火柴。突然，温暖的火炉出现了！她觉得自己好像坐在火炉旁。",
        "评书风格-抑扬顿挫": "话说那武松，提着哨棒，直奔景阳冈。天色将晚，酒劲上头，只听一阵狂风，老虎来啦！",
        "新闻风格-平静专业": "本台讯，今日凌晨，我国成功发射新一代载人飞船试验船。此次任务验证了多项关键技术，为后续空间站建设奠定基础。",
        "相声风格-夸张幽默": "我这个人啊，最大的优点就是太谦虚。谦虚到什么程度？连谦虚本身都觉得我太谦虚了！",
        "游戏直播-亢奋激昂": "大招！大招好了！开团了！ACE！团灭！这波操作神了！冠军相尽显无疑！",
        "悬疑小说-低沉神秘": "深夜，他独自走在空无一人的小巷。脚步声，回声，还有……另一个人的呼吸声。他猛地回头——什么也没有。",
        "戏剧表演-夸张戏剧": "我疯了！彻底疯了！你们都说我疯了！可疯的是这个世界！清醒的人反而被当成疯子！",
        "法治节目-庄严庄重": "天网恢恢，疏而不漏。任何触犯法律的行为，终将受到公正的审判。正义或许会迟到，但绝不会缺席。",
        "纪录片旁白-低沉磁性": "在这片广袤的非洲草原上，生命与死亡每天都在上演。猎豹的速度，羚羊的敏捷，都是生存的代价。",
        "广告配音-沧桑浑厚": "一杯敬过往，一杯敬远方。传承千年的酿造工艺，只在每一滴醇香。老朋友，值得好酒。",
        "冥想引导师-空灵悠长": "想象你是一片叶子，随风飘落。没有牵挂，没有重量。只有呼吸，只有当下，只有宁静。",
        "ASMR-气声耳语": "现在，让我在你耳边轻声细语。听到我的声音了吗？放松你的头皮，感受每一个毛孔都在呼吸。",
    }  

    with gr.Blocks(theme=gr.themes.Base()) as app:
        gr.Markdown(HEADER_MD)

        app.load(
            None,
            None,
            js="() => {const params = new URLSearchParams(window.location.search);if (!params.has('__theme')) {params.set('__theme', '%s');window.location.search = params.toString();}}"
            % args.theme,
        )

        all_output_components = [] 
        all_output_components_cs2 = []
        
        with gr.Column():
            # Input area
            with gr.Column(scale=3):
                gr.Markdown("## 🪄 Voice Design By LLaSA-Instruct")
                gr.Markdown("### 请使用以下模板给出指令：这是一位【人设】，【音色描述】，【语速、语调、音量、情感、场景等】。")
                gr.Markdown("### 若启用指令检索增强，可提升生成效果。否则，输入的文本应该与指令模板一致，以达到最优的音色设计效果。若不符合您的需求，请多次生成选取最优。")
                gr.Markdown("### 细粒度声音控制（可选）")

                with gr.Row():
                    age_ctrl = gr.Dropdown(
                        label="年龄",
                        choices=["不指定", "小孩", "青年", "中年", "老年"],
                        value="不指定"
                    )
                    gender_ctrl = gr.Dropdown(
                        label="性别",
                        choices=["不指定", "男性", "女性"],
                        value="不指定"
                    )

                with gr.Row():
                    pitch_ctrl = gr.Dropdown(
                        label="音调高度",
                        choices=["不指定", "音调很高", "音调较高", "音调中等", "音调较低", "音调很低"],
                        value="不指定"
                    )
                    pitch_var_ctrl = gr.Dropdown(
                        label="音调变化",
                        choices=["不指定", "音调变化很强", "音调变化较强", "音调变化一般", "音调变化较弱", "音调变化很弱"],
                        value="不指定"
                    )

                with gr.Row():
                    volume_ctrl = gr.Dropdown(
                        label="音量",
                        choices=["不指定", "音量很大", "音量较大", "音量中等", "音量较小", "音量很小"],
                        value="不指定"
                    )
                    speed_ctrl = gr.Dropdown(
                        label="语速",
                        choices=["不指定", "语速很快", "语速较快", "语速中等", "语速较慢", "语速很慢"],
                        value="不指定"
                    )
                    
                with gr.Row():
                    emo_ctrl = gr.Dropdown(
                        label="情感",
                        choices=["不指定", "开心", "生气", "难过", "惊讶", "厌恶", "害怕"],
                        value="不指定"
                    )

                instruct_template = gr.Dropdown(
                    choices=list(INSTRUCT_TEMPLATES.keys()),
                    value="幼儿园女教师-温柔甜美",
                    # label=i18n("Instruct Examples"),
                    label=i18n("Instruct Examples") + "  📌 必选项",
                    interactive=True,
                    scale=3,
                )
                if_rag_text = gr.Checkbox(
                    label=i18n("Instruct Rag Control"),
                    value=False,
                    scale=3,
                    visible=True,
                )
                
                instruct_text = gr.Textbox(
                    label=i18n("Instruct Text"),
                    placeholder=TEXTBOX_PLACEHOLDER,
                    lines=4,
                )
                
                text = gr.Textbox(
                    label=i18n("Input Text"), 
                    placeholder=TEXTBOX_PLACEHOLDER, 
                    lines=4
                )

                def apply_template(tpl_name):
                    return INSTRUCT_TEMPLATES[tpl_name], TEXT_REQUIREMENTS[tpl_name]
            
                def apply_template_single(tpl_name):
                    return INSTRUCT_TEMPLATES[tpl_name]


                # def detect_custom(instruct, text_req):
                #     for k, v in INSTRUCT_TEMPLATES.items():
                #         if instruct == v and TEXT_REQUIREMENTS[k] == text_req:
                #             return k
                #     return "自定义"

                # instruct_text.change(
                #     detect_custom,
                #     inputs=[instruct_text, text],  # 添加 text_requirement 作为输入
                #     outputs=[instruct_template],
                # )
                instruct_template.change(
                    apply_template,
                    inputs=[instruct_template],
                    outputs=[instruct_text, text],  # 同时更新两个组件
                )
                
                with gr.Row():
                    if_refine_text = gr.Checkbox(
                        label=i18n("Text Normalization"),
                        value=True,
                        scale=1,
                        visible=False
                    )
                
                # Generate button
                with gr.Row():
                    generate = gr.Button(
                        value="\U0001F3A7 " + i18n("Generate 3 Audios"), 
                        variant="primary"
                    )

            with gr.Column(scale=3):
                gr.Markdown("## 🎵 Generation Results")
                
                for idx in range(3):
                    with gr.Row():
                        gr.Markdown(f"### Results {idx + 1}")
                    
                    with gr.Row():
                        # 先创建audio组件
                        audio_output = gr.Audio(
                            label=i18n(f"Generated Audio {idx + 1}"),
                            type="numpy",
                            interactive=False,
                            visible=True,
                        )
                        # 再创建error组件
                        error_output = gr.Text(
                            label=i18n(f"Error Message {idx + 1}"),
                            visible=True,
                        )
                        
                        # 按交错顺序添加到列表：audio1, error1, audio2, error2...
                        all_output_components.append(audio_output)
                        all_output_components.append(error_output)
                        
                gr.Markdown("## 🎙️ Voice Clone By CosyVoice2-Instruct")
                gr.Markdown("### 该部分指令主要控制风格，音色主要由prompt音频提供，建议和voice design保持联动，使用voice design的生成的音频和其指令作为语音合成的条件。")
                gr.Markdown("### 若选择不提供控制指令，默认为音色克隆模式。")
                
                # 如果有输入的instruct，则需要显示这个按钮
                user_instruct = gr.Checkbox(
                    label=i18n("是否提供控制指令"),
                    value=False,
                    scale=1
                )
                
                # 根据选择显示不同的组件
                with gr.Row(visible=False) as cs2_instruct_text_row:
                    with gr.Column(scale=2):
                        cs2_instruct_template = gr.Dropdown(
                            choices=list(INSTRUCT_TEMPLATES.keys()),
                            value="幼儿园女教师-温柔甜美",
                            label=i18n("Instruct Examples") + "  📌 必选项",
                            interactive=True,
                            scale=3,
                        )
                        cs2_if_rag_text = gr.Checkbox(
                            label=i18n("Instruct Rag Control"),
                            value=True,
                            scale=3,
                            visible=True,
                        )
                    cs2_instruct_text = gr.Textbox(
                            label=i18n("CosyVoice2 Instruct Text"),
                            placeholder=TEXTBOX_PLACEHOLDER,
                            lines=4,
                    )
                    cs2_instruct_template.change(
                        apply_template_single,
                        inputs=[cs2_instruct_template],
                        outputs=[cs2_instruct_text],
                    )
                
                cs2_target_text = gr.Textbox(
                        label=i18n("CosyVoice2 Input Text"), 
                        placeholder=TEXTBOX_PLACEHOLDER, 
                        lines=4
                )
                    
                # 根据选择显示不同的组件
                with gr.Row(visible=True) as ref_audio_row:
                    ref_audio = gr.Audio(
                        label=i18n("请上传参考音频"),
                        type="filepath",
                        interactive=True
                    )

                def toggle_instruct_input(user_provide):
                    if user_provide:
                        return gr.update(visible=True)
                    else:
                        return gr.update(visible=False)
                    
                def toggle_audio_input(user_provide):
                    if user_provide:
                        return gr.update(visible=True), gr.update(visible=False)
                    else:
                        return gr.update(visible=False), gr.update(visible=True)
                
                user_instruct.change(
                    toggle_instruct_input,
                    inputs=[user_instruct],
                    outputs=[cs2_instruct_text_row]
                )

                with gr.Row():
                    audio = gr.Audio(
                        label=f"合成音频",
                        type="numpy",
                        interactive=False,
                        visible=True
                    )
                    all_output_components_cs2.append(audio)
                    error = gr.Text(
                        label=i18n(f"Error Message"),
                        visible=True,
                    )
                    all_output_components_cs2.append(error)
                    
                with gr.Row():
                    with gr.Column(scale=3):
                        generate_cs2 = gr.Button(
                            value="\U0001F3A7 " + i18n("Generate"), variant="primary"
                        )


        text.input(
            fn=normalize_text, 
            inputs=[text, if_refine_text], 
            outputs=[gr.Textbox(visible=False)]  # Remove non-existent refined_text
        )

        generate.click(
            fn=inference_parallel,
            inputs=[
                text,
                instruct_text,
                if_rag_text,
                age_ctrl,
                gender_ctrl,
                pitch_ctrl,
                pitch_var_ctrl,
                volume_ctrl,
                speed_ctrl,
                emo_ctrl
            ],
            outputs=all_output_components
        )

        
        generate_cs2.click(
            inference_cs2,
            [
                cs2_target_text,
                cs2_instruct_text,
                ref_audio,
                user_instruct,
                cs2_if_rag_text
            ],
            all_output_components_cs2
        )
    return app

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--llasa_ckpt_path",
        type=Path,
    )
    parser.add_argument(
        "--xcodec_ckpt_path",
        type=Path,
    )
    
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-gradio-length", type=int, default=0)
    parser.add_argument("--theme", type=str, default="light")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    llasa_1b = args.llasa_ckpt_path
    codec_path = args.xcodec_ckpt_path
    device = args.device


    logger.info("Launching the web UI...")


    app = build_app(llasa_1b, codec_path)
    app.launch(share=True, 
               server_name= "127.0.0.1",
               server_port = 7899)
