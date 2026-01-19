<div align="center">
    <h1>
    VoiceSculptor
    </h1>
    <p>Official inference code for <br>
    <b><em>VoiceSculptor: Your Voice, Designed By You</em></b>
    </p>
    <p>
    <img src="assets/logo.png" style="width: 400px; height: 400px;">
    </p>
    <a href="https://hujingbin1.github.io/VoiceSculptor-Demo"><img src="https://img.shields.io/badge/Demo-Page-lightgrey" alt="version"></a>
    <a href="https://huggingface.co/ASLP-lab/VoiceSculptor-VD"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue' alt="HF-model"></a>
    <a href="https://arxiv.org/pdf/2601.10629"><img src='https://img.shields.io/badge/Report-Github?label=Technical&color=red' alt="technical report"></a>
    <a href="https://huggingface.co/spaces/ASLP-lab/VoiceSculptor"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue' alt="HF-demo"></a>
    <a href="https://github.com/ASLP-lab/VoiceSculptor"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="Apache-2.0"></a>
</div>

## 💡 System Overview

<p>
    <img src="assets/llasa3b.png">
</p>
VoiceSculptor is composed of two core components: voice design and voice clone. The voice design module enables the generation of timbre from natural language descriptions and supports command refinement through Retrieval-Augmented Generation (RAG). It also provides fine-grained control over voice attributes, including gender, age, speaking rate, fundamental frequency, volume, and emotional expression. The synthesized audio produced by the voice design module can be used as a prompt waveform for the CosyVoice2 voice cloning model, enabling timbre cloning and downstream speech synthesis tasks.

## 📊 Instruct TTS Eval

#### Instruct TTS Eval (ZH)

| Model | APS (%) | DSD (%) | RP (%) | AVG (%) |
|------|---------|---------|--------|---------|
| Gemini 2.5-Flash* | 88.2 | 90.9 | 77.3 | 85.4 |
| Gemini 2.5-Pro* | 89.0 | 90.1 | 75.5 | 84.8 |
| GPT-4o-Mini-TTS* | 54.9 | 52.3 | 46.0 | 51.1 |
| ElevenLabs* | 42.8 | 50.9 | 59.1 | 50.9 |
| VoxInstruct | 47.5 | 52.3 | 42.6 | 47.5 |
| MiMo-Audio-7B-Instruct | 70.1 | 66.1 | 57.1 | 64.5 |
| **VoiceSculptor** | **75.7** | **64.7** | **61.5** | **67.6** |

> **Note**
>
> - Models marked with `*` are commercial models.  
> - InstructTTSEval — InstructTTSEval: Benchmarking Complex Natural-Language Instruction Following in Text-to-Speech Systems. arXiv preprint arXiv:2506.16381.

## ✨ Demo Video

<div align="center">

<https://github.com/user-attachments/assets/40eee464-182a-4e1e-ad92-684a7cbbbce6>

</div>

## 🔥 News

- **[2026-1-18]** We have released the RAG and WebUI code, and the WebUI supports vLLM.
- **[2026-1-16]** We have released the technical report! [VoiceSculptor](https://arxiv.org/abs/2601.10629)
- **[2026-1-8]** We have released the Demo Page and Demo Video! [VoiceSculptor-Demo-Page](https://hujingbin1.github.io/VoiceSculptor-Demo)
- **[2026-1-2]** We opened the repository and uploaded the voice design models! [VoiceSculptor](https://huggingface.co/ASLP-lab/VoiceSculptor-VD)

## 🚀 Getting Started

### 1. Environment Setup

Follow the steps below to clone the repository and install the required environment.

```bash
# Clone the repository and enter the directory
git clone https://github.com/ASLP-lab/VoiceSculptor.git
cd VoiceSculptor

# Create and activate a Conda environment
conda create -n VoiceSculptor python=3.10 -y
conda activate VoiceSculptor

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Pre-trained Models

```bash
git lfs install
git clone https://huggingface.co/ASLP-lab/VoiceSculptor-VD
git clone https://huggingface.co/HKUSTAudio/xcodec2
```

### 3. Infer

For detailed instructions on how to design high-quality voice prompts,  
please refer to [Voice Design Guide](docs/voice_design.md) or [Voice Design Guide EN](docs/voice_design_en.md).

You need to specify the local paths to the voice-design model and the xcodec2 model in the infer.py file.

```bash
python infer.py
```


### 4. RAG

#### RAG Private Text Vector Database

This project provides a simple workflow to build and deploy a **private text vector database** for Retrieval-Augmented Generation (RAG).
You can create your own database from text files and run a lightweight server to query it via a client.

---

#### 1. Build the Private Vector Database

Use the `create_database.py` script located in the `rag/` directory.

Before running the script, make sure to:

* Replace the **model path**
* Replace the **input text file path**

**Input Text File Format**

Each line in the input file should follow this structure:

```

utt_file_name \t wav_path \t text_command <|endofprompt|> target_text

```

**Example**

```

ZH_B00074_S00400_W000029        Emilia/ZH/ZH_B00074/ZH_B00074_S00400/mp3/ZH_B00074_S00400_W000029.mp3        这是一位中年男性的中低音有声书朗读，嗓音浑厚略带粗砺，以标准普通话清晰咬字，通过多变的语调 动态的语速和戏剧化的停顿，生动演绎充满张力的动作场景<|endofprompt|>而就在此时，邵飞忽然露出坏笑，他一脚踹在赵和的窝锅子上。赵和顿时扑通一下，跪了下去。

```

**Run the Database Creation Script**

```bash
python rag/create_database.py
````

Once completed, your private RAG vector database will be generated.

---

After building the database, you can launch the query service using the server script in the `rag/` folder.

Before starting the server:

* Update the **database path**
* Update the **model path**

#### 2. Start the Server

```bash
bash run_server.sh
```

---

#### 3. Query the Database Using the Client

To connect to the running service, modify the **IP address and port** in `rag/client.py` so they match the server configuration.

**Run the Client**

```bash
python client.py
```

---

#### 4. Workflow Summary

| Step | Action                                    |
| ---- | ----------------------------------------- |
| 1    | Prepare and format your text dataset      |
| 2    | Build the vector database                 |
| 3    | Configure and start the RAG server        |
| 4    | Query and retrieve results via the client |

---


### 5. WebUI
#### TODO：add instruction manual
```bash
python gradio/webui.py
```

### 6. vLLM
#### TODO：add instruction manual
```bash
python gradio/webui.py
```



## 📋 TODO
- [x] 🌐 **Demo website**
- [x] 🔓 **Release inference code**
- [x] 🤗 **Release HuggingFace model**
- [x] 🤗 **HuggingFace Space**
- [x] 📝 **Release Technical Report**
- [x] 💡 **Release RAG code**
- [x] ✨ **Release gradio code**
- [x] 🚀 **Release vLLM code**
- [ ] 🔓 **Release both the text generation code and a sample dataset**
- [ ] 🔓 **Release training code**

## Citation

```bibtex
@misc{hu2026voicesculptorvoicedesigned,
      title={VoiceSculptor: Your Voice, Designed By You}, 
      author={Jingbin Hu and Huakang Chen and Linhan Ma and Dake Guo and Qirui Zhan and Wenhao Li and Haoyu Zhang and Kangxiang Xia and Ziyu Zhang and Wenjie Tian and Chengyou Wang and Jinrui Liang and Shuhan Guo and Zihang Yang and Bengu Wu and Binbin Zhang and Pengcheng Zhu and Pengyuan Xie and Chuan Xie and Qiang Zhang and Jie Liu and Lei Xie},
      year={2026},
      eprint={2601.10629},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2601.10629}, 
}
@misc{ye2025llasascalingtraintimeinferencetime,
      title={Llasa: Scaling Train-Time and Inference-Time Compute for Llama-based Speech Synthesis},
      author={Zhen Ye and Xinfa Zhu and Chi-Min Chan and Xinsheng Wang and Xu Tan and Jiahe Lei and Yi Peng and Haohe Liu and Yizhu Jin and Zheqi Dai and Hongzhan Lin and Jianyi Chen and Xingjian Du and Liumeng Xue and Yunlin Chen and Zhifei Li and Lei Xie and Qiuqiang Kong and Yike Guo and Wei Xue},
      year={2025},
      eprint={2502.04128},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2502.04128},
}
```

## License

We use the Apache 2.0 license. Researchers and developers are free to use the codes and model weights of our VoiceSculptor. Check the license at [LICENSE](LICENSE.txt) for more details.

## Acknowledgement
- This repo benefits from [LLaSA](https://github.com/zhenye234/LLaSA_training)
- This repo benefits from [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)


##  Usage Disclaimer
Additional Notice on Generated Voices

This project provides a speech synthesis model for voice design, intended for academic research, educational purposes, and legitimate applications, such as personalized speech synthesis, assistive technologies, and linguistic research.

Please note:

Do not use this model for unauthorized voice cloning, impersonation, fraud, scams, deepfakes, or any illegal or malicious activities.

Ensure compliance with local laws and regulations when using this model and uphold ethical standards.

The developers assume no liability for any misuse of this model.

Important clarification regarding generated voices:

As a generative model, the voices produced by this system are synthetic outputs inferred by the model, not recordings of real human voices.

The generated voice characteristics do not represent or reproduce any specific real individual, and are not derived from or intended to imitate identifiable persons.

We advocate for the responsible development and use of AI and encourage the community to uphold safety and ethical principles in AI research and applications. 

## Contact Us
If you are interested in leaving a message to our work, feel free to email jingbin.hu@mail.nwpu.edu.cn or lxie@nwpu.edu.cn

You’re welcome to join our WeChat group for technical discussions, updates.
<p align="center">
  <!-- <em>Due to group limits, if you can't scan the QR code, please add my WeChat for group access  -->
      <!-- : <strong>Tiamo James</strong></em> -->
  <br>
  <span style="display: inline-block; margin-right: 10px;">
    <img src="assets/wechat_new.jpg" width="300" alt="WeChat Group QR Code"/>
  </span>
</p>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ASLP-lab/VoiceSculptor&type=date&legend=top-left)](https://www.star-history.com/#ASLP-lab/VoiceSculptor&type=date&legend=top-left)
