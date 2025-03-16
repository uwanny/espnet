# eval_script.py
import os
import json
import torch
import numpy as np
import copy
from typing import List, Dict, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from espnet2.bin.asr_inference import Speech2Text
from espnet2.bin.tts_inference import Text2Speech
import time
from contextlib import contextmanager
from typing import Optional, Tuple
import argparse
import numpy as np
import torch
from packaging.version import parse as V
from typeguard import typechecked
from espnet2.sds.asr.espnet_asr import ESPnetASRModel
from espnet2.sds.asr.owsm_asr import OWSMModel
from espnet2.sds.asr.owsm_ctc_asr import OWSMCTCModel
from espnet2.sds.asr.whisper_asr import WhisperASRModel
from espnet2.sds.end_to_end.mini_omni_e2e import MiniOmniE2EModel
from espnet2.sds.tts.chat_tts import ChatTTSModel
from espnet2.sds.tts.espnet_tts import ESPnetTTSModel
from espnet2.sds.utils.chat import Chat
from pyscripts.utils.dialog_eval.LLM_Metrics import (
    DialoGPT_perplexity,
    bert_score,
    perplexity,
    vert,
) 
from espnet2.sds.vad.webrtc_vad import WebrtcVADModel
from espnet2.train.abs_espnet_model import AbsESPnetModel

from typing import List
import json
import torch
from typeguard import typechecked

from espnet2.sds.llm.abs_llm import AbsLLM
import os
os.environ["HF_HOME"] = "/ocean/projects/cis210027p/jsunc/espnet_ml_superb2/egs2/spoken_chatbot_arena/sds1"
os.environ["TRANSFORMERS_CACHE"] = "/ocean/projects/cis210027p/jsunc/espnet_ml_superb2/egs2/spoken_chatbot_arena/sds1"

class HuggingFaceLLM(AbsLLM):
    """Hugging Face LLM"""

    @typechecked
    def __init__(
        self,
        access_token: str,
        tag: str = "meta-llama/Llama-3.2-1B-Instruct",
        device: str = "cuda",
        dtype: str = "float16",
    ):
        """
        A class for initializing a text response generator
        using the Transformers library.

        Args:
            access_token (str):
                The access token required for downloading models from Hugging Face.
            tag (str, optional):
                The model tag for the pre-trained language model.
                Defaults to "meta-llama/Llama-3.2-1B-Instruct".
            device (str, optional):
                The device to run the inference on. Defaults to "cuda".
            dtype (str, optional):
                The data type for model computation. Defaults to "float16".

        Raises:
            ImportError:
                If the `transformers` library is not installed.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except Exception as e:
            print(
                "`transformers` is not available. Please install it via `pip install"
                " transformers` or `cd /path/to/espnet/tools && . ./activate_python.sh"
                " && ./installers/install_transformers.sh`."
            )
            raise e
        super().__init__()
        LM_tokenizer = AutoTokenizer.from_pretrained(tag, token=access_token)
        LM_model = AutoModelForCausalLM.from_pretrained(
            tag, torch_dtype=dtype, trust_remote_code=True, token=access_token
        ).to(device)
        
         # ✅ **手动检查 `chat_template`，如果为空就赋值**
        if not hasattr(LM_tokenizer, "chat_template") or LM_tokenizer.chat_template is None:
            print(f"⚠️ Warning: Model `{tag}` has no chat_template. Using default template.")
            LM_tokenizer.chat_template = "{role}: {content}\n"
        else:
            print(f"current chat_template is {LM_tokenizer.chat_template}")    
        self.LM_pipe = pipeline(
            "text-generation", model=LM_model, tokenizer=LM_tokenizer, device=device
        )

    def warmup(self):
        """
        Perform a single forward pass with dummy input to
        pre-load and warm up the model.
        """
        with torch.no_grad():
            dummy_input_text = "Hello, how are you?"
            dummy_chat = [{"role": "user_A", "content": dummy_input_text}]
            self.LM_pipe(
                dummy_chat,
                max_new_tokens=32,
                min_new_tokens=0,
                temperature=0.0,
                do_sample=False,
            )

    def forward(self, chat_messages: List[dict]) -> str:
        """
        Generate a response from the language model based on the provided chat messages.

        Args:
            chat_messages (List[dict]):
                A list of chat messages, where each message is a
                dictionary containing the
                conversation history. Each dictionary should have
                keys like "role" (e.g., "user", "assistant")
                and "content" (the message text).

        Returns:
            str:
                The generated response text from the language model.

        Notes:
            - The model generates a response with a maximum of 64
            new tokens and a deterministic sampling strategy
            (temperature set to 0 and `do_sample` set to False).
        """
        with torch.no_grad():
            output = self.LM_pipe(
                chat_messages,
                max_new_tokens=64,
                min_new_tokens=0,
                temperature=0.0,
                do_sample=False,
            )
            generated_text = output[0]["generated_text"][-1]["content"]
            return generated_text


class LLMEvalInterface:
    """用于评估LLM在对话任务上的性能的接口"""

    def __init__(
        self,
        LLM_option: str,
        access_token: str = None,
    ):
        """
        Args:
            LLM_option (str): 要使用的LLM模型名称
            access_token (str): HuggingFace的token（如果需要的话）
        """
        self.LLM_option = LLM_option
        self.access_token = access_token
        self.LM_pipe = None
        self.chat = Chat(10)  # keep the conversation history
        # initialize the system prompt
        # self.chat.init_chat(
        #     {
        #         "role": "system",
        #         "content": "You are simulating a conversation between two people, you will be either of them", 
        #         "I will give you history chats between them, the format is like '{role: speaker_id, content: xxxxxx}, {role: speaker_id, content: xxxxxx},...'", 
        #         "you will simulate one of the people's response according to the context information", 
        #         "the person you will simulate will be provided in a format as {role: speaker_id, content: ,}, and you will generate content only",
        #     }
        # )
        self.chat.init_chat({"role": "system", "content": "You are simulating a conversation between two people"})
        # initialize the LLM model
        self.LM_pipe = HuggingFaceLLM(access_token=self.access_token, tag=LLM_option)
        self.LM_pipe.warmup()
        self.chat_history = []

    def process_dialog(self, dialog_lines: list):
        """处理一个完整的对话

        Args:
            dialog_lines: 对话数据列表，每行格式为"speaker_id utterance"

        Returns:
            responses: LLM针对每个用户输入的响应列表
            latencies: 每次响应的延迟时间列表
        """
        responses = []
        latencies = []
        # current_context = []

        for i, line in enumerate(dialog_lines):
            # parse the speaker ID and content
            line_header = line.split()[0]
            content = " ".join(line.split()[1:])

            if "A" in line_header:
                speaker_id = "user_A"
            elif "B" in line_header:
                speaker_id = "user_B"
            else:
                raise ValueError(f"Invalid line header: {line_header}")

            # add to the current context
            self.chat.append(
                {
                    "role": speaker_id,
                    "content": content,
                }
            )

            # find the next speaker id
            if i < len(dialog_lines) - 1:
                next_line_header = dialog_lines[i + 1].split()[0]
                if "A" in next_line_header:
                    next_speaker_id = "user_A"
                elif "B" in next_line_header:
                    next_speaker_id = "user_B"
                else:
                    raise ValueError(f"Invalid next line header: {next_line_header}")

                # generate the response according to next speaker id
                start_time = time.time()

                # build the conversation history
                self.chat.append(
                    {
                        "role": next_speaker_id,
                        "content": "",
                    }
                )
                chat_messages = self.chat.to_list()
                # print(f"current chat message is: {chat_messages}")
                self.chat_history.append(copy.deepcopy(chat_messages))
                # ✅ **存储 chat_messages 到日志**
                log_file = "llm_chat_messages_log.txt"
                with open(log_file, "a", encoding="utf-8") as log_f:
                    log_f.write(f"Chat Messages before LLM call (Turn {i}):\n")
                    for msg in chat_messages:
                        log_f.write(f"{msg['role']}: {msg['content']}\n")
                    log_f.write("=" * 80 + "\n")
                # generate the response
                response = self.LM_pipe(chat_messages)
                latency = time.time() - start_time

                responses.append(response)
                latencies.append(latency)
                self.chat.pop()
        
        # print(f"check the fist of chat_history: {self.chat_history[0]}")
        output_file = "chat_history.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.chat_history, f, indent=4, ensure_ascii=False)

        print(f"✅ chat_history 已保存至 {output_file}")
        return responses, latencies


    def evaluate_responses(self, responses: list, references: list):
        """
        评估 LLM 生成的回复质量。

        Args:
            responses (list): LLM 生成的回复列表
            references (list): 参考回复列表（实际对话中的回复）
            user_utterances (list): 对应的用户输入（用于 DialoGPT 计算对话连贯性）

        Returns:
            dict: 评估指标结果，包括 Perplexity、BERT Score、Diversity、DialoGPT Perplexity
        """
        metrics = {}
        
        def parse_vert_result(result: str):
            """
            解析 `vert()` 计算出的文本，提取 Self-BLEU2、Auto-BLEU2 和 VERT 指标。

            Args:
                result (str): `vert()` 返回的字符串

            Returns:
                dict: 提取的指标数值，例如：
                    {
                        "Self-BLEU2-geometric": 42.13,
                        "Auto-BLEU2-geometric": 38.94,
                        "VERT": 40.5
                    }
            """
            metrics = {}
            for line in result.strip().split("\n"):  # 按行拆分
                if ":" in line:  # 过滤掉空行
                    key, value = line.split(":")  # 拆分成 "指标名" 和 "数值"
                    metrics[key.strip()] = float(value.strip())  # 转换为浮点数
            return metrics


        # ✅ (1) **计算 Perplexity**
        print("🔹 计算 Perplexity ...")
        ppl_scores = [float(perplexity(resp.replace("\n", " ")).split(":")[1]) for resp in responses]
        metrics["perplexity"] = sum(ppl_scores) / len(ppl_scores)

        # ✅ (2) **计算 Diversity（VERT）**
        print("🔹 计算 Diversity(VERT) ...")
        vert_result = vert(responses)
        # metrics["diversity"] = float(vert_score.split("\n")[-2].split(":")[1])
        metrics["diversity"] = parse_vert_result(vert_result)

        # ✅ (3) **计算 BERT Score**
        print("🔹 计算 BERT Score ...")
        summary = 0
        print(f'check the first chat history: {self.chat_history[0]}')
        for chat, response in zip(self.chat_history, responses):
            chat = [json.dumps(d) for d in chat]
            total_response_arr = chat + [response]
            print(f"total_response_arr: {total_response_arr}")
            summary += float(bert_score(total_response_arr).split(":")[1])
        metrics["bert_score"] = summary / len(responses)

        # ✅ (4) **计算 DialoGPT Perplexity**
        summary = 0
        print("🔹 计算 DialoGPT Perplexity ...")
        for chat, response in zip(self.chat_history, responses):
            chat = [json.dumps(d) for d in chat]
            chat = ','.join(chat)
            print(f"user_utterance: {chat}, response: {response}")
            summary += float(DialoGPT_perplexity(chat, response).split(":")[1])
        metrics["dialogpt_perplexity"] = summary / len(responses)

        return metrics



parser = argparse.ArgumentParser(description="Run LLM evaluation with a specified model.")
parser.add_argument("--model", type=str, required=True, help="Hugging Face model name")
parser.add_argument("--metrics_file", type=str, required=True, help="Metric file name")

args = parser.parse_args()



# read the dialog data
dialog_lines = []
with open("sorted_dialog.txt", "r") as f:
    dialog_lines = f.readlines()




# initialize the evaluator
evaluator = LLMEvalInterface(
    LLM_option=args.model,
    access_token="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",  # 如果需要的话设置token
)

# process the dialog and get the responses
responses, latencies = evaluator.process_dialog(dialog_lines)

# get the reference responses (the responses from B in the actual dialog)
references = [line.split(" ", 1)[1] for line in dialog_lines]
references = references[1:]
print(f"len(responses): {len(responses)}")
print(f"len(references): {len(references)}")
print(f"len(dialog_lines) - 1: {len(dialog_lines) - 1}")
# **写入对比文件**
output_file = "generated_vs_reference.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write("Reference Response\tGenerated Response\n")
    f.write("=" * 80 + "\n")
    
    for ref, resp in zip(references, responses):
        f.write(f"Reference: {ref.strip()}\n")
        f.write(f"Generated: {resp.strip()}\n")
        f.write("-" * 80 + "\n")

print(f"✅ Responses saved to {output_file}")

assert len(responses) == len(references) == len(dialog_lines) - 1

# evaluate the generated responses
metrics = evaluator.evaluate_responses(responses, references)

# print the evaluation results
# print("Evaluation results:")
# print(f"Average perplexity: {metrics['perplexity']:.2f}")
# print(f"BERT score: {metrics['bert_score']:.2f}")
# print(f"Diversity: {metrics['diversity']:.2f}")
# print(f"Average response time: {sum(latencies)/len(latencies):.2f} seconds")
metrics_file = args.metrics_file
with open(metrics_file, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

print(f"✅ Evaluation metrics saved to {metrics_file}")

