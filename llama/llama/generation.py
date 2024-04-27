# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer


Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
    ) -> "Llama":
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(1)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        self.h = None 
        self.h_list = None
        self.h_last_real_token_bl = None # The term 'b' represents the simplification of calculations using the smallest number of tokens in a batch.
        self.h_ave_real_token_bl = None
        self.h_last_real_token_s = None # The symbol 's' represents the precise calculation using the actual number of tokens from each text.
        self.h_ave_real_token_s = None
        self.h_pca_real_token_s = None
        self.h_eis_ave_real_token_s = None
        self.h_mat_real_token_s = None
        self._prompt_tokens = None # initialization
        self.min_prompt_len = 0
        self.h_last_real_token_sl = []
        self.h_ave_real_token_sl = []
        self.h_pca_real_token_sl = []
        self.h_eis_ave_real_token_sl = []
        self.h_mat_real_token_sl = []

        self.previous_chunk_length = 0

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        self.min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        for cur_pos in range(self.min_prompt_len, total_len):
            logits, self.h_last_real_token_bl, self.h_ave_real_token_bl, self.h_list = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos) # Obtain both the output of the Transformer model and the text embedding vectors from the final layer.
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        self._prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts] # Define `prompt_tokens` as a protected member variable of the class to facilitate access.
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=self._prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]

        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]

    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        self._prompt_tokens = []
        unsafe_requests = []
        for dialog in dialogs:
            unsafe_requests.append(
                any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
            )
            if dialog[0]["role"] == "system":
                dialog = [
                    {
                        "role": dialog[1]["role"],
                        "content": B_SYS
                        + dialog[0]["content"]
                        + E_SYS
                        + dialog[1]["content"],
                    }
                ] + dialog[2:]
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )
            dialog_tokens: List[int] = sum(
                [
                    self.tokenizer.encode(
                        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                        bos=True,
                        eos=True,
                    )
                    for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ],
                [],
            )
            assert (
                dialog[-1]["role"] == "user"
            ), f"Last message must be from user, got {dialog[-1]['role']}"
            dialog_tokens += self.tokenizer.encode(
                f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
                bos=True,
                eos=False,
            )
            self._prompt_tokens.append(dialog_tokens)

        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=self._prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        if logprobs:
            return [
                {
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.decode(t)
                        if not unsafe
                        else UNSAFE_ERROR,
                    },
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i, unsafe in zip(
                    generation_tokens, generation_logprobs, unsafe_requests
                )
            ]
        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": self.tokenizer.decode(t) if not unsafe else UNSAFE_ERROR,
                }
            }
            for t, unsafe in zip(generation_tokens, unsafe_requests)
        ]
    
    def pca_torch(self, matrix, k):
        """
        Compute PCA using PyTorch and return the transformed matrix.
        """
        # Center the data
        matrix_mean = torch.mean(matrix, dim=0)
        matrix = matrix - matrix_mean.unsqueeze(0)

        # Compute the covariance matrix
        cov_matrix = matrix.t().mm(matrix) / matrix.size(0)

        # Compute the eigenvectors and eigenvalues
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

        # Sort eigenvectors by eigenvalues in descending order
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # Transform the original matrix
        transformed_matrix = matrix.mm(eigenvectors[:, :k])

        return transformed_matrix
    
    def matrix_to_vector_pca_rescaled(self, matrix):
        matrix = matrix.to(dtype=torch.float32)

        a, b = matrix.size()
        pca_result = self.pca_torch(matrix, k=b)
        
        # Calculate the mean of the PCA results.
        pca_mean = pca_result.mean(dim=0)

        # Retrieve the maximum and minimum values of the original matrix.
        original_min = matrix.min()
        original_max = matrix.max()
        
        # Obtain the maximum and minimum values of the PCA mean.
        pca_mean_min = pca_mean.min()
        pca_mean_max = pca_mean.max()
        
        # Scale the mean of PCA to match the range of the original matrix.
        scale_factor = max(original_max / pca_mean_max, original_min / pca_mean_min)
        pca_rescaled_vector = pca_mean * scale_factor

        # transfer to float16
        pca_rescaled_vector = pca_rescaled_vector.to(dtype=torch.float16)
        
        return pca_rescaled_vector

    def get_current_chunk(self):
        h_list_current_chunk = self.h_list[self.previous_chunk_length:]
        self.previous_chunk_length = len(self.h_list)

        return h_list_current_chunk
    
    # The number of tokens contained in the first tensor output of a text completion is determined by truncating to the length of the shortest sentence in the batch; the batch size corresponds to the number of input sentences within the batch.
    # text completion输出的第一个tensor包含的tokens数是按照batch中最短的句子截断统一输出的；batchsize是batch中输入句子的数量
    def get_text_prompt_token_embedding(self):
        # print(len(self.h_list))
        # 同一个batch中的sentence是按照batch_idx纵向衔接在h_list中的，而同一个chunk中的sentence是直接横向衔接在h_list中的. 因此先横向选择正确的chunk.
        # In the same batch, sentences are vertically concatenated in the h_list according to their batch index, while sentences within the same chunk are horizontally concatenated directly in the h_list. Therefore, the correct chunk is initially selected horizontally.
        h_list_current_chunk = self.get_current_chunk()
        # print(len(h_list_current_chunk))

        # print(self._prompt_tokens)
        for batch_idx, pt in enumerate(self._prompt_tokens): # print tokens number
            if len(pt) == self.min_prompt_len:
                prompt_pos = 0
                h_ave_real_token_s_cat = h_list_current_chunk[prompt_pos][batch_idx][:]
            elif len(pt) > self.min_prompt_len:
                prompt_pos = len(pt) - self.min_prompt_len
                h_ave_real_token_s_cat = h_list_current_chunk[0][batch_idx][:]
                temp_prompt_pos = 1
                for temp_prompt_pos in range(1, prompt_pos+1):
                    temp_prompt = h_list_current_chunk[temp_prompt_pos][batch_idx][:]
                    h_ave_real_token_s_cat = torch.cat((h_ave_real_token_s_cat, temp_prompt), dim=0)
            # print('h_last/ave/pca/eis/mat_real_token_s_cat in get_text_prompt_token_embedding')
            # print(h_ave_real_token_s_cat.shape) # [t, d] t: number of tokens, d: dimension of llama output
            # print(h_ave_real_token_s_cat.dtype) # torch.float16
            # print(h_ave_real_token_s_cat) # This matrix represents the token embedding matrix for all inputs.

            # Only extract the last real token from each transcript.
            # 只取每个文本的最后一个real token
            self.h_last_real_token_s = h_list_current_chunk[prompt_pos][batch_idx][-1]  
            self.h_last_real_token_sl.append(self.h_last_real_token_s)
            self.h_last_real_token_sl_tensor = torch.stack(self.h_last_real_token_sl, dim=0)

            # Calculate the average along the first dimension (all tokens of the same text).
            # 沿着第一个维度(同一个文本的所有tokens)求平均
            self.h_ave_real_token_s = torch.mean(h_ave_real_token_s_cat, dim=0) 
            self.h_ave_real_token_sl.append(self.h_ave_real_token_s)
            self.h_ave_real_token_sl_tensor = torch.stack(self.h_ave_real_token_sl, dim=0)
            # This line below can be included or omitted. If included, outputs will be generated one by one. Be sure to adjust the output position in the example accordingly (the method for extracting other tokens is similar).
            # 以下这行可加可不加，加的话就一个一个输出了，注意配合着调整example中的输出位置（其它token取法同理）
            # self.h_ave_real_token_sl = [] 

            # Use the PCA method to perform principal component analysis on all tokens from the same text, reducing dimensions to a [d]-dimensional shape, and scale the values back to their original value scale.
            # 用pca方法取同一个文本中所有tokens的主成分降维至[d]的形状，并把数值缩放到原始大小规模
            self.h_pca_real_token_s = self.matrix_to_vector_pca_rescaled(h_ave_real_token_s_cat) 
            self.h_pca_real_token_sl.append(self.h_pca_real_token_s)
            self.h_pca_real_token_sl_tensor = torch.stack(self.h_pca_real_token_sl, dim=0)
            
            # Use the method of 'by tokens' to obtain the embeddings for all tokens.
            # 用by tokens的方法，取所有tokens的embedding
            self.h_mat_real_token_s = h_ave_real_token_s_cat
            self.h_mat_real_token_sl.append(self.h_mat_real_token_s)
            
        return self.h_last_real_token_bl, self.h_ave_real_token_bl, self.h_last_real_token_sl_tensor, self.h_ave_real_token_sl_tensor, self.h_pca_real_token_sl_tensor, self.h_mat_real_token_sl

    # The first tensor output from the chat completion contains the number of tokens for the actual length of each prompt in the batch; batch size refers to the number of sentences in a single conversation.
    # chat completion输出的第一个tensor包含的tokens数量是batch中每个prompt的实际长度；batchsize是一次对话中的句子数量
    def get_chat_prompt_token_embedding(self):
        # 取每个文本的emotion，intention和speaking style的均值作为输出
        self.h_eis_ave_real_token_s = torch.mean(self.h_list[-1], dim=0).squeeze(0)
        self.h_eis_ave_real_token_sl.append(self.h_eis_ave_real_token_s)
        self.h_eis_ave_real_token_sl_tensor = torch.stack(self.h_eis_ave_real_token_sl, dim=0)
        return self.h_eis_ave_real_token_sl_tensor

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token