# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments

from transformers.loss.loss_utils import ForCausalLMLoss

logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # Configure FP8 environment if enabled
        if model_args is not None and model_args.fp8:
            configure_fp8_environment(model_args)
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        # Verify FP8 status after trainer initialization (accelerator should be available)
        if model_args is not None and model_args.fp8 and hasattr(self, "accelerator"):
            verify_fp8_status(self.accelerator, model_args)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):

        if self.finetuning_args.freeze_memory_grad:
            memory_attention_mask = inputs.pop("memory_attention_mask") # (B, N_mem, max_mem_len)
            memory_input_ids = inputs.pop("memory_input_ids") # (B, N_mem, max_mem_len)

            memory_attn_sum = memory_attention_mask.sum(dim=2) # (B, N_mem)
            valid_memory_mask = memory_attn_sum > 0 # (B, N_mem)

            valid_memory_input_ids = memory_input_ids[valid_memory_mask] # (N_valid, max_mem_len)
            valid_memory_attention_mask = memory_attention_mask[valid_memory_mask] # (N_valid, max_mem_len)
            print(f"valid_memory_input_ids: {valid_memory_input_ids.shape}; valid_memory_attention_mask: {valid_memory_attention_mask.shape}")

            # if not valid_memory_mask.any():
                

            with torch.no_grad():
                valid_memory_embeds = model(
                    memory_input_ids=valid_memory_input_ids,
                    memory_attention_mask=valid_memory_attention_mask,
                    do_encoding=True,
                )
                valid_memory_embeds = valid_memory_embeds.detach()

            inputs["memory_embeds"] = valid_memory_embeds

        return super().compute_loss(model, inputs, *args, **kwargs)
        # batch_task_types = inputs.pop("task_type", None)
        # kwargs['return_outputs'] = True
        # loss, outputs = super().compute_loss(model, inputs, *args, **kwargs)

        # logits = outputs.logits          # (B, L, V)
        # labels = inputs["labels"]        # (B, L)
        # B, L, V = logits.shape
        # device = labels.device

        # IGNORE_INDEX = -100
        # labels = nn.functional.pad(labels, (0, 1), value=IGNORE_INDEX)
        # shift_labels = labels[..., 1:].contiguous()

        # ce_per_token = F.cross_entropy(
        #     logits.float().reshape(-1, V),   # (B*Ls, V)
        #     shift_labels.reshape(-1),              # (B*Ls,)
        #     ignore_index=IGNORE_INDEX,
        #     reduction="none"
        # ).view(B, L)   

        # valid_mask = (shift_labels != IGNORE_INDEX)                # (B, L)
        # valid_cnt = valid_mask.sum(dim=1).clamp_min(1)             # (B,)

        # # 3) 每样本平均（样本内部按有效 token 平均）
        # ce_sum_per_sample = (ce_per_token * valid_mask).sum(dim=1)         # (B,)
        # loss_per_sample_mean = ce_sum_per_sample / valid_cnt               # (B,)
        # loss = loss_per_sample_mean.mean()                                 # 标量

        # # 4) 计算 hard 窗口（基于 shift_labels 的坐标系，长度 L）
        # pos = torch.arange(L, device=device).expand(B, -1)
        # has_label = valid_mask.any(dim=1)                                   # (B,)

        # # 为 min/max 准备哨兵
        # masked_for_min = torch.where(valid_mask, pos, torch.full_like(pos, L))
        # masked_for_max = torch.where(valid_mask, pos, torch.full_like(pos, -1))

        # first_pos = masked_for_min.min(dim=1).values      # ∈ [0..L] (L 表示无有效标签)
        # last_pos  = masked_for_max.max(dim=1).values      # ∈ [-1..L-1] (-1 表示无有效标签)

        # hard_left_offset = 10
        # hard_right_offset = 3
        # hard_start = first_pos + hard_left_offset
        # hard_end   = last_pos  - hard_right_offset

        # # 对有标签样本 clamp；无标签保持哨兵
        # hard_start = torch.where(has_label, hard_start.clamp(0, L - 1), first_pos)
        # hard_end   = torch.where(has_label, hard_end.clamp(-1, L - 1), last_pos)

        # # hard 有效性
        # window_valid = (hard_start <= hard_end) & has_label             # (B,)
        # ar = torch.arange(L, device=device).expand(B, -1)
        # hard_mask = (ar >= hard_start[:, None]) & (ar <= hard_end[:, None]) & window_valid[:, None]  # (B, L)

        # # hard 段每样本平均（只对 hard_mask 内做均值）
        # hard_cnt = hard_mask.sum(dim=1)                                 # (B,)
        # # 避免 0 除：只在 window_valid==True 的样本上读取
        # hard_ce_sum_per_sample = (ce_per_token * hard_mask).sum(dim=1)  # (B,)
        # hard_ce_per_sample_mean = torch.zeros(B, device=device)
        # valid_idx = window_valid.nonzero(as_tuple=False).squeeze(-1)
        # if valid_idx.numel() > 0:
        #     hard_ce_per_sample_mean[valid_idx] = (
        #         hard_ce_sum_per_sample[valid_idx] / hard_cnt[valid_idx].clamp_min(1)
        #     )

        # # 5) 聚合输出
        # task_type_to_loss = defaultdict(list)
        # # 常规：按 task_type 存每样本平均
        # for i, tt in enumerate(batch_task_types):
        #     if tt is not None:
        #         task_type_to_loss[tt].append(loss_per_sample_mean[i].detach().cpu().item())

        # # policy_baseline_hard：只对有效 hard 窗口样本存
        # for i, tt in enumerate(batch_task_types):
        #     if tt == "policy_baseline" and window_valid[i]:
        #         task_type_to_loss["policy_action"].append(
        #             hard_ce_per_sample_mean[i].detach().cpu().item()
        #         )

        # return loss, task_type_to_loss


    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")

