# Copyright 2025 the LlamaFactory team.
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

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, greedy_knapsack, infer_seqlen


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)

from copy import deepcopy


@dataclass
class SupervisedDatasetProcessor(DatasetProcessor):
    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> tuple[list[int], list[int]]:
        messages = self.template.mm_plugin.process_messages(prompt + response, images, videos, audios, self.processor)
        input_ids, labels = self.template.mm_plugin.process_token_ids(
            [], [], images, videos, audios, self.tokenizer, self.processor
        )
        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools)
        total_length = len(input_ids) + (1 if self.template.efficient_eos else 0)
        if self.data_args.mask_history:
            encoded_pairs = encoded_pairs[::-1]  # high priority for last turns

        for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
            if total_length >= self.data_args.cutoff_len:
                break

            source_len, target_len = infer_seqlen(
                len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length
            )
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            total_length += source_len + target_len

            if self.data_args.train_on_prompt:
                source_label = source_ids
            elif self.template.efficient_eos and turn_idx != 0:
                source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
            else:
                source_label = [IGNORE_INDEX] * source_len

            if self.data_args.mask_history and turn_idx != 0:  # train on the last turn only
                target_label = [IGNORE_INDEX] * target_len
            else:
                target_label = target_ids

            if self.data_args.mask_history:  # reversed sequences
                input_ids = source_ids + target_ids + input_ids
                labels = source_label + target_label + labels
            else:
                input_ids += source_ids + target_ids
                labels += source_label + target_label

        if self.template.efficient_eos:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]

        return input_ids, labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])
            model_inputs["task_type"].append(examples["_task_type"][i])

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        # print("input_ids:\n{}".format(example["input_ids"]))
        if len(example["input_ids"]) > 100:
            print(f"input_ids:\n [{example['input_ids'][:5]}, ..., {example['input_ids'][-5:]}]; total {len(example['input_ids'])}")
        else:
            print(f"input_ids:\n {example['input_ids']}; total {len(example['input_ids'])}")
        decoded_input = self.tokenizer.decode(example['input_ids'], skip_special_tokens=False)
        if len(decoded_input) > 100:
            print(f"inputs:\n [{decoded_input[:5]}, ..., {decoded_input[-5:]}]; total length{len(decoded_input)}")
        else:
            print(f"inputs:\n {decoded_input}; total length {len(decoded_input)}")

        if len(example['labels']) > 100:
            print(f"label_ids:\n [{example['labels'][:5]}, ..., {example['labels'][-5:]}]; total {len(example['labels'])}")
        else:
            print(f"label_ids:\n {example['labels']}; total {len(example['labels'])}")
        if len(valid_labels) > 100:
            print(f"valid_labels:\n [{valid_labels[:5]}, ..., {valid_labels[-5:]}]; total {len(valid_labels)}")
        else:
            print(f"valid_labels:\n {valid_labels}; total {len(valid_labels)}")
        decoded_labels = self.tokenizer.decode(valid_labels, skip_special_tokens=False)
        if len(decoded_labels) > 100:
            print(f"labels:\n [{decoded_labels[:5]}, ..., {decoded_labels[-5:]}]; total length {len(decoded_labels)}")
        else:
            print(f"labels:\n {decoded_labels}; total length {len(decoded_labels)}")

@dataclass
class LegacySupervisedDatasetProcessorWithMemory(SupervisedDatasetProcessor):
    """Legacy single-turn memory processor. Use MultiTurnSupervisedDatasetProcessorWithMemory instead."""
    mem_pad_token_id: int = -1
    num_query_tokens: int = 1

    def __post_init__(self):
        self.mem_pad_token_id = self.tokenizer.convert_tokens_to_ids("<|mem_pad|>")
        # Detect num_query_tokens by applying chat template with a single memory
        test_msg = [{'role': 'user', 'content': [
            {'type': 'memory_text', 'memory_text': {'text': 'test'}, 'is_memory': True}
        ]}]
        test_ids = self.tokenizer.apply_chat_template(test_msg, add_generation_prompt=True)
        self.num_query_tokens = sum(1 for tid in test_ids if tid == self.mem_pad_token_id)
        logger.info_rank0(f"Detected num_query_tokens={self.num_query_tokens} per memory")

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            # Check max_memory_num constraint
            memory_texts_check = examples.get("_memory", [None])[i] or []
            max_memory_num = getattr(self.data_args, 'max_memory_num', 10000)
            if max_memory_num is None:
                max_memory_num = 10000
            num_memory_to_drop = len(memory_texts_check) - max_memory_num

            # strange aligned
            prompt_content_list = []
            current_mem_num = 0
            for cnt_item in examples["_prompt"][i][0]['content']:
                if isinstance(cnt_item, dict):
                    if cnt_item.get('type', '') == 'text':
                        # del cnt_item['memory_text']
                        prompt_content_list.append({'type': 'text', 'text': cnt_item['text']})
                    elif cnt_item.get('type', '') == 'memory_text':
                        current_mem_num += 1
                        if current_mem_num <= num_memory_to_drop:
                            continue
                        else:
                            if cnt_item['is_memory'] is None:
                                is_memory = True
                            else:
                                is_memory = cnt_item['is_memory']
                            prompt_content_list.append({'type': 'memory_text', 'memory_text': {'text': cnt_item['memory_text']['text']}, 'is_memory': is_memory})
                else:
                    prompt_content_list.append(cnt_item)
            copied_prompt = [{'role': 'user', 'content': prompt_content_list}]
            if len(examples["_system"][i]) > 0:
                copied_prompt = [{'role': 'system', 'content': examples["_system"][i]}] + copied_prompt
            input_ids = self.tokenizer.apply_chat_template(
                copied_prompt + examples["_response"][i],
            )
            source_len = len(self.tokenizer.apply_chat_template(
                copied_prompt,
                add_generation_prompt=True,
            ))
            labels = [IGNORE_INDEX] * source_len + input_ids[source_len:]

            # Apply left truncation if sequence exceeds cutoff_len
            cutoff_len = self.data_args.cutoff_len
            num_truncated_memories = 0
            if len(input_ids) > cutoff_len:
                # Left truncation: keep the rightmost (most recent) tokens
                truncate_len = len(input_ids) - cutoff_len
                # Count how many mem_pad tokens are in the truncated portion
                truncated_mem_pad_count = sum(1 for tid in input_ids[:truncate_len] if tid == self.mem_pad_token_id)
                # Calculate how many memories were truncated (including partial ones)
                # If a memory is partially truncated, we must discard it entirely
                num_truncated_memories = (truncated_mem_pad_count + self.num_query_tokens - 1) // self.num_query_tokens  # ceiling division
                input_ids = input_ids[truncate_len:]
                labels = labels[truncate_len:]

            # Count valid memory placeholders in truncated sequence
            remaining_mem_pad_count = sum(1 for token_id in input_ids if token_id == self.mem_pad_token_id)
            # Number of complete memories remaining (floor division - only count complete ones)
            valid_mem_num = remaining_mem_pad_count // self.num_query_tokens

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)

            # Encode memory texts for this sample
            memory_texts = examples.get("_memory", [None])[i] or []
            # Truncate memory num before encoding: keep the rightmost (most recent) memories
            if num_memory_to_drop > 0:
                memory_texts = memory_texts[num_memory_to_drop:]
            # Drop memories that were truncated from the left (including partially truncated)
            if num_truncated_memories > 0:
                memory_texts = memory_texts[num_truncated_memories:]
            # Ensure alignment: keep only as many memories as we have complete placeholders
            if len(memory_texts) > valid_mem_num:
                memory_texts = memory_texts[:valid_mem_num]

            memory_input_ids = []
            memory_attention_mask = []
            memory_truncate_length = getattr(self.data_args, 'memory_truncate_length', 1024) or 1024
            for mem_text in memory_texts:
                m_ids = self.tokenizer.encode(mem_text, add_special_tokens=False)
                # Apply left truncation to memory: keep the rightmost (most recent) tokens
                if len(m_ids) > memory_truncate_length:
                    m_ids = m_ids[-memory_truncate_length:]
                memory_input_ids.append(m_ids)
                memory_attention_mask.append([1] * len(m_ids))

            model_inputs["memory_input_ids"].append(memory_input_ids)
            model_inputs["memory_attention_mask"].append(memory_attention_mask)
            model_inputs["task_type"].append(examples["_task_type"][i])
        return model_inputs

@dataclass
class MultiTurnSupervisedDatasetProcessorWithMemory(SupervisedDatasetProcessor):
    """Processor for V4 multi-turn format where each round is a separate user/assistant turn.

    This processor handles multi-turn conversations with memory, properly masking
    each user turn and training on each assistant turn.
    """
    mem_pad_token_id: int = -1
    num_query_tokens: int = 1

    def __post_init__(self):
        self.mem_pad_token_id = self.tokenizer.convert_tokens_to_ids("<|mem_pad|>")
        # Detect num_query_tokens by applying chat template with a single memory
        test_msg = [{'role': 'user', 'content': [
            {'type': 'memory_text', 'memory_text': {'text': 'test'}, 'is_memory': True}
        ]}]
        test_ids = self.tokenizer.apply_chat_template(test_msg, add_generation_prompt=True)
        self.num_query_tokens = sum(1 for tid in test_ids if tid == self.mem_pad_token_id)
        logger.info_rank0(f"Detected num_query_tokens={self.num_query_tokens} per memory")

    def _process_user_content(self, content: list, num_memory_to_drop: int, current_mem_count: int) -> tuple[list, int]:
        """Process user message content, handling memory_text items.

        Args:
            content: List of content items from user message
            num_memory_to_drop: Number of memories to skip (for max_memory_num constraint)
            current_mem_count: Current count of memories processed so far

        Returns:
            Tuple of (processed content list, updated memory count)
        """
        if not isinstance(content, list):
            return content, current_mem_count

        processed_content = []
        for cnt_item in content:
            if isinstance(cnt_item, dict):
                if cnt_item.get('type', '') == 'text':
                    processed_content.append({'type': 'text', 'text': cnt_item['text']})
                elif cnt_item.get('type', '') == 'memory_text':
                    current_mem_count += 1
                    if current_mem_count <= num_memory_to_drop:
                        continue
                    else:
                        is_memory = cnt_item.get('is_memory')
                        if is_memory is None:
                            is_memory = True
                        processed_content.append({
                            'type': 'memory_text',
                            'memory_text': {'text': cnt_item['memory_text']['text']},
                            'is_memory': is_memory
                        })
            else:
                processed_content.append(cnt_item)

        return processed_content, current_mem_count

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """Preprocess multi-turn dataset with memory.

        For multi-turn conversations:
        - Each user turn is masked (IGNORE_INDEX)
        - Each assistant turn is trained on
        - Memory texts are extracted and encoded separately
        """
        model_inputs = defaultdict(list)
        # from ...debug_utils import wait_for_debugger
        # wait_for_debugger()
        for i in range(len(examples["_prompt"])):
            prompt = examples["_prompt"][i]
            response = examples["_response"][i]

            # Validate: prompt should have odd number of messages (user, assistant, ..., user)
            # and response should have exactly 1 message (the final assistant response)
            if len(prompt) % 2 != 1 or len(response) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(prompt + response)
                )
                continue

            # Check max_memory_num constraint
            memory_texts_check = examples.get("_memory", [None])[i] or []
            max_memory_num = getattr(self.data_args, 'max_memory_num', 10000)
            if max_memory_num is None:
                max_memory_num = 10000
            num_memory_to_drop = max(0, len(memory_texts_check) - max_memory_num)

            # Build the full conversation with processed content
            messages = []
            system = examples["_system"][i]
            if system:
                messages.append({'role': 'system', 'content': system})

            current_mem_count = 0
            # Process prompt messages (alternating user/assistant)
            for msg in prompt:
                if msg['role'] == 'user':
                    processed_content, current_mem_count = self._process_user_content(
                        msg['content'], num_memory_to_drop, current_mem_count
                    )
                    messages.append({'role': 'user', 'content': processed_content})
                else:
                    messages.append({'role': 'assistant', 'content': msg['content']})

            # Process final response
            messages.append({'role': 'assistant', 'content': response[0]['content']})

            # Tokenize the full conversation
            input_ids = self.tokenizer.apply_chat_template(messages)

            # Compute labels by masking user turns and keeping assistant turns
            # We need to tokenize incrementally to find boundaries
            # IMPORTANT: Reuse the already-processed `messages` list to ensure consistency
            labels = []
            current_pos = 0

            # Process each message in the already-built messages list
            # Count total assistant turns for only_predict_last_turn
            only_predict_last_turn = getattr(self.data_args, 'only_predict_last_turn', False)
            if only_predict_last_turn:
                total_assistant_turns = sum(1 for msg in messages if msg['role'] == 'assistant')
                current_assistant_turn = 0

            conversation_so_far = []
            for msg in messages:
                conversation_so_far.append(msg)

                if msg['role'] == 'system' or msg['role'] == 'user':
                    # Get position after this message (with generation prompt for user/system)
                    tokens_so_far = self.tokenizer.apply_chat_template(
                        conversation_so_far, add_generation_prompt=True
                    )
                    new_pos = len(tokens_so_far)
                    # Mask system/user turn
                    labels.extend([IGNORE_INDEX] * (new_pos - current_pos))
                    current_pos = new_pos

                else:  # assistant
                    # Get position after this assistant message
                    tokens_so_far = self.tokenizer.apply_chat_template(conversation_so_far)
                    new_pos = len(tokens_so_far)

                    if only_predict_last_turn:
                        current_assistant_turn += 1
                        if current_assistant_turn < total_assistant_turns:
                            # Mask non-last assistant turns
                            labels.extend([IGNORE_INDEX] * (new_pos - current_pos))
                        else:
                            # Train on last assistant turn only
                            labels.extend(input_ids[current_pos:new_pos])
                    else:
                        # Train on all assistant turns
                        labels.extend(input_ids[current_pos:new_pos])
                    current_pos = new_pos

            # Apply left truncation if sequence exceeds cutoff_len
            cutoff_len = self.data_args.cutoff_len
            num_truncated_memories = 0
            if len(input_ids) > cutoff_len:
                truncate_len = len(input_ids) - cutoff_len
                # Count how many mem_pad tokens are in the truncated portion
                truncated_mem_pad_count = sum(1 for tid in input_ids[:truncate_len] if tid == self.mem_pad_token_id)
                # Calculate how many memories were truncated (including partial ones)
                # If a memory is partially truncated, we must discard it entirely
                num_truncated_memories = (truncated_mem_pad_count + self.num_query_tokens - 1) // self.num_query_tokens  # ceiling division
                input_ids = input_ids[truncate_len:]
                labels = labels[truncate_len:]

            # Count valid memory placeholders in truncated sequence
            remaining_mem_pad_count = sum(1 for token_id in input_ids if token_id == self.mem_pad_token_id)
            # Number of complete memories remaining (floor division - only count complete ones)
            valid_mem_num = remaining_mem_pad_count // self.num_query_tokens

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)

            # Encode memory texts
            memory_texts = examples.get("_memory", [None])[i] or []
            if num_memory_to_drop > 0:
                memory_texts = memory_texts[num_memory_to_drop:]
            # Drop memories that were truncated from the left (including partially truncated)
            if num_truncated_memories > 0:
                memory_texts = memory_texts[num_truncated_memories:]
            # Ensure alignment: keep only as many memories as we have complete placeholders
            if len(memory_texts) > valid_mem_num:
                memory_texts = memory_texts[:valid_mem_num]

            memory_input_ids = []
            memory_attention_mask = []
            memory_truncate_length = getattr(self.data_args, 'memory_truncate_length', 1024) or 1024
            for mem_text in memory_texts:
                m_ids = self.tokenizer.encode(mem_text, add_special_tokens=False)
                if len(m_ids) > memory_truncate_length:
                    m_ids = m_ids[-memory_truncate_length:]
                memory_input_ids.append(m_ids)
                memory_attention_mask.append([1] * len(m_ids))

            model_inputs["memory_input_ids"].append(memory_input_ids)
            model_inputs["memory_attention_mask"].append(memory_attention_mask)
            model_inputs["task_type"].append(examples["_task_type"][i])
            

        return model_inputs


@dataclass
class PackedSupervisedDatasetProcessor(SupervisedDatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # TODO: use `position_ids` to achieve packing
        # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
        # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
        valid_num = 0
        batch_input_ids, batch_labels, batch_images, batch_videos, batch_audios = [], [], [], [], []
        lengths = []
        length2indexes = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            length = len(input_ids)
            if length > self.data_args.cutoff_len:
                logger.warning_rank0(f"Dropped lengthy example with length {length} > {self.data_args.cutoff_len}.")
            else:
                lengths.append(length)
                length2indexes[length].append(valid_num)
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                batch_images.append(examples["_images"][i] or [])
                batch_videos.append(examples["_videos"][i] or [])
                batch_audios.append(examples["_audios"][i] or [])
                valid_num += 1

        model_inputs = defaultdict(list)
        knapsacks = greedy_knapsack(lengths, self.data_args.cutoff_len)
        for knapsack in knapsacks:
            packed_input_ids, packed_attention_masks, packed_position_ids, packed_labels = [], [], [], []
            packed_images, packed_videos, packed_audios = [], [], []
            for i, length in enumerate(knapsack):
                index = length2indexes[length].pop()
                packed_input_ids += batch_input_ids[index]
                packed_position_ids += list(range(len(batch_input_ids[index])))  # NOTE: pad_to_multiple_of ignore this
                packed_labels += batch_labels[index]
                packed_images += batch_images[index]
                packed_videos += batch_videos[index]
                packed_audios += batch_audios[index]
                if self.data_args.neat_packing:
                    packed_attention_masks += [i + 1] * len(batch_input_ids[index])  # start from 1
                else:
                    packed_attention_masks += [1] * len(batch_input_ids[index])

            if len(packed_input_ids) < self.data_args.cutoff_len + 1:  # avoid flash_attn drops attn mask
                pad_length = self.data_args.cutoff_len - len(packed_input_ids) + 1
                packed_input_ids += [self.tokenizer.pad_token_id] * pad_length
                packed_position_ids += [0] * pad_length
                packed_labels += [IGNORE_INDEX] * pad_length
                if self.data_args.neat_packing:
                    packed_attention_masks += [0] * pad_length
                else:
                    packed_attention_masks += [1] * pad_length  # more efficient flash_attn

            if len(packed_input_ids) != self.data_args.cutoff_len + 1:
                raise ValueError("The length of packed example should be identical to the cutoff length.")

            model_inputs["input_ids"].append(packed_input_ids)
            model_inputs["attention_mask"].append(packed_attention_masks)
            model_inputs["position_ids"].append(packed_position_ids)
            model_inputs["labels"].append(packed_labels)
            model_inputs["images"].append(packed_images or None)
            model_inputs["videos"].append(packed_videos or None)
            model_inputs["audios"].append(packed_audios or None)

        return model_inputs
