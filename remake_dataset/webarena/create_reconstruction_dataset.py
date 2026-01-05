"""
Create reconstruction training dataset from WebArena trajectories.

训练目标：
- Action 样本: o_k + z_k + z_{0..k-1} + Task → Action_k
- 重建样本: o_k + z_k + z_i → o_i (打散的 i, k 组合)

混合训练让 encoder 学会保留语义信息，而不是 action 捷径。
"""

import json
import random
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any
from transformers import AutoTokenizer
import argparse

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
MEMORY_TRUNCATE_LENGTH = 4096


# ============== System Prompts ==============

ACTION_SYSTEM_PROMPT = """# Setup
You are a professional web browsing agent assistant that can fulfill user's high-level instructions.
Given the compressed tokens of the browsed webpage at each step, you plan operations in python-style pseudo code using provided functions, or customize functions (if necessary) and then provide their implementations.
# More details about the code
Your code should be readable, simple, and only **ONE-LINE-OF-CODE** at a time, avoid using loop statement and only use if-else control if necessary. Predefined functions are as follow:

```
def do(action, argument, element):
    \"\"\"A single browsing operation on the webpage.
    Args:
        :param action: one of the actions from ["Click", "Right Click", "Type", "Search", "Hover", "Scroll Up", "Scroll Down", "Press Enter", "Switch Tab", "Select Dropdown Option", "Wait"].
        :param argument: optional. Only for "Type", "Search", "Switch Page", and "Select Dropdown Option", indicating the content to type in, page number(start from 0) to switch, or key to press.
                                   "Search" action is equivalent to "Type" action plus "Enter" key press.
        :param element: optional. Only for "Click", "Right Click", "Type", "Search", "Select Dropdown Option", and "Hover". Should be specific element id in the html.
    Returns:
        None. The webpage will be updated after executing the action.
    \"\"\"

def exit(message):
    \"\"\"Ending the browsing process if the assistant think it has fulfilled the goal.
    Args:
        :param message: optional. If user's instruction is a question, return assistant's answer in the message based on the browsing content.
    Returns:
        None.
    \"\"\"

def go_backward():
    \"\"\"Go back to the previous page.
    \"\"\"

def go_forward():
    \"\"\"Go forward to the next page.
    \"\"\"
```
"""

RECONSTRUCTION_SYSTEM_PROMPT = """You are a web page reconstruction assistant.
Given a compressed memory of another webpage from the same browsing session, reconstruct the original webpage content from the memory.
"""


# ============== Data Conversion Functions ==============

def trajectory_to_action_samples(traj: Dict) -> List[Dict]:
    """
    Convert trajectory to action prediction samples (v4 multi-turn format).

    Format:
    - History rounds: z_i (memory tokens) + action_i
    - Current round: o_k (full HTML) + z_k → predict action_k
    """
    samples = []
    task_inst = traj['task_instruction']
    rounds = traj['rounds']

    for current_round_idx in range(len(rounds)):
        messages = []

        # System message
        messages.append({'role': 'system', 'content': ACTION_SYSTEM_PROMPT})

        # History rounds as separate turns
        for prev_round in range(current_round_idx):
            rd = rounds[prev_round]

            prompt_list = []
            if prev_round == 0:
                prompt_list.append({'type': 'text', 'text': f"Task Instruction: {task_inst}\n\n"})

            prompt_list.append({'type': 'text', 'text': f"Round {prev_round} observation:"})
            prompt_list.append({'type': 'memory_text', 'memory_text': {'text': rd['observation']}})

            messages.append({'role': 'user', 'content': prompt_list})
            messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': rd['action']}]})

        # Current round: full HTML + memory
        current_rd = rounds[current_round_idx]
        prompt_list = []

        if current_round_idx == 0:
            prompt_list.append({'type': 'text', 'text': f"Task Instruction: {task_inst}\n\n"})

        prompt_list.append({'type': 'text', 'text': f"Round {current_round_idx} observation: "})
        prompt_list.append({'type': 'text', 'text': f"{current_rd['observation']}"})

        messages.append({'role': 'user', 'content': prompt_list})
        messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': current_rd['action']}]})

        samples.append({
            'messages': messages,
            'type': 'action',
            'round': current_round_idx,
            'total_rounds': len(rounds),
            'task_instruction': task_inst,
        })

    return samples


def trajectory_to_reconstruction_samples(
    traj: Dict,
    num_demos_per_target: int = 1,  # 每个 target 随机选几个 demo
) -> List[Dict]:
    """
    Convert trajectory to reconstruction samples.

    Format: z_demo → o_demo, z_target → o_target (multi-turn)

    对于每个 target 轮次，随机选择一个（或多个）其他轮次作为 demo。
    这样生成的样本数 = n * num_demos_per_target，与 action 样本数 (n) 接近。

    Args:
        traj: Trajectory dict with 'rounds' containing observations
        num_demos_per_target: Number of random demos to pair with each target
    """

    def _truncate_memory(memory: str) -> str:
        tokens = tokenizer.encode(memory)
        if len(tokens) > MEMORY_TRUNCATE_LENGTH:
            return tokenizer.decode(tokens[:MEMORY_TRUNCATE_LENGTH])
        return memory

    samples = []
    rounds = traj['rounds']
    n = len(rounds)

    if n < 2:
        return samples

    # 遍历所有 target 轮次
    for target_idx in range(n):
        # 从其他轮次中随机选择 demo
        other_indices = [i for i in range(n) if i != target_idx]

        # 选择 num_demos_per_target 个 demo（如果可用轮次不够就全选）
        num_demos = min(num_demos_per_target, len(other_indices))
        demo_indices = random.sample(other_indices, num_demos)

        for demo_idx in demo_indices:
            o_demo = rounds[demo_idx]['observation']
            o_target = rounds[target_idx]['observation']
            o_demo = _truncate_memory(o_demo)
            o_target = _truncate_memory(o_target)

            messages = [
                {'role': 'system', 'content': RECONSTRUCTION_SYSTEM_PROMPT},
                {'role': 'user', 'content': [
                    {'type': 'memory_text', 'memory_text': {'text': o_demo}},
                ]},
                {'role': 'assistant', 'content': [{'type': 'text', 'text': o_demo}]},
                {'role': 'user', 'content': [
                    {'type': 'memory_text', 'memory_text': {'text': o_target}},
                ]},
                {'role': 'assistant', 'content': [{'type': 'text', 'text': o_target}]},
            ]

            samples.append({
                'messages': messages,
                'type': 'reconstruction',
                'demo_idx': demo_idx,
                'target_idx': target_idx,
                'total_rounds': n,
            })

    return samples


def create_mixed_dataset(
    trajectories: List[Dict],
    num_demos_per_target: int = 1,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """
    Create mixed dataset with action and reconstruction samples.

    使用 num_demos_per_target=1 时，action 和 reconstruction 样本数自然 1:1
    （每个 trajectory 的 n 轮产生 n 个 action 样本和 n 个 reconstruction 样本）

    Args:
        trajectories: List of trajectory dicts
        num_demos_per_target: Number of random demos per target (1 = 1:1 ratio)
        seed: Random seed

    Returns:
        Dict with 'action', 'reconstruction', and 'mixed' sample lists
    """
    random.seed(seed)

    action_samples = []
    recon_samples = []

    for traj in trajectories:
        action_samples.extend(trajectory_to_action_samples(traj))
        recon_samples.extend(trajectory_to_reconstruction_samples(
            traj,
            num_demos_per_target=num_demos_per_target,
        ))

    print(f"Generated {len(action_samples)} action samples")
    print(f"Generated {len(recon_samples)} reconstruction samples")
    print(f"Ratio: {len(action_samples) / (len(action_samples) + len(recon_samples)):.2%} action, "
          f"{len(recon_samples) / (len(action_samples) + len(recon_samples)):.2%} reconstruction")

    # Create mixed dataset
    mixed_samples = action_samples + recon_samples
    random.shuffle(mixed_samples)

    return {
        'action': action_samples,
        'reconstruction': recon_samples,
        'mixed': mixed_samples,
    }


def print_dataset_stats(datasets: Dict[str, List[Dict]]):
    """Print statistics about the generated datasets."""
    print("\n" + "=" * 50)
    print("Dataset Statistics")
    print("=" * 50)

    for name, samples in datasets.items():
        print(f"\n{name.upper()} ({len(samples)} samples):")

        if not samples:
            continue

        # Count by type
        type_counts = Counter(s.get('type', 'unknown') for s in samples)
        for t, c in type_counts.items():
            print(f"  - {t}: {c}")

        # For action samples, show round distribution
        if name == 'action':
            round_dist = Counter(s.get('round', -1) for s in samples)
            print(f"  - Round distribution: {dict(sorted(round_dist.items()))}")


def save_datasets(
    datasets: Dict[str, List[Dict]],
    output_dir: Path,
    prefix: str = "webarena"
):
    """Save mixed dataset to a single parquet file with all metadata."""
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Only save mixed dataset
    samples = datasets.get('mixed', [])
    if not samples:
        print("No samples to save!")
        return

    # Build DataFrame with messages (json.dumps) and all metadata
    data = {
        'messages': [json.dumps(s['messages']) for s in samples],
        'type': [s.get('type', '') for s in samples],
        'round': [s.get('round', -1) for s in samples],
        'total_rounds': [s.get('total_rounds', -1) for s in samples],
        'task_instruction': [s.get('task_instruction', '') for s in samples],
        'demo_idx': [s.get('demo_idx', -1) for s in samples],
        'target_idx': [s.get('target_idx', -1) for s in samples],
    }

    # Save as parquet
    parquet_path = output_dir / f"{prefix}_mixed.parquet"
    df = pd.DataFrame(data)
    df.to_parquet(parquet_path)
    print(f"Saved {parquet_path} ({len(df)} rows)")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Type distribution: {df['type'].value_counts().to_dict()}")


def main():
    parser = argparse.ArgumentParser(description="Create reconstruction training dataset")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="remake_dataset/webarena/webarena_merged_trajectories.json",
        help="Input trajectory JSON file"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="remake_dataset/webarena/reconstruction",
        help="Output directory"
    )
    parser.add_argument(
        "--num-demos-per-target",
        type=int,
        default=1,
        help="Number of random demos per target (1 = 1:1 action:recon ratio)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Load trajectories
    print(f"Loading trajectories from {args.input}")
    with open(args.input, 'r') as f:
        trajectories = json.load(f)
    print(f"Loaded {len(trajectories)} trajectories")

    # Create datasets
    datasets = create_mixed_dataset(
        trajectories,
        num_demos_per_target=args.num_demos_per_target,
        seed=args.seed,
    )

    # Print stats
    print_dataset_stats(datasets)

    # Save
    save_datasets(datasets, args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
