"""
Create inverse dynamics training dataset from WebArena trajectories.

Training objectives:
- Action samples: o_k + z_k + z_{0..k-1} + Task → Action_k
- Inverse Dynamics samples: [mem_t] + [mem_{t+1}] → action_t (predict action between consecutive states)

Mixed training helps the encoder learn to preserve state transition semantics.
"""

import json
import random
from collections import Counter
from pathlib import Path
from typing import List, Dict
import argparse

# ============== System Prompts ==============

NEW_ACTION_SYSTEM_PROMPT = [
    {
        'type': 'text',
        'text': '''# Setup
You are a professional web browsing agent assistant that can fulfill user's high-level instructions.
Given the simplified html of the browsed webpage at each step, you plan operations in python-style pseudo code using provided functions, or customize functions (if necessary) and then provide their implementations.
Each past will be a compressed memory of the browsed webpage at each step. For example: '''
    },
    {
        'type': 'memory_text',
        'memory_text': {'text': "This is an HTML webpage"}
    },
    {
        'type': 'text',
        'text': '''# More details about the code
Your code should be readable, simple, and only **ONE-LINE-OF-CODE** at a time, avoid using loop statement and only use if-else control if necessary. Predefined functions are as follow:

```
def do(action, argument, element):
    """A single browsing operation on the webpage.
    Args:
        :param action: one of the actions from ["Click", "Right Click", "Type", "Search", "Hover", "Scroll Up", "Scroll Down", "Press Enter", "Switch Tab", "Select Dropdown Option", "Wait"].
        :param argument: optional. Only for "Type", "Search", "Switch Page", and "Select Dropdown Option", indicating the content to type in, page number(start from 0) to switch, or key to press.
                                   "Search" action is equivalent to "Type" action plus "Enter" key press.
        :param element: optional. Only for "Click", "Right Click", "Type", "Search", "Select Dropdown Option", and "Hover". Should be specific element id in the html.
    Returns:
        None. The webpage will be updated after executing the action.
    """

def exit(message):
    """Ending the browsing process if the assistant think it has fulfilled the goal.
    Args:
        :param message: optional. If user's instruction is a question, return assistant's answer in the message based on the browsing content.
    Returns:
        None.
    """

def go_backward():
    """Go back to the previous page.
    """

def go_forward():
    """Go forward to the next page.
    """
```
'''
    }
]

INVERSE_DYNAMICS_SYSTEM_PROMPT = [
    {
        'type': 'text',
        'text': '''You are a web browsing action predictor.
Given the memory of two consecutive webpage states from the same browsing session, predict what action was taken to transition from the first state to the second.

Output the action in python-style pseudo code using the following functions:

```
def do(action, argument, element):
    """A single browsing operation on the webpage.
    Args:
        :param action: one of the actions from ["Click", "Right Click", "Type", "Search", "Hover", "Scroll Up", "Scroll Down", "Press Enter", "Switch Tab", "Select Dropdown Option", "Wait"].
        :param argument: optional. Only for "Type", "Search", "Switch Page", and "Select Dropdown Option".
        :param element: optional. Only for "Click", "Right Click", "Type", "Search", "Select Dropdown Option", and "Hover". Should be specific element id in the html.
    """

def exit(message):
    """Ending the browsing process."""

def go_backward():
    """Go back to the previous page."""

def go_forward():
    """Go forward to the next page."""
```
'''
    }
]

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
        messages.append({'role': 'system', 'content': NEW_ACTION_SYSTEM_PROMPT})

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


def trajectory_to_inverse_dynamics_samples(traj: Dict) -> List[Dict]:
    """
    Convert trajectory to inverse dynamics samples.

    Format: [mem_t] + [mem_{t+1}] → action_t

    For each pair of consecutive states (t, t+1), predict action_t.
    A trajectory with n rounds produces n-1 samples.
    """
    samples = []
    rounds = traj['rounds']
    task_inst = traj['task_instruction']
    n = len(rounds)

    if n < 2:
        return samples

    # Iterate over all consecutive state pairs
    for t in range(n - 1):
        mem_t = rounds[t]['observation']
        mem_t_plus_1 = rounds[t + 1]['observation']
        action_t = rounds[t]['action']

        messages = [
            {'role': 'system', 'content': INVERSE_DYNAMICS_SYSTEM_PROMPT},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': f'Task Instruction: {task_inst}\n\n'},
                {'type': 'text', 'text': f'Round {t} observation:'},
                {'type': 'memory_text', 'memory_text': {'text': mem_t}},
                {'type': 'text', 'text': f'\n\nRound {t + 1} observation:'},
                {'type': 'memory_text', 'memory_text': {'text': mem_t_plus_1}},
                {'type': 'text', 'text': f'\n\nWhat action was taken after Round {t}?'},
            ]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': action_t}]},
        ]

        samples.append({
            'messages': messages,
            'type': 'inverse_dynamics',
            'round_t': t,
            'round_t_plus_1': t + 1,
            'total_rounds': n,
        })

    return samples


def create_mixed_dataset(
    trajectories: List[Dict],
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """
    Create mixed dataset with action and inverse dynamics samples.

    A trajectory with n rounds produces:
    - n action samples
    - n-1 inverse dynamics samples

    Ratio is approximately 1:1.

    Args:
        trajectories: List of trajectory dicts
        seed: Random seed

    Returns:
        Dict with 'action', 'inverse_dynamics', and 'mixed' sample lists
    """
    random.seed(seed)

    action_samples = []
    inv_dyn_samples = []

    for traj in trajectories:
        action_samples.extend(trajectory_to_action_samples(traj))
        inv_dyn_samples.extend(trajectory_to_inverse_dynamics_samples(traj))

    print(f"Generated {len(action_samples)} action samples")
    print(f"Generated {len(inv_dyn_samples)} inverse dynamics samples")
    total = len(action_samples) + len(inv_dyn_samples)
    print(f"Ratio: {len(action_samples) / total:.2%} action, "
          f"{len(inv_dyn_samples) / total:.2%} inverse dynamics")

    # Create mixed dataset
    mixed_samples = action_samples + inv_dyn_samples
    random.shuffle(mixed_samples)

    return {
        'action': action_samples,
        'inverse_dynamics': inv_dyn_samples,
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

        # For inverse dynamics samples, show round_t distribution
        if name == 'inverse_dynamics':
            round_dist = Counter(s.get('round_t', -1) for s in samples)
            print(f"  - Round t distribution: {dict(sorted(round_dist.items()))}")


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
        'round_t': [s.get('round_t', -1) for s in samples],
        'round_t_plus_1': [s.get('round_t_plus_1', -1) for s in samples],
        'total_rounds': [s.get('total_rounds', -1) for s in samples],
        'task_instruction': [s.get('task_instruction', '') for s in samples],
    }

    # Save as parquet
    parquet_path = output_dir / f"{prefix}_inverse_dynamics_mixed.parquet"
    df = pd.DataFrame(data)
    df.to_parquet(parquet_path)
    print(f"Saved {parquet_path} ({len(df)} rows)")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Type distribution: {df['type'].value_counts().to_dict()}")


def main():
    parser = argparse.ArgumentParser(description="Create inverse dynamics training dataset")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="remake_dataset/webarena/webarena_merged_trajectories.json",
        help="Input trajectory JSON file"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="remake_dataset/webarena/inverse_dynamics",
        help="Output directory"
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
        seed=args.seed,
    )

    # Print stats
    print_dataset_stats(datasets)

    # Save
    save_datasets(datasets, args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
