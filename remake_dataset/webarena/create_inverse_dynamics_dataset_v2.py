"""
Create inverse dynamics training dataset from WebArena trajectories (v2).

v2 Changes:
- Convert actions to natural language (remove element="id", keep semantic descriptions)
- Deduplicate by (obs_t, obs_t+1) pairs to remove redundant samples
- Mix action samples and inverse dynamics samples

Training objectives:
- Action samples: o_k + z_k + z_{0..k-1} + Task → Action_k
- Inverse Dynamics samples: [mem_t] + [mem_{t+1}] → action_t (predict action between consecutive states)

Mixed training helps the encoder learn to preserve state transition semantics.
"""

import json
import random
import re
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

INVERSE_DYNAMICS_SYSTEM_PROMPT_V2 = [
    {
        'type': 'text',
        'text': '''You are a web browsing action predictor.
Given the memory of two consecutive webpage states from the same browsing session, predict what action was taken to transition from the first state to the second.

Describe the action in natural language, including:
- The action type (Click, Type, Search, Scroll, etc.)
- The target element (if applicable)
- The input content (if applicable)

Examples:
- Click on the 'Submit' button.
- Type "hello world" into the search bar.
- Scroll down the page.
- Select "Option A" from the dropdown menu.
'''
    }
]

# ============== Action Cleaning Functions ==============

def extract_action_type(action_str: str) -> str:
    """Extract action type from raw action string."""
    # Format: do(action="Click", ...) or do(action="Scroll Down")
    match = re.search(r'do\(action=["\']([^"\']+)["\']', action_str)
    if match:
        return match.group(1)
    if 'exit(' in action_str:
        return 'exit'
    if 'go_backward' in action_str:
        return 'go_backward'
    if 'go_forward' in action_str:
        return 'go_forward'
    if 'quote(' in action_str:
        return 'quote'
    return 'unknown'


def clean_action_to_natural_language(action_str: str) -> str:
    """
    Convert action string to natural language description.
    - Remove element="id" (non-semantic)
    - Keep semantic element descriptions from comments
    - Convert quote() to natural language
    """

    # Handle quote() - convert to natural language
    quote_match = re.search(r'quote\(content=["\'](.+?)["\']\)', action_str, re.DOTALL)
    if quote_match:
        content = quote_match.group(1)
        return f"Record the following information: {content}"

    # Handle go_backward / go_forward
    if 'go_backward()' in action_str:
        return "Go back to the previous page."
    if 'go_forward()' in action_str:
        return "Go forward to the next page."

    # Extract element description from comment (# Element: ...)
    element_desc = ""
    comment_match = re.search(r'#\s*Element:\s*(.+?)(?:\n|$)', action_str)
    if comment_match:
        element_desc = comment_match.group(1).strip()

    # Parse do() call
    action_match = re.search(r'do\(action=["\']([^"\']+)["\']', action_str)
    if not action_match:
        return action_str  # Cannot parse, return original

    action_type = action_match.group(1)

    # Extract argument (if exists)
    arg_match = re.search(r'argument=["\']([^"\']*)["\']', action_str)
    argument = arg_match.group(1) if arg_match else None

    # Generate natural language based on action type
    if action_type == "Click":
        if element_desc:
            return f"Click on {element_desc}."
        return "Click on the element."

    elif action_type == "Type":
        if element_desc and argument:
            return f"Type \"{argument}\" into {element_desc}."
        elif argument:
            return f"Type \"{argument}\"."
        return "Type text into the field."

    elif action_type == "Search":
        if element_desc and argument:
            return f"Search for \"{argument}\" in {element_desc}."
        elif argument:
            return f"Search for \"{argument}\"."
        return "Perform a search."

    elif action_type == "Hover":
        if element_desc:
            return f"Hover over {element_desc}."
        return "Hover over the element."

    elif action_type == "Select Dropdown Option":
        if element_desc and argument:
            return f"Select \"{argument}\" from {element_desc}."
        elif argument:
            return f"Select dropdown option \"{argument}\"."
        return "Select a dropdown option."

    elif action_type == "Scroll Down":
        return "Scroll down the page."

    elif action_type == "Scroll Up":
        return "Scroll up the page."

    elif action_type == "Press Enter":
        return "Press the Enter key."

    elif action_type == "Wait":
        return "Wait for the page to load."

    else:
        # Unknown action type
        if element_desc:
            return f"{action_type} on {element_desc}."
        return f"{action_type}."


# ============== Data Conversion Functions ==============

def trajectory_to_action_samples(traj: Dict) -> List[Dict]:
    """
    Convert trajectory to action prediction samples (multi-turn format).

    Format:
    - History rounds: z_i (memory tokens) + action_i
    - Current round: o_k (full HTML) + z_k → predict action_k

    NOTE: No cleaning/deduplication applied to action samples.
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

    Format: [mem_t] + [mem_{t+1}] → action_t (natural language)

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
        raw_action = rounds[t]['action']

        # Clean action to natural language
        action_t = clean_action_to_natural_language(raw_action)
        action_type = extract_action_type(raw_action)

        messages = [
            {'role': 'system', 'content': INVERSE_DYNAMICS_SYSTEM_PROMPT_V2},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': f'Task Instruction: {task_inst}\n\n'},
                {'type': 'text', 'text': f'State before action:'},
                {'type': 'memory_text', 'memory_text': {'text': mem_t}},
                {'type': 'text', 'text': f'\n\nState after action:'},
                {'type': 'memory_text', 'memory_text': {'text': mem_t_plus_1}},
                {'type': 'text', 'text': f'\n\nWhat action was taken to transition from the first state to the second?'},
            ]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': action_t}]},
        ]

        samples.append({
            'messages': messages,
            'type': 'inverse_dynamics',
            'action_type': action_type,
            'round_t': t,
            'round_t_plus_1': t + 1,
            'total_rounds': n,
            'task_instruction': task_inst,
            # Store hashes for deduplication
            '_obs_t_hash': hash(mem_t),
            '_obs_t_plus_1_hash': hash(mem_t_plus_1),
        })

    return samples


def deduplicate_by_obs(samples: List[Dict]) -> List[Dict]:
    """
    Deduplicate samples by (obs_t, obs_t+1) pairs.

    Keeps the first sample encountered for each unique obs pair.
    """
    seen_obs = set()
    deduplicated = []

    for s in samples:
        obs_key = (s['_obs_t_hash'], s['_obs_t_plus_1_hash'])

        if obs_key not in seen_obs:
            seen_obs.add(obs_key)
            # Remove internal hash fields before saving
            sample_clean = {k: v for k, v in s.items() if not k.startswith('_')}
            deduplicated.append(sample_clean)

    return deduplicated


def create_mixed_dataset(
    trajectories: List[Dict],
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """
    Create mixed dataset with action samples and deduplicated inverse dynamics samples.

    Args:
        trajectories: List of trajectory dicts
        seed: Random seed

    Returns:
        Dict with 'action', 'inverse_dynamics', 'inverse_dynamics_dedup', 'mixed' sample lists
    """
    random.seed(seed)

    # Generate action samples (no cleaning/deduplication)
    action_samples = []
    for traj in trajectories:
        action_samples.extend(trajectory_to_action_samples(traj))
    print(f"Generated {len(action_samples)} action samples (no dedup)")

    # Generate inverse dynamics samples
    inv_dyn_samples = []
    for traj in trajectories:
        inv_dyn_samples.extend(trajectory_to_inverse_dynamics_samples(traj))
    print(f"Generated {len(inv_dyn_samples)} raw inverse dynamics samples")

    # Deduplicate inverse dynamics by obs pairs
    inv_dyn_dedup = deduplicate_by_obs(inv_dyn_samples)
    print(f"After deduplication: {len(inv_dyn_dedup)} inverse dynamics samples")
    print(f"Removed {len(inv_dyn_samples) - len(inv_dyn_dedup)} duplicates ({(len(inv_dyn_samples) - len(inv_dyn_dedup))/len(inv_dyn_samples)*100:.1f}%)")

    # Create mixed dataset (action + deduplicated inverse dynamics)
    mixed_samples = action_samples + inv_dyn_dedup
    random.shuffle(mixed_samples)

    total = len(mixed_samples)
    print(f"\nMixed dataset: {total} samples")
    print(f"  Action: {len(action_samples)} ({len(action_samples)/total*100:.1f}%)")
    print(f"  Inverse Dynamics (dedup): {len(inv_dyn_dedup)} ({len(inv_dyn_dedup)/total*100:.1f}%)")

    return {
        'action': action_samples,
        'inverse_dynamics': inv_dyn_samples,
        'inverse_dynamics_dedup': inv_dyn_dedup,
        'mixed': mixed_samples,
    }


def print_dataset_stats(datasets: Dict[str, List[Dict]]):
    """Print statistics about the generated datasets."""
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)

    for name, samples in datasets.items():
        print(f"\n{name.upper()} ({len(samples)} samples):")

        if not samples:
            continue

        # Count by type (action vs inverse_dynamics)
        type_counts = Counter(s.get('type', 'unknown') for s in samples)
        print(f"  Type distribution: {dict(type_counts)}")

        # For inverse dynamics, show action type distribution
        inv_dyn_samples = [s for s in samples if s.get('type') == 'inverse_dynamics']
        if inv_dyn_samples:
            action_counts = Counter(s.get('action_type', 'unknown') for s in inv_dyn_samples)
            print(f"  Inverse dynamics action types:")
            for t, c in sorted(action_counts.items(), key=lambda x: -x[1])[:5]:
                print(f"    {t}: {c} ({c/len(inv_dyn_samples)*100:.1f}%)")

        # Round distribution (summarized)
        # For action samples, use 'round'; for inverse dynamics, use 'round_t'
        action_samples = [s for s in samples if s.get('type') == 'action']
        if action_samples:
            round_dist = Counter(s.get('round', -1) for s in action_samples)
            early = sum(v for k, v in round_dist.items() if 0 <= k <= 3)
            mid = sum(v for k, v in round_dist.items() if 4 <= k <= 7)
            late = sum(v for k, v in round_dist.items() if k >= 8)
            print(f"  Action round distribution: early(0-3)={early}, mid(4-7)={mid}, late(8+)={late}")


def save_dataset(
    samples: List[Dict],
    output_path: Path,
):
    """Save dataset to parquet file."""
    import pandas as pd

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not samples:
        print("No samples to save!")
        return

    # Build DataFrame with all metadata
    data = {
        'messages': [json.dumps(s['messages']) for s in samples],
        'type': [s.get('type', '') for s in samples],
        'action_type': [s.get('action_type', '') for s in samples],  # only for inverse_dynamics
        'round': [s.get('round', -1) for s in samples],  # for action samples
        'round_t': [s.get('round_t', -1) for s in samples],  # for inverse_dynamics
        'round_t_plus_1': [s.get('round_t_plus_1', -1) for s in samples],
        'total_rounds': [s.get('total_rounds', -1) for s in samples],
        'task_instruction': [s.get('task_instruction', '') for s in samples],
    }

    df = pd.DataFrame(data)
    df.to_parquet(output_path)
    print(f"\nSaved {output_path} ({len(df)} rows)")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Type distribution: {df['type'].value_counts().to_dict()}")


def main():
    parser = argparse.ArgumentParser(description="Create mixed action + inverse dynamics training dataset v2")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="remake_dataset/webarena/webarena_merged_trajectories.json",
        help="Input trajectory JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="remake_dataset/webarena/webarena_mixed_v2.parquet",
        help="Output parquet file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--inverse-only",
        action="store_true",
        help="Only generate inverse dynamics samples (no action samples)"
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

    # Save dataset based on mode
    if args.inverse_only:
        save_dataset(datasets['inverse_dynamics_dedup'], args.output)
    else:
        save_dataset(datasets['mixed'], args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
