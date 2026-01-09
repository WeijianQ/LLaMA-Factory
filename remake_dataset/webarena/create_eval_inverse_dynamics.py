"""
Create inverse dynamics test set from WebArena evaluation traces.

Input: eval traces (JSONL files, one per task)
Output: JSON with same message format as training set (but no assistant output)

Uses same dedup and action paraphrasing logic as create_inverse_dynamics_dataset_v2.py
"""

import json
import re
from pathlib import Path
from collections import Counter
from typing import List, Dict
import argparse


# ============== System Prompt (same as training) ==============

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


# ============== Action Cleaning Functions (from v2) ==============

def extract_action_type(action_str: str) -> str:
    """Extract action type from raw action string."""
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

    # Handle exit()
    exit_match = re.search(r'exit\((?:message=["\'](.+?)["\'])?\)', action_str, re.DOTALL)
    if exit_match:
        message = exit_match.group(1)
        if message:
            return f"Exit and respond: {message}"
        return "Exit the task."

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

    elif action_type == "Right Click":
        if element_desc:
            return f"Right click on {element_desc}."
        return "Right click on the element."

    elif action_type == "Switch Tab":
        if argument:
            return f"Switch to tab {argument}."
        return "Switch to another tab."

    else:
        # Unknown action type
        if element_desc:
            return f"{action_type} on {element_desc}."
        return f"{action_type}."


# ============== Trace Processing ==============

def load_trace(trace_path: Path) -> List[Dict]:
    """Load a single trace file (JSONL format)."""
    steps = []
    with open(trace_path, 'r') as f:
        for line in f:
            if line.strip():
                steps.append(json.loads(line))
    return steps


def trace_to_inverse_dynamics_samples(steps: List[Dict]) -> List[Dict]:
    """
    Convert a trace to inverse dynamics samples.

    Format matches training set:
    - messages: [system, user] (no assistant - for inference)
    - target: ground truth action (natural language)
    """
    samples = []
    n = len(steps)

    if n < 2:
        return samples

    # Get task instruction from first step
    task_inst = steps[0].get('target', steps[0].get('prompt', ''))
    trace_id = steps[0].get('trace_id', -1)

    for t in range(n - 1):
        obs_t = steps[t]['html']
        obs_t_plus_1 = steps[t + 1]['html']
        raw_action = steps[t]['response']

        # Clean action to natural language
        action_nl = clean_action_to_natural_language(raw_action)
        action_type = extract_action_type(raw_action)

        # Build messages (same format as training, but no assistant output)
        messages = [
            {'role': 'system', 'content': INVERSE_DYNAMICS_SYSTEM_PROMPT_V2},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': f'Task Instruction: {task_inst}\n\n'},
                {'type': 'text', 'text': 'State before action:'},
                {'type': 'memory_text', 'memory_text': {'text': obs_t}},
                {'type': 'text', 'text': '\n\nState after action:'},
                {'type': 'memory_text', 'memory_text': {'text': obs_t_plus_1}},
                {'type': 'text', 'text': '\n\nWhat action was taken to transition from the first state to the second?'},
            ]},
            # No assistant message - this is for inference
        ]

        samples.append({
            'messages': messages,
            'target': action_nl,  # Ground truth for evaluation
            'target_raw': raw_action,
            'action_type': action_type,
            'trace_id': trace_id,
            'step_t': t,
            'step_t_plus_1': t + 1,
            'total_steps': n,
            'task_instruction': task_inst,
            # For deduplication
            '_obs_t_hash': hash(obs_t),
            '_obs_t_plus_1_hash': hash(obs_t_plus_1),
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


def print_stats(samples: List[Dict], name: str = "Dataset"):
    """Print statistics about the samples."""
    print(f"\n{'='*60}")
    print(f"{name} Statistics")
    print(f"{'='*60}")
    print(f"Total samples: {len(samples)}")

    if not samples:
        return

    # Action type distribution
    action_counts = Counter(s.get('action_type', 'unknown') for s in samples)
    print(f"\nAction type distribution:")
    for action_type, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"  {action_type}: {count} ({count/len(samples)*100:.1f}%)")

    # Trace distribution
    trace_ids = set(s.get('trace_id', -1) for s in samples)
    print(f"\nUnique traces: {len(trace_ids)}")

    # Step distribution
    step_counts = Counter(s.get('step_t', -1) for s in samples)
    early = sum(v for k, v in step_counts.items() if 0 <= k <= 2)
    mid = sum(v for k, v in step_counts.items() if 3 <= k <= 5)
    late = sum(v for k, v in step_counts.items() if k >= 6)
    print(f"Step distribution: early(0-2)={early}, mid(3-5)={mid}, late(6+)={late}")


def main():
    parser = argparse.ArgumentParser(description="Create inverse dynamics test set from eval traces")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="/fs/ess/PAS1576/qwjian/webarena_s/WebAgent-R1/WebAgent-R1/Eval/eval_results/qwen3-sft_trained/traces",
        help="Input directory containing trace JSONL files"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="remake_dataset/webarena/webarena_eval_inverse_dynamics.json",
        help="Output JSON file"
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Skip deduplication"
    )

    args = parser.parse_args()

    # Find all trace files
    input_dir = Path(args.input)
    trace_files = sorted(input_dir.glob("*.jsonl"))
    print(f"Found {len(trace_files)} trace files in {input_dir}")

    # Process all traces
    all_samples = []
    for trace_file in trace_files:
        steps = load_trace(trace_file)
        samples = trace_to_inverse_dynamics_samples(steps)
        all_samples.extend(samples)

    print(f"\nGenerated {len(all_samples)} raw samples from {len(trace_files)} traces")

    # Deduplicate
    if args.no_dedup:
        final_samples = [{k: v for k, v in s.items() if not k.startswith('_')} for s in all_samples]
        print("Skipping deduplication")
    else:
        final_samples = deduplicate_by_obs(all_samples)
        removed = len(all_samples) - len(final_samples)
        print(f"After deduplication: {len(final_samples)} samples")
        print(f"Removed {removed} duplicates ({removed/len(all_samples)*100:.1f}%)")

    # Print stats
    print_stats(final_samples, "Final Test Set")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(final_samples, f, indent=2)

    print(f"\nSaved to {output_path}")

    # Also save a compact version (one sample per line)
    compact_path = output_path.with_suffix('.jsonl')
    with open(compact_path, 'w') as f:
        for sample in final_samples:
            f.write(json.dumps(sample) + '\n')
    print(f"Saved compact version to {compact_path}")


if __name__ == "__main__":
    main()
