"""
Inspect inverse dynamics dataset content.
"""

import json
import pandas as pd
import argparse


def _is_html_content(content: str) -> bool:
    """Check if content is HTML"""
    return "<html " in content


def format_content(content) -> str:
    """Format content for display, replacing HTML with [HTML]"""
    if isinstance(content, str):
        if _is_html_content(content):
            return "[HTML]"
        return content
    elif isinstance(content, list):
        parts = []
        for item in content:
            if item.get('type') == 'text':
                text = item.get('text', '')
                if _is_html_content(text):
                    parts.append("[HTML]")
                else:
                    parts.append(text)
            elif item.get('type') == 'memory_text':
                mem_text = item.get('memory_text', {}).get('text', '')
                if _is_html_content(mem_text):
                    parts.append("[MEMORY: HTML]")
                else:
                    # Truncate long memory text
                    if len(mem_text) > 200:
                        parts.append(f"[MEMORY: {mem_text[:200]}...]")
                    else:
                        parts.append(f"[MEMORY: {mem_text}]")
        return "".join(parts)
    return str(content)


def print_sample(sample: dict, idx: int):
    """Print a single sample"""
    print(f"\n{'='*60}")
    print(f"Sample {idx} | Type: {sample.get('type', 'unknown')}")
    print(f"{'='*60}")

    messages = json.loads(sample['messages'])

    for msg in messages:
        role = msg['role'].upper()
        content = format_content(msg['content'])

        # Truncate very long content
        if len(content) > 500:
            content = content[:500] + "..."

        print(f"\n[{role}]")
        print(content)


def main():
    parser = argparse.ArgumentParser(description="Inspect inverse dynamics dataset")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="remake_dataset/webarena/inverse_dynamics/webarena_inverse_dynamics_mixed.parquet",
        help="Input parquet file"
    )
    parser.add_argument(
        "--num", "-n",
        type=int,
        default=3,
        help="Number of samples to show"
    )
    parser.add_argument(
        "--type", "-t",
        type=str,
        choices=["action", "inverse_dynamics", "all"],
        default="all",
        help="Filter by sample type"
    )

    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} samples from {args.input}")
    print(f"Type distribution: {df['type'].value_counts().to_dict()}")

    if args.type != "all":
        df = df[df['type'] == args.type]
        print(f"Filtered to {len(df)} {args.type} samples")

    for i, (_, row) in enumerate(df.head(args.num).iterrows()):
        print_sample(row.to_dict(), i)


if __name__ == "__main__":
    main()
