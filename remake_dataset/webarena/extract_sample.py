import pandas as pd
import json
import random

# Read the parquet file
df = pd.read_parquet('remake_dataset/webarena/webarena_repeat_obs_sft.parquet')

# Randomly select 3 indices
random.seed(42)
sample_indices = random.sample(range(len(df)), 3)

# Extract messages
samples = []
for idx in sample_indices:
    msg = df['messages'].iloc[idx]
    # If it's a string, try to parse it as JSON
    if isinstance(msg, str):
        try:
            msg = json.loads(msg)
        except:
            # If parsing fails, keep as string
            pass
    samples.append(msg)

# Save to JSON file
with open('remake_dataset/webarena/sample_messages.json', 'w', encoding='utf-8') as f:
    json.dump(samples, f, indent=2, ensure_ascii=False)

print(f'Saved {len(samples)} random messages to remake_dataset/webarena/sample_messages.json')
print(f'Sample indices: {sample_indices}')



