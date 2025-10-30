import pandas as pd
import json

# TRAIN_DATA_PATH = "data/webshop_sft_data/webshop_KEEP_ACTION_proxy_tasks_only_12000.parquet"
# EVAL_DATA_PATH = "data/webshop_sft_data/webshop_KEEP_ACTION_proxy_tasks_only_1600.parquet"
TRAIN_DATA_PATH = "data/webshop_sft_data/webshop_KEEP_ACTION_policy_only_TRAIN_13342.parquet"
EVAL_DATA_PATH = "data/webshop_sft_data/webshop_KEEP_ACTION_policy_only_VAL_1736.parquet"


def remove_long_memory(input_data_path, max_memory_num=30):
    df = pd.read_parquet(input_data_path)
    original_length = len(df)
    indices_to_remove = []
    for i, row in df.iterrows():
        message = json.loads(row['messages'])
        memory_count = 0
        content_list = message[0]['content']
        for content_item in content_list:
            if isinstance(content_item, dict):
                if content_item.get("type") == "memory_text":
                    memory_count += 1
        if memory_count > max_memory_num:
            indices_to_remove.append(i)
    df = df.drop(indices_to_remove)
    new_length = len(df)
    print(f"Removed {original_length - new_length} rows from {input_data_path}")
    return df

if __name__ == "__main__":
    train_df = remove_long_memory(TRAIN_DATA_PATH)
    eval_df = remove_long_memory(EVAL_DATA_PATH)
    TRAIN_OUTPUT_DATA_PATH = f"data/webshop_sft_data/webshop_train_keep_action_with_cm_policy_only_{len(train_df)}.parquet"
    EVAL_OUTPUT_DATA_PATH = f"data/webshop_sft_data/webshop_val_keep_action_with_cm_policy_only_{len(eval_df)}.parquet"
    print(f"Saved train data to {TRAIN_OUTPUT_DATA_PATH}")
    print(f"Saved eval data to {EVAL_OUTPUT_DATA_PATH}")
    train_df.to_parquet(TRAIN_OUTPUT_DATA_PATH)
    eval_df.to_parquet(EVAL_OUTPUT_DATA_PATH)