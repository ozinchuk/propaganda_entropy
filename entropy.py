import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

MODEL_NAME = "mlx-community/Qwen2.5-7B-Instruct-4bit"

def calculate_metrics_per_row(model, tokenizer, df, text_column='processed'):
    results = []
    ln2 = float(np.log(2))
    
    hf_tokenizer = getattr(tokenizer, "_tokenizer", tokenizer)

    rows = df.to_dict('records')

    for row in tqdm(rows):
        text = str(row[text_column])

        encoding = hf_tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        tokens = encoding['input_ids']
        offsets = encoding['offset_mapping']

        if len(tokens) < 2:
            continue

        bos_id = hf_tokenizer.bos_token_id
        if bos_id is not None:
            tokens = [bos_id] + tokens
            offsets = [(0, 0)] + offsets

        input_ids = mx.array([tokens])
        targets = mx.array([tokens[1:]])

        logits = model(input_ids)
        shift_logits = logits[:, :-1, :]

        ce = nn.losses.cross_entropy(shift_logits, targets, reduction='none')

        mask_list = [1.0 if start >= 70 else 0.0 for start, end in offsets[1:]]
        mask = mx.array([mask_list])

        valid_tokens_count = mx.sum(mask).item()
        if valid_tokens_count == 0:
            continue

        loss_sum_nats = mx.sum(ce * mask).item()
        total_bits = loss_sum_nats / ln2
        valid_chars = len(text) - 70

        res_row = dict(row)
        res_row['total_bits'] = total_bits
        res_row['valid_chars'] = valid_chars
        res_row['bpc'] = total_bits / valid_chars if valid_chars > 0 else float('nan')

        results.append(res_row)

    return pd.DataFrame(results)

def run_pipeline(df, output_dir="./results"):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nLoading model: {MODEL_NAME}")
    model, tokenizer = load(MODEL_NAME)

    df_results = calculate_metrics_per_row(model, tokenizer, df)
    df_results['model_name'] = MODEL_NAME

    safe_model_name = MODEL_NAME.replace("/", "_")
    output_path = os.path.join(output_dir, f"entropy_{safe_model_name}.csv")

    df_results.to_csv(output_path, index=False)
    print(f"Results saved: {output_path}")


if __name__ == "__main__":
    df = pd.read_csv("merged_dataset_split.csv")
    run_pipeline(df)