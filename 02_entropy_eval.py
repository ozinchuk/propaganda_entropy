import gc
import os
import re
import uuid

import numpy as np
import pandas as pd
import torch
from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def explode_into_sentences(df, text_column='processed'):
    new_rows = []
    for idx, row in df.iterrows():
        original_text = str(row[text_column])
        doc_id = str(uuid.uuid4())
        sentences = re.split(r'[.!;?]+', original_text)
        for sentence in sentences:
            sentence = sentence.strip()
            if 120 <= len(sentence) <= 200:
                new_row = row.copy()
                new_row['document_id'] = doc_id
                new_row['original_text'] = original_text
                new_row[text_column] = sentence
                new_rows.append(new_row)
    return pd.DataFrame(new_rows)

huggingface_token = os.environ.get('HF_TOKEN')
login(token=huggingface_token)

DEVICE = "cuda"

MODELS = [
    "google/gemma-3-1b-pt",
    "meta-llama/Llama-3.2-3B",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-3-12b-pt",
    "lapa-llm/lapa-12b-pt"
]

USE_4BIT = {
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-3-12b-pt",
    "lapa-llm/lapa-12b-pt"
}

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=huggingface_token)
    quant_config = None
    if model_name in USE_4BIT:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quant_config,
        token=huggingface_token
    )
    model.eval()
    return model, tokenizer

def calculate_metrics_per_row(model, tokenizer, df, text_column='text'):
    results = []
    ln2 = float(np.log(2))
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row[text_column])
        if len(text) < 120 or len(text) > 200:
            continue
        encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        tokens = encoding['input_ids']
        offsets = encoding['offset_mapping']
        if len(tokens) < 2:
            continue
        bos_id = tokenizer.bos_token_id
        if bos_id is not None:
            tokens = [bos_id] + tokens
            offsets = [(0, 0)] + offsets
        input_ids = torch.tensor([tokens]).to(DEVICE)
        labels = torch.tensor([tokens]).to(DEVICE)
        for i, (start, end) in enumerate(offsets):
            if start < 70:
                labels[0, i] = -100
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)
            valid_tokens_count = (labels[0] != -100).sum().item()
            if valid_tokens_count == 0:
                continue
            loss_sum_nats = outputs.loss.item() * valid_tokens_count
        total_bits = loss_sum_nats / ln2
        valid_chars = len(text) - 70
        res_row = row.to_dict()
        res_row['total_bits'] = total_bits
        res_row['valid_chars'] = valid_chars
        res_row['bpc'] = total_bits / valid_chars if valid_chars > 0 else float('nan')
        results.append(res_row)
    return pd.DataFrame(results)

def run_pipeline(df, output_dir="./results"):
    os.makedirs(output_dir, exist_ok=True)
    for model_name in MODELS:
        print(f"\nRunning model: {model_name}")
        model, tokenizer = load_model_and_tokenizer(model_name)
        df_results = calculate_metrics_per_row(model, tokenizer, df)
        df_results['model_name'] = model_name
        safe_model_name = model_name.replace("/", "_")
        output_path = os.path.join(output_dir, f"entropy_{safe_model_name}.csv")
        df_results.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    df = pd.read_csv('merged_dataset.csv')
    df_sentences = explode_into_sentences(df, text_column='processed')
    run_pipeline(df_sentences, output_dir="./entropy_results")
