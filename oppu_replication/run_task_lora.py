#!/usr/bin/env python3
"""Patched copy of third_party/OPPU/task_LoRA.py @ 87f8c69 for SmolLM3-3B.

Trains the task-level LoRA on user_others (their recipe verbatim: r=8, α=8,
q/v/k(+dead "out_proj"), LR=1e-4, linear, warmup 0.1, wd 1e-2, grad-norm 0.3,
batch 16, 3 epochs, loss masked to completion), merges nothing, then
evaluates the RAG arm on the 100 test users' queries with their sampled
decoding. Every deviation from upstream is marked `# PATCH Pn` and listed in
PATCHES.md. Training/eval logic is otherwise theirs, line for line.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                       # PATCH P2: vendored rank_bm25
sys.path.insert(0, str(_HERE.parent / "third_party" / "OPPU"))  # PATCH P2: their utils

import torch
# PATCH P1: dropped unused `import bitsandbytes as bnb` (not in container)
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from rank_bm25 import BM25Okapi
import transformers
from utils import split_batch, get_first_k_tokens, print_trainable_parameters, name2taskid
from utils import extract_citation_title, extract_option, extract_movie, extract_news_cat, \
    extract_news_headline, extract_product_review, extract_scholarly_title, \
    extract_tweet_paraphrasing
import json
from tqdm import tqdm

from wrapper_common import (PROJECT_ROOT, TEST_FILE, banner, refuse_overwrite,
                            write_meta, lora_delta_stats, export_predictions)

parser = argparse.ArgumentParser(description="Parser for LoRA")
# PATCH P4: default model -> local SmolLM3-3B
parser.add_argument('--model_name', type=str, default=str(PROJECT_ROOT / 'data/models/SmolLM3-3B'))
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--k', type=int, default=0)
parser.add_argument('--max_step', type=int, default=5000)
parser.add_argument('--cut_off', type=int, default=2048)
parser.add_argument('--max_epoch', type=int, default=3)
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--task_name', type=str, default='movie_tagging')
parser.add_argument('--add_profile', action='store_true')
parser.add_argument('--access_token', type=str, default=None)
# PATCH P8/P9/P11/P13: wrapper-layer args (paths, smoke limits, seed, overwrite)
parser.add_argument('--data-root', type=str, default=str(PROJECT_ROOT / 'data/oppu_release/data'))
parser.add_argument('--prompt-file', type=str, default=str(PROJECT_ROOT / 'third_party/OPPU/prompt/prompt.json'))
parser.add_argument('--ckpt-root', type=str, default=str(PROJECT_ROOT / 'train/checkpoints/oppu_rep'))
parser.add_argument('--out-root', type=str, default=str(PROJECT_ROOT / 'results/oppu_rep'))
parser.add_argument('--limit', type=int, default=0, help='smoke: cap test users')
parser.add_argument('--limit-train', type=int, default=0, help='smoke: cap train users')
parser.add_argument('--seed', type=int, default=0, help='generation seed (their code sets none)')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--base-only', action='store_true',
                    help='PATCH P17: skip all training — evaluate the bare base model '
                         'under the identical retrieval prompts/decoding (chart baseline arm)')

args = parser.parse_args()
model_name = args.model_name
task_name = args.task_name
batch_size = args.batch_size
k = args.k
cutoff_len = args.cut_off
add_eos_token = False
max_epoch = args.max_epoch

banner("run_task_lora", args)

# PATCH P8: our output layout + refuse-to-overwrite + _limitN smoke suffix
suffix = ""
if args.limit > 0 or args.limit_train > 0:
    suffix = f"_limit{args.limit}t{args.limit_train}"
stem = "base" if args.base_only else "task"
ckpt_dir = Path(args.ckpt_root) / task_name / f"task_lora_k{k}{suffix}"
out_dir = Path(args.out_root) / task_name
pred_json = out_dir / f"{stem}_k{k}{suffix}_preds.json"
pred_jsonl = out_dir / f"{stem}_k{k}{suffix}_preds.jsonl"
meta_json = out_dir / f"{stem}_k{k}{suffix}_meta.json"
refuse_overwrite(([pred_json, pred_jsonl] if args.base_only else [ckpt_dir, pred_json, pred_jsonl]), args.overwrite)

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token=args.access_token)
# PATCH P4: their Llama-2-specific special-token surgery ("</s>", '[PAD]') only
# applies when those tokens exist; for SmolLM3 pad with its own eos (same intent:
# left-pad using eos, generation stops on eos).
if "</s>" in tokenizer.get_vocab():
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = '[PAD]'
    tokenizer.pad_token_id = tokenizer.eos_token_id
else:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    local_files_only=False,
    device_map='auto',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

base_model.config.use_cache = False
base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.config.eos_token_id = tokenizer.eos_token_id
base_model.config.bos_token_id = tokenizer.bos_token_id

from peft import prepare_model_for_kbit_training

base_model.gradient_checkpointing_enable()
base_model = prepare_model_for_kbit_training(base_model)

from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# PATCH P15: the container's transformers (v5-era) removed some 2024
# TrainingArguments params (first hit: group_by_length). Build the kwargs
# dict, filter against the installed signature, and log what was dropped —
# each dropped key is a recorded deviation, not a silent one.
import inspect
# PATCH P16: batch 16 x 2048 tokens OOMs 40GB GPUs on the two long-document
# tasks (product reviews, scholarly abstracts) — their paper's Table 5 says
# batch is "3-16, task-dependent" while the code hardcodes 16 for all. Use
# per-device 4 x grad-accum 4 there (effective batch stays 16).
if task_name in ("product_rating", "scholarly_title") and batch_size == 16:
    _per_device, _grad_accum = 4, 4
else:
    _per_device, _grad_accum = batch_size, 1

_ta_kwargs = dict(
    output_dir=str(ckpt_dir / "trainer_tmp"),
    per_device_train_batch_size=_per_device,
    gradient_accumulation_steps=_grad_accum,
    optim='adamw_torch',
    num_train_epochs=max_epoch,
    save_steps=int(1e9),                        # PATCH P6
    logging_steps=50,
    learning_rate=1e-4,
    weight_decay=1e-2,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.1,
    group_by_length=True,
    lr_scheduler_type='linear',
    report_to='none',
)
_ta_sig = set(inspect.signature(transformers.TrainingArguments.__init__).parameters)
_dropped = sorted(k for k in _ta_kwargs if k not in _ta_sig)
if _dropped:
    print(f"[PATCH P15] transformers {transformers.__version__} dropped "
          f"TrainingArguments params, omitting: {_dropped}", flush=True)
training_arguments = transformers.TrainingArguments(
    **{k: v for k, v in _ta_kwargs.items() if k in _ta_sig})

if args.base_only:
    train = []   # PATCH P17: no training corpus needed
else:
    with open(f"{args.data_root}/{task_name}/user_others.json", 'r') as f:
        train = json.load(f)

# PATCH P5: per-task test filename (tweet_paraphrase ships user_more_100_history.json)
with open(f"{args.data_root}/{task_name}/{TEST_FILE[task_name]}", 'r') as f:
    test_data = json.load(f)

# PATCH P13: smoke caps
if args.limit_train > 0:
    train = train[:args.limit_train]
if args.limit > 0:
    test_data = test_data[:args.limit]

if args.task_name == "movie_tagging":
    extract_article = extract_movie
elif args.task_name == "news_categorize":
    extract_article = extract_news_cat
elif args.task_name == "news_headline":
    extract_article = extract_news_headline
elif args.task_name == "product_rating":
    extract_article = extract_product_review        # PATCH P3: upstream NameError typo
elif args.task_name == "scholarly_title":
    extract_article = extract_scholarly_title
elif args.task_name == "tweet_paraphrase":
    extract_article = extract_tweet_paraphrasing    # PATCH P3: upstream NameError typo

with open(args.prompt_file, 'r') as f:
    prompt_template = json.load(f)

if args.add_profile:
    with open(f'{args.data_root}/{task_name}/profile_user_100.json', 'r') as f:
        test_profile = json.load(f)
    with open(f'{args.data_root}/{task_name}/profile_user_others.json', 'r') as f:
        train_profile = json.load(f)


def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


def generate_and_tokenize_prompt(data_point):
    full_prompt = data_point['full_prompt']
    tokenized_full_prompt = tokenize(full_prompt)
    user_prompt = data_point['prompt']

    tokenized_user_prompt = tokenize(
        user_prompt, add_eos_token=add_eos_token
    )
    user_prompt_len = len(tokenized_user_prompt["input_ids"])

    if add_eos_token:
        user_prompt_len -= 1

    tokenized_full_prompt["labels"] = [
        -100
    ] * user_prompt_len + tokenized_full_prompt["labels"][
        user_prompt_len:
    ]
    return tokenized_full_prompt


# training
from datasets import Dataset
if args.base_only:
    model = base_model   # PATCH P17
else:
    model = get_peft_model(base_model, peft_config)
    print_trainable_parameters(model)

pred_all = []
train_data = []

for i in tqdm(range(len(train))):
    if args.add_profile:
        profile = train_profile[i]['output']

    for idx, q in enumerate(train[i]['query']):

        if args.task_name != "citation":
            article = get_first_k_tokens(extract_article(q['input']), 768)
            prompt = prompt_template[args.task_name]['prompt'].format(article)
            full_prompt = prompt_template[args.task_name]['full_prompt'].format(get_first_k_tokens(extract_article(q['input']), 768), q['gold'])

        else:
            question = q['input']
            article = extract_citation_title(question)
            option1, option2 = extract_option(question, 1), extract_option(question, 2)

            prompt = prompt_template[args.task_name]['prompt'].format(article, option1, option2)
            full_prompt = prompt_template[args.task_name]['full_prompt'].format(article, option1, option2, q['gold'])

        if k > 0:
            visible_history_list = train[i]['profile']

            for p in visible_history_list:
                for key, value in p.items():
                    p[key] = get_first_k_tokens(str(p[key]), 368)  # PATCH P14: int fields (citation/scholarly 'date') crash .split()

            history_list = [prompt_template[args.task_name]['retrieval_history'].format(**p) for p in visible_history_list]
            tokenized_corpus = [doc.split(" ") for doc in history_list]
            bm25 = BM25Okapi(tokenized_corpus)

            tokenized_query = prompt_template[args.task_name]["retrieval_query_wokey"].format(article).split(' ')
            retrieved_history = bm25.get_top_n(tokenized_query, history_list, n=args.k)

            history_string = "".join(retrieved_history)
            prompt = history_string + "\n" + prompt
            full_prompt = history_string + "\n" + full_prompt

        if args.add_profile:
            prompt = profile + "\n" + prompt
            full_prompt = profile + "\n" + full_prompt

        train_data.append(
            {
                "prompt": prompt,
                "full_prompt": full_prompt
            }
        )

# PATCH P7: upstream `print(train_data)` (dumps the whole corpus to stdout) removed
print(f"[run_task_lora] built {len(train_data)} training examples", flush=True)

if not args.base_only:   # PATCH P17: base arm skips training entirely
    train_dataset = Dataset.from_list(train_data)
    train_dataset = train_dataset.map(generate_and_tokenize_prompt).shuffle()

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_arguments,
        data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    model.config.use_cache = False
    trainer.train()

    # PATCH P8/P9: save to our layout + weight-delta guard + meta sidecar
    delta = lora_delta_stats(model)
    if delta["lora_B_abs_sum"] == 0.0:
        print("FATAL: lora_B still all-zero after training — adapter did not train")
        sys.exit(2)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir))
    write_meta(meta_json, args, {
        "stage": "task_lora", "n_train_examples": len(train_data),
        "n_train_users": len(train), "n_test_users": len(test_data), **delta,
    })
    print(f"[run_task_lora] saved adapter to {ckpt_dir} ({delta})", flush=True)
else:
    write_meta(meta_json, args, {"stage": "base_only", "n_test_users": len(test_data)})

model.eval()
model.config.use_cache = True

transformers.set_seed(args.seed)   # PATCH P11: their sampled decoding has no seed

for i in tqdm(range(len(test_data))):
    if args.add_profile:
        profile = test_profile[i]['output']

    if k > 0:
        visible_history_list = test_data[i]['profile']
        for p in visible_history_list:
            for key, value in p.items():
                p[key] = get_first_k_tokens(str(p[key]), 368)  # PATCH P14: int fields (citation/scholarly 'date') crash .split()

        history_list = [prompt_template[args.task_name]['retrieval_history'].format(**p) for p in visible_history_list]

        tokenized_corpus = [doc.split(" ") for doc in history_list]
        bm25 = BM25Okapi(tokenized_corpus)

    test_question_list = []
    question_id_list = []

    for q in test_data[i]['query']:

        if args.task_name == 'citation':
            test_question = q['input']
            test_article = extract_citation_title(test_question)
            option1, option2 = extract_option(test_question, 1), extract_option(test_question, 2)
            test_prompt = prompt_template[args.task_name]['prompt'].format(test_article, option1, option2)

        else:
            test_question = q['input']
            test_article = extract_article(test_question)
            test_prompt = prompt_template[args.task_name]['prompt'].format(test_article)

        if k > 0:
            tokenized_query = prompt_template[args.task_name]['retrieval_query_wokey'].format(test_article).split(" ")
            retrieved_history = bm25.get_top_n(tokenized_query, history_list, n=args.k)

            history_string = "".join(retrieved_history)
            test_prompt = history_string + "\n" + test_prompt

        if args.add_profile:
            test_prompt = profile + "\n" + test_prompt

        test_question_list.append(test_prompt)
        question_id_list.append(q['id'])

    test_batch_list = split_batch(test_question_list, 1)
    out_list = []

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(test_batch_list), total=len(test_batch_list)):
            sentences = batch
            inputs = tokenizer(sentences, return_tensors="pt", padding=True, return_token_type_ids=False)
            inputs = inputs.to(model.device)

            with torch.autocast(device_type="cuda"):
                outputs = model.generate(
                    **inputs,
                    do_sample=True,
                    top_k=10,
                    temperature=args.temperature,
                    top_p=0.9,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=200
                )

            out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            out_list += out_sentence

    # PATCH P10: keep their replace-based prompt stripping, in a fresh scope
    # (upstream shadowed the user loop variable `i` here — harmless but ugly)
    for j in range(len(out_list)):
        output = out_list[j].replace(test_question_list[j], '')
        pred_all.append({
            "id": question_id_list[j],
            "output": output
        })

# PATCH P8/P10: their-format JSON at our path + per-query JSONL for stats
gold_by_id = {str(q['id']): q.get('gold') for u in test_data for q in u['query']}
export_predictions(pred_json, pred_jsonl, name2taskid[args.task_name], model_name,
                   pred_all, gold_by_id)
expected = sum(len(u['query']) for u in test_data)
print(f"[run_task_lora] wrote {len(pred_all)} predictions (expected {expected}) to {pred_json}", flush=True)
if len(pred_all) != expected or expected == 0:
    print("FATAL: prediction count mismatch / zero predictions")
    sys.exit(3)
