#!/usr/bin/env python3
"""Patched copy of third_party/OPPU/OPPU.py @ 87f8c69 for SmolLM3-3B.

Loads the task-LoRA, merges it into the base (their design), then for each
test user trains a fresh per-user LoRA (their recipe verbatim: r=8, α=8, q+v,
LR=1e-4, linear, warmup 0.1, wd 1e-2, grad-norm 0.3, batch 16, 2 epochs) on
the user's profile entries and evaluates that user's queries with their
sampled decoding. Every deviation is marked `# PATCH Pn`; see PATCHES.md.

PATCH P12 adds --user-start/--user-end sharding so the K users can run as
parallel cluster jobs. Upstream runs all users sequentially in one process
by calling get_peft_model() repeatedly on the same base model; sharding at
user granularity gives every user a fresh process, which removes any
cross-user adapter-state carryover that pattern might have. A per-user
diagnostic (base-weight hash + fresh-adapter zero check) is logged either way.
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
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training

from wrapper_common import (PROJECT_ROOT, TEST_FILE, banner, refuse_overwrite,
                            write_meta, lora_delta_stats, base_weights_hash,
                            export_predictions)

parser = argparse.ArgumentParser(description="Parser for LoRA")
# PATCH P4: default model -> local SmolLM3-3B
parser.add_argument('--model_name', type=str, default=str(PROJECT_ROOT / 'data/models/SmolLM3-3B'))
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--k', type=int, default=0)
parser.add_argument('--max_step', type=int, default=5000)
parser.add_argument('--cut_off', type=int, default=2048)
parser.add_argument('--max_epoch', type=int, default=2)
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--task_name', type=str, default='movie_tagging')
parser.add_argument('--add_profile', action='store_true')
parser.add_argument('--task_lora', type=str, required=True)
parser.add_argument('--access_token', type=str, default=None)
# PATCH P8/P9/P11/P12: wrapper-layer args
parser.add_argument('--data-root', type=str, default=str(PROJECT_ROOT / 'data/oppu_release/data'))
parser.add_argument('--prompt-file', type=str, default=str(PROJECT_ROOT / 'third_party/OPPU/prompt/prompt.json'))
parser.add_argument('--ckpt-root', type=str, default=str(PROJECT_ROOT / 'train/checkpoints/oppu_rep'))
parser.add_argument('--out-root', type=str, default=str(PROJECT_ROOT / 'results/oppu_rep'))
parser.add_argument('--user-start', type=int, default=0, help='shard: first test-user index (inclusive)')
parser.add_argument('--user-end', type=int, default=-1, help='shard: last test-user index (exclusive; -1 = all)')
parser.add_argument('--tag', type=str, default='', help="output namespace tag, e.g. '_smoke' (keeps smoke artifacts off real-run paths)")
parser.add_argument('--seed', type=int, default=0, help='generation seed (their code sets none)')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--user-recipe', choices=['code', 'r5'], default='code',
                    help="PATCH P19 (recipe ablation): 'code' = their released recipe "
                         "(LR 1e-4, alpha 8, linear, warmup 0.1, 2 epochs, batch 16); "
                         "'r5' = this project's R5 user recipe verbatim (LR 1e-5, alpha 16, "
                         "cosine, warmup 0.03, 3 epochs, per-device 2 x accum 4, grad-norm 1.0)")
parser.add_argument('--eval-only', action='store_true',
                    help='PATCH P20: skip training — load each user\'s saved adapter and '
                         'only generate (seed re-decodes)')
parser.add_argument('--ckpt-tag', type=str, default='',
                    help="PATCH P20: tag of the saved checkpoints to load under --eval-only "
                         "(default '' = the original run's untagged adapters)")

args = parser.parse_args()
model_name = args.model_name
task_name = args.task_name
batch_size = args.batch_size
k = args.k
cutoff_len = args.cut_off
add_eos_token = False
max_epoch = args.max_epoch

banner("run_oppu", args)

# PATCH P5: per-task test filename
with open(f"{args.data_root}/{task_name}/{TEST_FILE[task_name]}", 'r') as f:
    test_data = json.load(f)

user_start = args.user_start
user_end = args.user_end if args.user_end >= 0 else len(test_data)
shard = f"{args.tag}_u{user_start:03d}-{user_end:03d}"

# PATCH P8: our output layout + refuse-to-overwrite
out_dir = Path(args.out_root) / task_name
pred_json = out_dir / f"oppu_k{k}{shard}_preds.json"
pred_jsonl = out_dir / f"oppu_k{k}{shard}_preds.jsonl"
meta_json = out_dir / f"oppu_k{k}{shard}_meta.json"
user_ckpt_dirs = [Path(args.ckpt_root) / task_name / f"oppu_k{k}{args.tag}_user{i:03d}"
                  for i in range(user_start, user_end)]
if args.eval_only:
    # PATCH P20: outputs are predictions only; the adapters to load must exist
    refuse_overwrite([pred_json, pred_jsonl], args.overwrite)
    load_ckpt_dirs = [Path(args.ckpt_root) / task_name / f"oppu_k{k}{args.ckpt_tag}_user{i:03d}"
                      for i in range(user_start, user_end)]
    _missing = [d for d in load_ckpt_dirs if not (d / "adapter_config.json").exists()]
    if _missing:
        print(f"FATAL: --eval-only but {len(_missing)} adapters missing, e.g. {_missing[:3]}")
        sys.exit(1)
else:
    refuse_overwrite([pred_json, pred_jsonl] + user_ckpt_dirs, args.overwrite)

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token=args.access_token)
# PATCH P4: Llama-2-specific token surgery only when those tokens exist
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

base_model.gradient_checkpointing_enable()
base_model = prepare_model_for_kbit_training(base_model)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16 if args.user_recipe == 'r5' else 8,   # PATCH P19
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
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
    output_dir=str(Path(args.ckpt_root) / task_name / f"trainer_tmp{shard}"),
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
if args.user_recipe == 'r5':
    # PATCH P19: the R5 user-stage recipe every null round R5->warm ran,
    # swapped in as a bundle; everything else (their data, prompts, trainer,
    # loss, decoding) stays their code's.
    _ta_kwargs.update(
        learning_rate=1e-5, warmup_ratio=0.03, lr_scheduler_type='cosine',
        num_train_epochs=3, per_device_train_batch_size=2,
        gradient_accumulation_steps=4, max_grad_norm=1.0)
    print("[PATCH P19] user recipe = r5 (LR 1e-5, alpha 16, cosine, wu 0.03, "
          "3 epochs, 2x4 batch, grad-norm 1.0)", flush=True)
_ta_sig = set(inspect.signature(transformers.TrainingArguments.__init__).parameters)
_dropped = sorted(k for k in _ta_kwargs if k not in _ta_sig)
if _dropped:
    print(f"[PATCH P15] transformers {transformers.__version__} dropped "
          f"TrainingArguments params, omitting: {_dropped}", flush=True)
training_arguments = transformers.TrainingArguments(
    **{k: v for k, v in _ta_kwargs.items() if k in _ta_sig})

format_flag = False
if args.task_name == "movie_tagging":
    extract_article = extract_movie
    format_flag = True
elif args.task_name == "news_categorize":
    extract_article = extract_news_cat
    format_flag = True
elif args.task_name == "news_headline":
    extract_article = extract_news_headline
    format_flag = True
elif args.task_name == "product_rating":
    extract_article = extract_product_review        # PATCH P3: upstream NameError typo
    format_flag = True
elif args.task_name == "scholarly_title":
    extract_article = extract_scholarly_title
    format_flag = True
elif args.task_name == "tweet_paraphrase":
    extract_article = extract_tweet_paraphrasing    # PATCH P3: upstream NameError typo

with open(args.prompt_file, 'r') as f:
    prompt_template = json.load(f)

if args.add_profile:
    with open(f'{args.data_root}/{task_name}/profile_user_100.json', 'r') as f:
        test_profile = json.load(f)


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
model = PeftModel.from_pretrained(model=base_model, model_id=args.task_lora, is_trainable=False)
base_model = model.merge_and_unload()
print_trainable_parameters(model)

merged_hash = base_weights_hash(base_model)   # PATCH P12: cross-user drift canary
print(f"[run_oppu] merged-base weights hash: {merged_hash}", flush=True)

pred_all = []
per_user_meta = []

for i in tqdm(range(user_start, user_end)):

    if args.eval_only:
        # PATCH P20: seed re-decode — load this user's already-trained adapter
        # and skip straight to generation. Never writes checkpoints.
        load_name = Path(args.ckpt_root) / task_name / f"oppu_k{k}{args.ckpt_tag}_user{i:03d}"
        model = PeftModel.from_pretrained(model=base_model, model_id=str(load_name),
                                          is_trainable=False)
        delta = lora_delta_stats(model)
        if delta["lora_B_abs_sum"] == 0.0:
            print(f"FATAL user {i}: loaded adapter {load_name} has all-zero lora_B")
            sys.exit(2)
        per_user_meta.append({"user_index": i, "user_id": str(test_data[i]['user_id']),
                              "loaded_adapter": str(load_name), **delta})
    else:
        train_data = []
        model = get_peft_model(base_model, peft_config)
        print_trainable_parameters(model)

        # PATCH P12: diagnostics — fresh adapter must start at zero, base unchanged
        fresh = lora_delta_stats(model)
        drift = base_weights_hash(base_model)
        if fresh["lora_B_abs_sum"] != 0.0 or drift != merged_hash:
            print(f"WARNING user {i}: fresh adapter nonzero ({fresh}) or base drift "
                  f"({drift} != {merged_hash}) — upstream get_peft_model reuse pattern", flush=True)

        if args.add_profile:
            profile = test_profile[i]['output']

        for idx, q in enumerate(test_data[i]['profile']):
            for key, value in q.items():
                q[key] = get_first_k_tokens(str(q[key]), 768)  # PATCH P14: int fields (citation/scholarly 'date') crash .split()

            prompt = prompt_template[args.task_name]['OPPU_input'].format(**q)
            full_prompt = prompt_template[args.task_name]['OPPU_full'].format(**q)

            if k > 0 and idx != 0 and format_flag == True:
                visible_history_list = test_data[i]['profile'][:idx]

                for p in visible_history_list:
                    for key, value in p.items():
                        p[key] = get_first_k_tokens(str(p[key]), 768)  # PATCH P14: int fields (citation/scholarly 'date') crash .split()

                history_list = [prompt_template[args.task_name]['retrieval_history'].format(**p) for p in visible_history_list]
                tokenized_corpus = [doc.split(" ") for doc in history_list]
                bm25 = BM25Okapi(tokenized_corpus)

                tokenized_query = prompt_template[args.task_name]["retrieval_query"].format(**q).split(' ')
                retrieved_history = bm25.get_top_n(tokenized_query, history_list, n=args.k)

                history_string = "".join(retrieved_history)
                prompt = history_string + "\n" + prompt
                full_prompt = history_string + "\n" + full_prompt

            if args.add_profile and format_flag == True:
                prompt = profile + "\n" + prompt
                full_prompt = profile + "\n" + full_prompt

            train_data.append(
                {
                    "prompt": prompt,
                    "full_prompt": full_prompt
                }
            )

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

        # PATCH P8/P9: per-user ckpt in our layout + weight-delta guard
        delta = lora_delta_stats(model)
        if delta["lora_B_abs_sum"] == 0.0:
            print(f"FATAL user {i}: lora_B still all-zero after training")
            sys.exit(2)
        output_name = Path(args.ckpt_root) / task_name / f"oppu_k{k}{args.tag}_user{i:03d}"
        output_name.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_name))
        per_user_meta.append({"user_index": i, "user_id": str(test_data[i]['user_id']),
                              "n_profile_examples": len(train_data), **delta})

    model.eval()
    model.config.use_cache = True

    transformers.set_seed(args.seed)   # PATCH P11: seed their sampled decoding

    # test inference
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

    # PATCH P10: fresh loop variable (upstream shadowed the user index `i`)
    for j in range(len(out_list)):
        output = out_list[j].replace(test_question_list[j], '')
        pred_all.append({
            "id": question_id_list[j],
            "output": output
        })

    # PATCH P12: unload this user's adapter so the next get_peft_model call
    # starts from the clean merged base (upstream relies on re-wrapping the
    # same mutated model object; the drift canary above reports if that ever
    # differs — with per-user sharding each process handles few users anyway)
    model = model.unload() if hasattr(model, "unload") else base_model

# PATCH P8/P10: their-format JSON at our path + per-query JSONL + meta
gold_by_id = {str(q['id']): q.get('gold')
              for u in test_data[user_start:user_end] for q in u['query']}
export_predictions(pred_json, pred_jsonl, name2taskid[args.task_name], model_name,
                   pred_all, gold_by_id)
expected = sum(len(u['query']) for u in test_data[user_start:user_end])
write_meta(meta_json, args, {
    "stage": "oppu_user", "task_lora": args.task_lora,
    "merged_base_hash": merged_hash, "n_users": user_end - user_start,
    "n_predictions": len(pred_all), "n_expected": expected,
    "per_user": per_user_meta,
})
print(f"[run_oppu] wrote {len(pred_all)} predictions (expected {expected}) to {pred_json}", flush=True)
if len(pred_all) != expected or expected == 0:
    print("FATAL: prediction count mismatch / zero predictions")
    sys.exit(3)
