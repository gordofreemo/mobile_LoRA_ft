#!/usr/bin/env python3
"""
Interactive chat session with SmolLM3-3B (optionally with a LoRA adapter).

Usage:
    python condor/chat.py
    python condor/chat.py --adapter /path/to/lora/checkpoint
    python condor/chat.py --system "You are a helpful assistant."

Commands during chat:
    /quit or /exit  — end the session
    /reset          — clear conversation history
    /system <text>  — change the system prompt mid-session
"""

import argparse
import os
import sys
from pathlib import Path

MODEL_DIR = os.environ.get(
    "MODEL_OUT_DIR",
    "/home/ange00008/projects/mobileFT_distill/data/models/SmolLM3-3B",
)

DEFAULT_SYSTEM = "You are a helpful assistant."
MAX_NEW_TOKENS = 2048


def load_model(model_dir: str, adapter_path: str | None):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print(f"Loading tokenizer from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model onto {device} (bf16) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map=device,
    )

    if adapter_path:
        print(f"Loading LoRA adapter from {adapter_path} ...")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print("Adapter merged.")

    model.eval()
    print("Model ready.\n")
    return tokenizer, model, device


def generate(tokenizer, model, device, messages: list[dict], system: str) -> str:
    import torch

    full_messages = [{"role": "system", "content": system}] + messages
    prompt = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_len:]

    # If we generated the full budget and the last token isn't EOS, the model
    # was cut off mid-response rather than finishing on its own.
    hit_limit = (
        len(new_tokens) >= MAX_NEW_TOKENS
        and new_tokens[-1].item() != tokenizer.eos_token_id
    )
    if hit_limit:
        print(
            f"[Output truncated — hit the {MAX_NEW_TOKENS}-token limit. "
            f"Raise MAX_NEW_TOKENS for longer replies.]"
        )

    return tokenizer.decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None, help="Path to LoRA checkpoint")
    parser.add_argument("--system", default=DEFAULT_SYSTEM, help="System prompt")
    parser.add_argument("--model-dir", default=MODEL_DIR, help="Model weights directory")
    args = parser.parse_args()

    if not Path(args.model_dir).exists():
        print(f"ERROR: model directory not found: {args.model_dir}", file=sys.stderr)
        print("Run condor/download_model.sub first.", file=sys.stderr)
        sys.exit(1)

    tokenizer, model, device = load_model(args.model_dir, args.adapter)

    system = args.system
    history: list[dict] = []

    print("=" * 60)
    print("SmolLM3-3B chat session  (type /quit to exit, /reset to clear history)")
    print(f"System: {system}")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input in ("/quit", "/exit"):
            print("Bye.")
            break

        if user_input == "/reset":
            history.clear()
            print("[History cleared]")
            continue

        if user_input.startswith("/system "):
            system = user_input[len("/system "):].strip()
            history.clear()
            print(f"[System prompt updated. History cleared.]\nSystem: {system}")
            continue

        history.append({"role": "user", "content": user_input})
        reply = generate(tokenizer, model, device, history, system)
        history.append({"role": "assistant", "content": reply})

        print(f"\nModel: {reply}")


if __name__ == "__main__":
    main()
