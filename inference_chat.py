#!/usr/bin/env python3
"""
inference_chat.py - Interactive multi-turn chat with SmolLM2 model.
Supports conversation history, context window, and system prompts.
"""

import os
import sys
import json
from pathlib import Path

import torch

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

MODEL_DIR = Path(__file__).parent / "models"
OUTPUT_DIR = MODEL_DIR / "finetuned"


def load_model(model_path=None, dtype=torch.float16):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if model_path is None:
        model_path = OUTPUT_DIR / "final"
        if not model_path.exists():
            model_path = MODEL_DIR / "smollm2-135m-instruct"
            if not model_path.exists():
                print("ERROR: No model found!")
                print("Run train_finetune.py first or provide --model-path")
                sys.exit(1)

    model_path = str(model_path)
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
    )

    has_lora = (OUTPUT_DIR / "final").exists() and (
        OUTPUT_DIR / "final" / "adapter_config.json"
    ).exists()

    if has_lora and "final" in model_path:
        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, str(OUTPUT_DIR / "final"))
        print("LoRA adapter loaded!")
    else:
        model = base_model

    model.eval()
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    messages,
    max_new_tokens=50,
    temperature=0.7,
    top_p=0.9,
    max_context_tokens=256,
):
    """Generate response from conversation history."""
    try:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        return f"[Template error: {e}]", False

    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    if input_ids.shape[1] > max_context_tokens:
        input_ids = input_ids[:, -max_context_tokens:]
        truncated = True
    else:
        truncated = False

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        output_ids[0][input_ids.shape[1] :], skip_special_tokens=True
    )

    response = response.strip()

    if truncated:
        response += " [truncated]"

    return response, True


def chat_loop(model, tokenizer, max_new_tokens=50, temperature=0.7, system_prompt=None):
    """Interactive multi-turn chat loop."""
    print("\n" + "=" * 50)
    print("Multi-Turn Chat with SmolLM2")
    print("=" * 50)
    print("Commands:")
    print("  /quit, /exit    - Exit chat")
    print("  /clear         - Clear conversation history")
    print("  /history       - Show conversation history")
    print("  /system <text> - Set system prompt")
    print("=" * 50)
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Temperature: {temperature}")
    print()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
        print(f"System: {system_prompt}")
        print()

    turn = 0
    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.startswith("/"):
                cmd = user_input[1:].lower().split()[0]
                args = user_input[1:].split(maxsplit=1)

                if cmd in ("quit", "exit", "q"):
                    print("Bye!")
                    break

                elif cmd == "clear":
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    turn = 0
                    print("Conversation cleared.")
                    print()
                    continue

                elif cmd == "history":
                    print("\n--- Conversation History ---")
                    for i, msg in enumerate(messages):
                        role = msg.get("role", "?")
                        content = msg.get("content", "")[:100]
                        print(f"[{i + 1}] {role}: {content}...")
                    print("--- End History ---\n")
                    continue

                elif cmd == "system":
                    if len(args) > 1:
                        system_prompt = args[1]
                        if messages and messages[0].get("role") == "system":
                            messages[0] = {"role": "system", "content": system_prompt}
                        else:
                            messages.insert(
                                0, {"role": "system", "content": system_prompt}
                            )
                        print(f"System prompt set to: {system_prompt}")
                    else:
                        print("Usage: /system <text>")
                    continue

                else:
                    print(f"Unknown command: {cmd}")
                    continue

            messages.append({"role": "user", "content": user_input})
            turn += 1

            print(f"Model: [thinking...]")
            response, ok = generate_response(
                model,
                tokenizer,
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            if ok:
                messages.append({"role": "assistant", "content": response})
                print(f"Model: {response}")
            else:
                print(f"Model: {response}")

            print(f"[Turn {turn}, Messages: {len(messages)}]")
            print()

        except KeyboardInterrupt:
            print("\n\nBye!")
            break
        except (EOFError, IOError):
            print("\n\nBye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-turn chat with SmolLM2")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model or HuggingFace model ID",
    )
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="float16"
    )
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--max-context", type=int, default=256, help="Max context tokens to keep"
    )
    parser.add_argument(
        "--prompt", type=str, default=None, help="Single prompt mode (non-interactive)"
    )
    parser.add_argument(
        "--system", type=str, default=None, help="System prompt for the assistant"
    )
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    print("Loading model...")
    model, tokenizer = load_model(args.model_path, dtype)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {num_params / 1e6:.1f}M parameters")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print()

    if args.prompt:
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt})

        print(f"System: {args.system or '(none)'}")
        print(f"User: {args.prompt}")

        response, ok = generate_response(
            model,
            tokenizer,
            messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_context_tokens=args.max_context,
        )

        print(f"Model: {response}")
    else:
        chat_loop(
            model,
            tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            system_prompt=args.system,
        )


if __name__ == "__main__":
    main()
