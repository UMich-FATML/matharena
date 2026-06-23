#!/usr/bin/env python3
"""SFT Qwen-style models from MathArena uploaded output datasets."""

import argparse
import inspect
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets, disable_caching, load_dataset


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def filter_dataset(dataset: Dataset, args: argparse.Namespace) -> Dataset:
    if not args.only_correct:
        return dataset
    return dataset.filter(lambda row: row["correct"])


def parse_dataset_ids(values: list[str]) -> list[str]:
    dataset_ids: list[str] = []
    for value in values:
        dataset_ids.extend(part.strip() for part in value.replace(",", " ").replace(":", " ").split() if part.strip())
    if not dataset_ids:
        raise ValueError("At least one outputs dataset id is required.")
    return dataset_ids


def load_outputs_datasets(dataset_ids: list[str], split: str, seed: int) -> Dataset:
    loaded_datasets = []
    for dataset_id in dataset_ids:
        dataset = load_dataset(dataset_id, split=split)
        loaded_datasets.append(dataset)
        rank_zero_print(f"Loaded SFT output dataset: {dataset_id} split={split} rows={len(dataset)}")
    if len(loaded_datasets) == 1:
        mixed_dataset = loaded_datasets[0]
    else:
        mixed_dataset = concatenate_datasets(loaded_datasets)
    return mixed_dataset.shuffle(seed=seed)


def tokenize_dataset(dataset: Dataset, tokenizer, args: argparse.Namespace) -> Dataset:
    def tokenize(row: dict[str, Any]) -> dict[str, Any]:
        messages = json.loads(row["all_messages"])
        prompt_messages = messages[:-1]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=args.enable_thinking,
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_length,
        )["input_ids"]
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        answer_tokens = len(full_ids) - prompt_len
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
            "answer_tokens": answer_tokens,
        }

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
    return tokenized.filter(lambda row: row["answer_tokens"] > 0)


@dataclass
class CausalLMCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        from torch.nn.utils.rnn import pad_sequence

        input_ids = [torch.tensor(feature["input_ids"], dtype=torch.long) for feature in features]
        attention_mask = [torch.tensor(feature["attention_mask"], dtype=torch.long) for feature in features]
        labels = [torch.tensor(feature["labels"], dtype=torch.long) for feature in features]
        return {
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id),
            "attention_mask": pad_sequence(attention_mask, batch_first=True, padding_value=0),
            "labels": pad_sequence(labels, batch_first=True, padding_value=-100),
        }


def split_dataset(dataset: Dataset, val_fraction: float, seed: int) -> tuple[Dataset, Dataset | None]:
    if val_fraction <= 0 or len(dataset) < 2:
        return dataset, None
    val_size = max(1, int(round(len(dataset) * val_fraction)))
    val_size = min(val_size, len(dataset) - 1)
    split = dataset.train_test_split(test_size=val_size, seed=seed, shuffle=True)
    return split["train"], split["test"]


def set_model_use_cache(model, value: bool) -> None:
    configs = [
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
        getattr(getattr(model, "language_model", None), "config", None),
        getattr(model, "generation_config", None),
    ]
    for config in configs:
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = value


def rank_zero_print(*values: Any, **kwargs: Any) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(*values, **kwargs)


def supports_kwargs(callable_obj: Any) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    return any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())


def filter_forward_kwargs(module: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    forward = getattr(module, "forward", module)
    if supports_kwargs(forward):
        return kwargs
    signature = inspect.signature(forward)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def first_output_tensor(outputs: Any):
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if isinstance(outputs, dict):
        if "last_hidden_state" in outputs:
            return outputs["last_hidden_state"]
        return next(iter(outputs.values()))
    return outputs[0]


def output_embeddings(model: Any):
    get_output_embeddings = getattr(model, "get_output_embeddings", None)
    if get_output_embeddings is not None:
        embeddings = get_output_embeddings()
        if embeddings is not None:
            return embeddings
    language_model = getattr(model, "language_model", None)
    if language_model is not None and hasattr(language_model, "get_output_embeddings"):
        embeddings = language_model.get_output_embeddings()
        if embeddings is not None:
            return embeddings
    return getattr(model, "lm_head", None)


def resolve_text_decoder_and_lm_head(model: Any) -> tuple[Any, Any, str]:
    lm_head = output_embeddings(model)
    if lm_head is None:
        raise ValueError("Could not find an LM head/output embedding module for chunked loss.")

    language_model = getattr(model, "language_model", None)
    candidates: list[tuple[str, Any]] = []
    if language_model is not None:
        candidates.extend(
            [
                ("language_model.model", getattr(language_model, "model", None)),
                ("language_model.transformer", getattr(language_model, "transformer", None)),
                ("language_model.decoder", getattr(language_model, "decoder", None)),
            ]
        )
    candidates.extend(
        [
            ("model", getattr(model, "model", None)),
            ("transformer", getattr(model, "transformer", None)),
            ("decoder", getattr(model, "decoder", None)),
        ]
    )

    for name, decoder in candidates:
        if decoder is not None and decoder is not model and callable(getattr(decoder, "forward", None)):
            return decoder, lm_head, name

    raise ValueError("Could not find the text decoder module for chunked loss.")


def chunked_causal_lm_loss(
    hidden_states,
    labels,
    lm_head,
    chunk_size: int,
    num_items_in_batch=None,
    ignore_index: int = -100,
):
    import torch
    import torch.nn.functional as F

    labels = labels.to(hidden_states.device)
    shift_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:]
    seq_len = hidden_states.shape[1]
    if shift_labels.shape[1] != seq_len:
        shift_labels = shift_labels[:, :seq_len]

    valid_tokens = shift_labels.ne(ignore_index).sum()
    if int(valid_tokens.detach().cpu()) == 0:
        return hidden_states.sum() * 0.0

    denominator = valid_tokens if num_items_in_batch is None else num_items_in_batch
    if not torch.is_tensor(denominator):
        denominator = torch.tensor(denominator, device=hidden_states.device)
    denominator = denominator.to(device=hidden_states.device, dtype=torch.float32).clamp_min(1.0)

    chunk_size = seq_len if chunk_size <= 0 else chunk_size
    loss_sum = hidden_states.new_zeros((), dtype=torch.float32)
    vocab_size = lm_head.weight.shape[0]
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        target = shift_labels[:, start:end]
        if not bool(target.ne(ignore_index).any().detach().cpu()):
            continue
        logits = lm_head(hidden_states[:, start:end, :]).float()
        target = target.reshape(-1).to(logits.device)
        loss_sum = loss_sum + F.cross_entropy(
            logits.reshape(-1, vocab_size),
            target,
            ignore_index=ignore_index,
            reduction="sum",
        )
    return loss_sum / denominator


def install_chunked_lm_head_loss(model: Any, chunk_size: int) -> str:
    import types

    decoder, lm_head, decoder_name = resolve_text_decoder_and_lm_head(model)
    original_forward = model.forward

    def forward_with_chunked_loss(self, *args, **kwargs):
        if args:
            return original_forward(*args, **kwargs)

        labels = kwargs.pop("labels", None)
        num_items_in_batch = kwargs.pop("num_items_in_batch", None)
        return_dict = kwargs.get("return_dict", getattr(getattr(self, "config", None), "return_dict", True))
        if labels is None:
            return original_forward(**kwargs)

        decoder_kwargs = dict(kwargs)
        decoder_kwargs.pop("logits_to_keep", None)
        decoder_kwargs.pop("num_logits_to_keep", None)
        decoder_kwargs["use_cache"] = False
        decoder_kwargs["output_attentions"] = kwargs.get("output_attentions", False)
        decoder_kwargs["output_hidden_states"] = False
        decoder_kwargs["return_dict"] = True
        decoder_outputs = decoder(**filter_forward_kwargs(decoder, decoder_kwargs))
        hidden_states = first_output_tensor(decoder_outputs)
        loss = chunked_causal_lm_loss(
            hidden_states=hidden_states,
            labels=labels,
            lm_head=lm_head,
            chunk_size=chunk_size,
            num_items_in_batch=num_items_in_batch,
        )
        if return_dict:
            return {"loss": loss}
        return (loss,)

    model._matharena_original_forward = original_forward
    model.forward = types.MethodType(forward_with_chunked_loss, model)
    return decoder_name


def freeze_parameters_outside_chunked_path(model: Any) -> tuple[int, int]:
    decoder, lm_head, _ = resolve_text_decoder_and_lm_head(model)
    trainable_param_ids = {id(parameter) for parameter in decoder.parameters()}
    trainable_param_ids.update(id(parameter) for parameter in lm_head.parameters())

    total_params = 0
    frozen_params = 0
    for parameter in model.parameters():
        total_params += parameter.numel()
        if id(parameter) in trainable_param_ids:
            continue
        if parameter.requires_grad:
            frozen_params += parameter.numel()
            parameter.requires_grad = False
    return frozen_params, total_params


def infer_fsdp_transformer_layer_cls_to_wrap(model: Any) -> list[str]:
    no_split_modules = getattr(model, "_no_split_modules", None)
    if no_split_modules:
        present_module_names = {module.__class__.__name__ for module in model.modules()}
        return [name for name in dict.fromkeys(no_split_modules) if name in present_module_names]

    layer_names: list[str] = []
    for module in model.modules():
        class_name = module.__class__.__name__
        children = getattr(module, "_modules", {})
        if class_name.endswith("DecoderLayer") and "self_attn" in children and "mlp" in children:
            layer_names.append(class_name)
    return list(dict.fromkeys(layer_names))


def fsdp_config(args: argparse.Namespace, model: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "limit_all_gathers": True,
        "use_orig_params": True,
        "forward_prefetch": False,
        "backward_prefetch": "NO_PREFETCH",
        "activation_checkpointing": False,
    }
    if args.fsdp_min_num_params > 0:
        config["min_num_params"] = args.fsdp_min_num_params
        return config

    layer_names = [
        layer_name.strip()
        for layer_name in args.fsdp_transformer_layer_cls_to_wrap.split(",")
        if layer_name.strip()
    ]
    if "auto_wrap" in args.fsdp.split() and not layer_names:
        layer_names = infer_fsdp_transformer_layer_cls_to_wrap(model)
        if not layer_names:
            raise ValueError(
                "FSDP auto_wrap was requested, but no decoder layer class could be inferred. "
                "Set --fsdp-transformer-layer-cls-to-wrap explicitly or use --fsdp-min-num-params."
            )
    if layer_names:
        config["transformer_layer_cls_to_wrap"] = layer_names
    return config


def training_args_kwargs(args: argparse.Namespace, model: Any) -> dict[str, Any]:
    import torch

    kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": args.lr_scheduler_type,
        "optim": args.optim,
        "max_grad_norm": args.max_grad_norm,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "bf16": torch.cuda.is_available(),
        "tf32": torch.cuda.is_available(),
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {
            "use_reentrant": args.gradient_checkpointing_use_reentrant,
        }
        if args.gradient_checkpointing
        else None,
        "remove_unused_columns": False,
        "prediction_loss_only": True,
        "eval_do_concat_batches": False,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": torch.cuda.is_available(),
        "report_to": args.report_to,
        "hub_model_id": args.hub_model_id,
        "push_to_hub": False,
        "eval_strategy": "steps" if args.val_fraction > 0 else "no",
        "eval_steps": args.eval_steps,
    }
    if not args.fsdp:
        kwargs["ddp_find_unused_parameters"] = args.ddp_find_unused_parameters
        kwargs["ddp_broadcast_buffers"] = args.ddp_broadcast_buffers
    if args.fsdp:
        kwargs["fsdp"] = args.fsdp
        kwargs["fsdp_config"] = fsdp_config(args, model)
    return kwargs


def push_output_dir_to_hub(output_dir: str, repo_id: str, private: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=output_dir,
        commit_message="Upload MathArena SFT checkpoint",
        ignore_patterns=["checkpoint-*", "runs/*"],
    )


def push_checkpoint_dir_to_hub(checkpoint_dir: Path, repo_id: str, private: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(checkpoint_dir),
        commit_message=f"Upload MathArena SFT checkpoint {checkpoint_dir.name}",
        ignore_patterns=[
            "optimizer.pt",
            "scheduler.pt",
            "rng_state*.pth",
            "scaler.pt",
            "trainer_state.json",
            "training_args.bin",
        ],
    )


def launch_async_checkpoint_upload(checkpoint_dir: Path, repo_id: str, private: bool):
    log_dir = Path("train/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"hf-upload-{checkpoint_dir.name}.log"
    uploader = Path(__file__).with_name("upload_sft_checkpoint.py")
    command = [
        sys.executable,
        str(uploader),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--hub-model-id",
        repo_id,
        "--private",
        str(private).lower(),
        "--save-tokenizer",
        "false",
    ]
    log_file = log_path.open("ab", buffering=0)
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    log_file.close()
    return log_path, process


def wait_for_async_uploads(upload_processes: list[tuple[Path, subprocess.Popen]]) -> None:
    for log_path, process in upload_processes:
        return_code = process.wait()
        if return_code == 0:
            rank_zero_print(f"Async Hub upload finished: log={log_path}")
        else:
            rank_zero_print(f"Async Hub upload failed with code {return_code}: log={log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dataset-id", nargs="+", required=True)
    parser.add_argument("--outputs-dataset-split", default="train")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help=(
            "Local Trainer checkpoint directory to resume from. "
            "Use 'latest', 'auto', or 'true' to resume from the latest checkpoint in output-dir."
        ),
    )
    parser.add_argument("--hub-model-id", required=True)
    parser.add_argument("--private", type=str_to_bool, default=True)
    parser.add_argument("--only-correct", type=str_to_bool, default=False)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=64000)
    parser.add_argument("--enable-thinking", type=str_to_bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-train-epochs", type=float, default=10)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--optim", default="adamw_torch_fused")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--loss-chunk-size", type=int, default=4096)
    parser.add_argument("--gradient-checkpointing", type=str_to_bool, default=True)
    parser.add_argument("--gradient-checkpointing-use-reentrant", type=str_to_bool, default=False)
    parser.add_argument("--fsdp", default="")
    parser.add_argument("--fsdp-transformer-layer-cls-to-wrap", default="")
    parser.add_argument("--fsdp-min-num-params", type=int, default=0)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--upload-checkpoints", type=str_to_bool, default=False)
    parser.add_argument("--async-checkpoint-upload", type=str_to_bool, default=True)
    parser.add_argument("--freeze-unused-chunked-parameters", type=str_to_bool, default=True)
    parser.add_argument("--ddp-find-unused-parameters", type=str_to_bool, default=True)
    parser.add_argument("--ddp-broadcast-buffers", type=str_to_bool, default=False)
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, Trainer, TrainerCallback, TrainingArguments

    # Multiple torchrun ranks run this preprocessing. Deterministic HF Datasets
    # cache files can race on shared filesystems when filter/map writes them.
    disable_caching()

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if tokenizer.model_max_length < args.max_length:
        tokenizer.model_max_length = args.max_length

    outputs_dataset_ids = parse_dataset_ids(args.outputs_dataset_id)
    dataset = load_outputs_datasets(outputs_dataset_ids, args.outputs_dataset_split, args.seed)
    dataset = filter_dataset(dataset, args)
    dataset = dataset.shuffle(seed=args.seed)
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    if len(dataset) == 0:
        raise ValueError("No SFT rows left after filtering.")
    tokenized = tokenize_dataset(dataset, tokenizer, args)
    if len(tokenized) == 0:
        raise ValueError("No SFT rows left after tokenization.")
    train_dataset, eval_dataset = split_dataset(tokenized, args.val_fraction, args.seed)
    rank_zero_print(f"SFT rows: train={len(train_dataset)} eval={0 if eval_dataset is None else len(eval_dataset)}")

    attn_implementation = None if args.attn_implementation.lower() in {"", "none"} else args.attn_implementation
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    set_model_use_cache(model, False)
    if args.gradient_checkpointing:
        gradient_checkpointing_kwargs = {"use_reentrant": args.gradient_checkpointing_use_reentrant}
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        except TypeError:
            model.gradient_checkpointing_enable()
    if args.loss_chunk_size > 0:
        decoder_name = install_chunked_lm_head_loss(model, args.loss_chunk_size)
        rank_zero_print(f"Chunked LM-head loss: chunk_size={args.loss_chunk_size} decoder={decoder_name}")
        if args.freeze_unused_chunked_parameters:
            frozen_params, model_params = freeze_parameters_outside_chunked_path(model)
            rank_zero_print(
                f"Chunked LM-head loss frozen non-text params: {frozen_params:,} / {model_params:,} "
                f"({100 * frozen_params / model_params:.2f}%)"
            )
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    rank_zero_print(
        f"Full-parameter SFT trainable params: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank_zero_print(
        "SFT runtime: "
        f"world_size={world_size} max_length={args.max_length} "
        f"per_device_train_batch_size={args.per_device_train_batch_size} "
        f"gradient_accumulation_steps={args.gradient_accumulation_steps} "
        f"fsdp={args.fsdp or 'disabled'} ddp_find_unused_parameters={args.ddp_find_unused_parameters} "
        f"ddp_broadcast_buffers={args.ddp_broadcast_buffers} "
        f"attn_implementation={attn_implementation or 'default'}"
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_args_kwargs(args, model)),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalLMCollator(tokenizer.pad_token_id),
    )
    async_upload_processes: list[tuple[Path, subprocess.Popen]] = []

    class HubCheckpointUploadCallback(TrainerCallback):
        def on_save(self, training_args, state, control, **kwargs):
            if not args.upload_checkpoints:
                return control
            checkpoint_dir = Path(training_args.output_dir) / f"checkpoint-{state.global_step}"
            if state.is_world_process_zero:
                if not checkpoint_dir.exists():
                    rank_zero_print(f"Skipping Hub upload; checkpoint dir not found: {checkpoint_dir}")
                else:
                    tokenizer.save_pretrained(checkpoint_dir)
                    if args.async_checkpoint_upload:
                        log_path, process = launch_async_checkpoint_upload(checkpoint_dir, args.hub_model_id, args.private)
                        async_upload_processes.append((log_path, process))
                        rank_zero_print(f"Started async Hub upload for {checkpoint_dir.name}; log={log_path}")
                    else:
                        rank_zero_print(f"Uploading intermediate checkpoint to Hub: {checkpoint_dir.name}")
                        try:
                            push_checkpoint_dir_to_hub(checkpoint_dir, args.hub_model_id, args.private)
                        except Exception as exc:
                            rank_zero_print(f"Intermediate checkpoint upload failed for {checkpoint_dir.name}: {exc!r}")
            if not args.async_checkpoint_upload and torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
            return control

    if args.upload_checkpoints:
        trainer.add_callback(HubCheckpointUploadCallback())

    resume_from_checkpoint = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint.lower() in {"1", "true", "yes", "latest", "auto"}:
            resume_from_checkpoint = True
        else:
            resume_from_checkpoint = args.resume_from_checkpoint
        rank_zero_print(f"Resuming SFT from checkpoint: {resume_from_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.accelerator.wait_for_everyone()
    set_model_use_cache(model, True)
    trainer.save_model(args.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        wait_for_async_uploads(async_upload_processes)
        push_output_dir_to_hub(args.output_dir, args.hub_model_id, args.private)


if __name__ == "__main__":
    main()
