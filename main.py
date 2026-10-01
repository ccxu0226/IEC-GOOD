import copy
import gc
import json
import os
import os.path as osp
import time
from collections.abc import Mapping
from datetime import datetime

import torch

from arguments import (
    arg_parse,
    prepare_dataset_args,
    validate_training_args,
)
from config import get_experiment_config

def build_loader(dataset, batch_size, shuffle, num_workers=0, pin_memory=False, seed=0):
    from torch_geometric.loader import DataLoader
    from utils import create_data_generator, seed_worker
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
        "worker_init_fn": seed_worker,
        "generator": create_data_generator(seed),
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **loader_kwargs)


def configure_torch_runtime():
    if not torch.cuda.is_available():
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def synchronize_device(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def format_elapsed_time(seconds):
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def save_experiment_result(args, config, metrics, script_dir):
    results_dir = osp.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    dataset_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(args.DS))
    bias = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(args.bias))
    result_path = osp.join(results_dir, f"{dataset_name}_{bias}.txt")
    hyperparameters = json.dumps(
        {"runtime_and_data": vars(args), "experiment_config": config.as_dict()},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    current_time = datetime.now().astimezone().isoformat(timespec="seconds")

    with open(result_path, "a", encoding="utf-8") as result_file:
        result_file.write(f"hyperparameters={hyperparameters}\n")

        if metrics.get("temporal_test", False):
            parts = [
                f"time={current_time}",
                f"validation_macro_f1={metrics['val_macro_f1']:.6f} +/- {metrics['val_macro_f1_std']:.6f}",
                f"validation_balanced_accuracy={metrics['val_balanced_accuracy']:.6f} +/- {metrics['val_balanced_accuracy_std']:.6f}",
                f"validation_recall={metrics['val_recall']:.6f} +/- {metrics['val_recall_std']:.6f}",
                f"validation_precision={metrics['val_precision']:.6f} +/- {metrics['val_precision_std']:.6f}",
            ]

            for test_name in metrics["test_names"]:
                test_metrics = metrics["test_metrics_by_environment"][test_name]
                parts.append(f"test_{test_name}_macro_f1={test_metrics['macro_f1']:.6f} +/- {test_metrics['macro_f1_std']:.6f}")
                parts.append(f"test_{test_name}_balanced_accuracy={test_metrics['balanced_accuracy']:.6f} +/- {test_metrics['balanced_accuracy_std']:.6f}")
                parts.append(f"test_{test_name}_recall={test_metrics['recall']:.6f} +/- {test_metrics['recall_std']:.6f}")
                parts.append(f"test_{test_name}_precision={test_metrics['precision']:.6f} +/- {test_metrics['precision_std']:.6f}")

            parts.append(f"average_test_macro_f1={metrics['average_test_macro_f1']:.6f} +/- {metrics['average_test_macro_f1_std']:.6f}")
            parts.append(f"average_test_balanced_accuracy={metrics['average_test_balanced_accuracy']:.6f} +/- {metrics['average_test_balanced_accuracy_std']:.6f}")
            parts.append(f"average_test_recall={metrics['average_test_recall']:.6f} +/- {metrics['average_test_recall_std']:.6f}")
            parts.append(f"average_test_precision={metrics['average_test_precision']:.6f} +/- {metrics['average_test_precision_std']:.6f}")
            result_file.write(", ".join(parts) + "\n")
        else:
            result_file.write(
                f"time={current_time}, "
                f"validation_macro_f1={metrics['val_macro_f1']:.6f} +/- {metrics['val_macro_f1_std']:.6f}, "
                f"validation_balanced_accuracy={metrics['val_balanced_accuracy']:.6f} +/- {metrics['val_balanced_accuracy_std']:.6f}, "
                f"validation_recall={metrics['val_recall']:.6f} +/- {metrics['val_recall_std']:.6f}, "
                f"validation_precision={metrics['val_precision']:.6f} +/- {metrics['val_precision_std']:.6f}, "
                f"test_macro_f1={metrics['test_macro_f1']:.6f} +/- {metrics['test_macro_f1_std']:.6f}, "
                f"test_balanced_accuracy={metrics['test_balanced_accuracy']:.6f} +/- {metrics['test_balanced_accuracy_std']:.6f}, "
                f"test_recall={metrics['test_recall']:.6f} +/- {metrics['test_recall_std']:.6f}, "
                f"test_precision={metrics['test_precision']:.6f} +/- {metrics['test_precision_std']:.6f}\n"
            )

    return result_path


def print_pretraining_configuration(config, runtime_args, dataset_name, input_dim, device, num_graphs):
    print("=" * 70)
    print("Stage 1: IEC-GOOD self-supervised pretraining")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {device}")
    print(f"Unlabeled graphs: {num_graphs}")
    print(f"Input dimension: {input_dim}")
    print(f"Hidden dimension: {config.hidden_dim}")
    print(f"GIN layers: {config.num_gc_layers}")
    print(f"Projection dimension: {config.projection_dim}")
    print(f"Maximum regions: {config.num_regions}")
    print(f"Minimum region size: {config.min_region_size}")
    print(f"Maximum region coverage: {config.max_region_coverage}")
    print(f"Semantic K-means clusters: {config.semantic_clusters}")
    print(f"Context K-means clusters: {config.context_clusters}")
    print(f"Cluster refresh interval: {config.cluster_refresh_interval}")
    print(f"Joint contribution balance lambda_jnt: {config.joint_balance}")
    print("EMA target: original graph representation")
    print(f"EMA momentum: {config.ema_momentum}")
    print("Semantic support: latent environment replacement")
    print(f"Environment fusion weight: {config.environment_fusion_weight}")
    print(f"Coalition weight: {config.coalition_weight}")
    print(f"Invariance weight: {config.invariance_weight}")
    print(f"Uniformity weight: {config.uniformity_weight}")
    print(f"Uniformity temperature: {config.uniformity_temperature}")
    print(f"Warmup epochs: {config.warmup_epochs}")
    print(f"Pretraining learning rate: {config.learning_rate}")
    print(f"Pretraining weight decay: {config.weight_decay}")
    print(f"Batch size: {config.batch_size}")
    print(f"Pretraining workers: {runtime_args.num_workers}")
    print(f"Pretraining epochs: {config.pretrain_epochs}")
    print("=" * 70)


def print_finetuning_configuration(config, runtime_args, num_classes, train_size, val_size, test_size, temporal_test_sizes=None):
    print("=" * 70)
    print("Stage 2: supervised end-to-end fine-tuning")
    print(f"Training graphs: {train_size}")
    print(f"Validation graphs: {val_size}")

    if temporal_test_sizes:
        print(f"Total OOD test graphs: {test_size}")
        for test_name, size in temporal_test_sizes.items():
            print(f'OOD test {test_name} graphs: {size}')
    else:
        print(f"OOD test graphs: {test_size}")

    print(f"Number of classes: {num_classes}")
    print(f"Fine-tuning learning rate: {config.learning_rate}")
    print(f"Fine-tuning weight decay: {config.weight_decay}")
    print(f"Fine-tuning batch size: {config.batch_size}")
    print(f"Fine-tuning workers: {runtime_args.num_workers}")
    print(f"Fine-tuning epochs: {config.finetune_epochs}")
    print("=" * 70)


def build_test_loaders(bundle, batch_size, num_workers, pin_memory, seed=0):
    temporal_test_datasets = getattr(bundle, "temporal_test_datasets", None)

    if not temporal_test_datasets:
        return build_loader(
            dataset=bundle.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        )

    return {
        str(test_name): build_loader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed + index,
        )
        for index, (test_name, dataset) in enumerate(temporal_test_datasets.items())
    }


def shutdown_test_loaders(test_loader):
    from utils import shutdown_loader_workers
    if isinstance(test_loader, Mapping):
        for loader in test_loader.values():
            shutdown_loader_workers(loader)
        return

    shutdown_loader_workers(test_loader)


def print_test_label_statistics(train_classes, val_classes, bundle):
    from utils import print_label_statistics, validate_dataset_classes
    temporal_test_datasets = getattr(bundle, "temporal_test_datasets", None)

    if not temporal_test_datasets:
        test_classes = print_label_statistics("OOD Test", bundle.test_dataset)
        validate_dataset_classes(train_classes, val_classes, test_classes)
        return

    environment_classes = []

    for test_name, dataset in temporal_test_datasets.items():
        classes = print_label_statistics(f"OOD Test {test_name}", dataset)
        environment_classes.append(classes)

    test_classes = torch.unique(torch.cat(environment_classes), sorted=True)
    validate_dataset_classes(train_classes, val_classes, test_classes)

def print_final_results(metrics):
    print("=" * 110)
    print("IEC-GOOD pretraining and downstream fine-tuning finished.")
    print(f"Fine-tuning runs: {metrics['num_runs']}")
    print(f"Validation Macro-F1: {metrics['val_macro_f1']:.4f} +/- {metrics['val_macro_f1_std']:.4f}")
    print(f"Validation Balanced Accuracy: {metrics['val_balanced_accuracy']:.4f} +/- {metrics['val_balanced_accuracy_std']:.4f}")
    print(f"Validation Recall: {metrics['val_recall']:.4f} +/- {metrics['val_recall_std']:.4f}")
    print(f"Validation Precision: {metrics['val_precision']:.4f} +/- {metrics['val_precision_std']:.4f}")

    if metrics.get("temporal_test", False):
        for test_name in metrics["test_names"]:
            test_metrics = metrics["test_metrics_by_environment"][test_name]
            print(f"Test {test_name} Macro-F1: {test_metrics['macro_f1']:.4f} +/- {test_metrics['macro_f1_std']:.4f}")
            print(f"Test {test_name} Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f} +/- {test_metrics['balanced_accuracy_std']:.4f}")
            print(f"Test {test_name} Recall: {test_metrics['recall']:.4f} +/- {test_metrics['recall_std']:.4f}")
            print(f"Test {test_name} Precision: {test_metrics['precision']:.4f} +/- {test_metrics['precision_std']:.4f}")

        print(f"Average Test Macro-F1: {metrics['average_test_macro_f1']:.4f} +/- {metrics['average_test_macro_f1_std']:.4f}")
        print(f"Average Test Balanced Accuracy: {metrics['average_test_balanced_accuracy']:.4f} +/- {metrics['average_test_balanced_accuracy_std']:.4f}")
        print(f"Average Test Recall: {metrics['average_test_recall']:.4f} +/- {metrics['average_test_recall_std']:.4f}")
        print(f"Average Test Precision: {metrics['average_test_precision']:.4f} +/- {metrics['average_test_precision_std']:.4f}")
    else:
        print(f"Test Macro-F1: {metrics['test_macro_f1']:.4f} +/- {metrics['test_macro_f1_std']:.4f}")
        print(f"Test Balanced Accuracy: {metrics['test_balanced_accuracy']:.4f} +/- {metrics['test_balanced_accuracy_std']:.4f}")
        print(f"Test Recall: {metrics['test_recall']:.4f} +/- {metrics['test_recall_std']:.4f}")
        print(f"Test Precision: {metrics['test_precision']:.4f} +/- {metrics['test_precision_std']:.4f}")

    print("=" * 110)


def main():
    args = arg_parse()
    configure_torch_runtime()

    script_dir = osp.dirname(osp.realpath(__file__))
    args = prepare_dataset_args(args, script_dir)
    validate_training_args(args)
    config = get_experiment_config(args.DS)

    from aug import ensure_node_features
    from eval import finetune_graph_classifier
    from iec_good import IECGOOD, IECGOODTrainer
    from utils import (
        UnlabeledGraphDataset,
        load_dataset_bundle,
        print_label_statistics,
        setup_seed,
        shutdown_loader_workers,
        validate_dataset_classes,
    )
    setup_seed(args.seed)

    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    pin_memory = device.type == "cuda"

    print("=" * 70)
    print(f"Dataset: {args.DS}")
    print(f"Dataset root: {args.root}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print("=" * 70)

    program_start_time = time.perf_counter()

    bundle = load_dataset_bundle(args)
    train_dataset = bundle.train_dataset
    val_dataset = bundle.val_dataset
    test_dataset = bundle.test_dataset
    temporal_test_datasets = getattr(bundle, "temporal_test_datasets", None)

    if len(train_dataset) == 0:
        raise RuntimeError("The unlabeled pretraining dataset is empty.")

    if len(val_dataset) == 0 or len(test_dataset) == 0:
        raise RuntimeError("Validation and OOD test datasets must be non-empty.")

    if temporal_test_datasets:
        for test_name, dataset in temporal_test_datasets.items():
            if len(dataset) == 0:
                raise RuntimeError(f"Temporal OOD test dataset '{test_name}' is empty.")

    first_graph = ensure_node_features(train_dataset[0].clone())
    input_dim = int(first_graph.x.size(-1))
    del first_graph

    unlabeled_dataset = UnlabeledGraphDataset(train_dataset)
    pretrain_loader = build_loader(
        dataset=unlabeled_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        seed=args.seed,
    )

    model = IECGOOD(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        num_gc_layers=config.num_gc_layers,
        projection_dim=config.projection_dim,
        num_regions=config.num_regions,
        min_region_size=config.min_region_size,
        max_region_coverage=config.max_region_coverage,
        region_temperature=config.region_temperature,
    ).to(device)
    optimizer = torch.optim.Adam(model.online_parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    trainer = IECGOODTrainer(model=model, optimizer=optimizer, device=device, config=config, seed=args.seed)

    print_pretraining_configuration(
        config=config,
        runtime_args=args,
        dataset_name=args.DS,
        input_dim=input_dim,
        device=device,
        num_graphs=len(unlabeled_dataset),
    )

    synchronize_device(device)
    pretrain_start_time = time.perf_counter()
    trainer.pretrain(loader=pretrain_loader, epochs=config.pretrain_epochs)
    synchronize_device(device)
    pretrain_elapsed_time = time.perf_counter() - pretrain_start_time

    print("=" * 70)
    print(f"Total pretraining time: {format_elapsed_time(pretrain_elapsed_time)} ({pretrain_elapsed_time:.2f}s)")
    print("=" * 70)

    shutdown_loader_workers(pretrain_loader)
    del pretrain_loader
    del unlabeled_dataset
    del optimizer
    del trainer

    pretrained_encoder = copy.deepcopy(model.online_encoder).cpu()
    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_classes = print_label_statistics("Train", train_dataset)
    val_classes = print_label_statistics("Validation", val_dataset)
    print_test_label_statistics(train_classes, val_classes, bundle)
    num_classes = int(train_classes.numel())

    if int(bundle.input_dim) != input_dim:
        print(f"Warning: bundle.input_dim={bundle.input_dim}, observed input dimension={input_dim}. Using the observed dimension.")

    if int(bundle.num_classes) != num_classes:
        print(f"Warning: bundle.num_classes={bundle.num_classes}, observed class count={num_classes}. Using the observed class count.")

    finetune_train_loader = build_loader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        seed=args.seed + 1000,
    )
    val_loader = build_loader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        seed=args.seed + 2000,
    )
    test_loader = build_test_loaders(
        bundle=bundle,
        batch_size=config.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        seed=args.seed + 3000,
    )

    temporal_test_sizes = {str(test_name): len(dataset) for test_name, dataset in temporal_test_datasets.items()} if temporal_test_datasets else None

    print_finetuning_configuration(
        config=config,
        runtime_args=args,
        num_classes=num_classes,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        test_size=len(test_dataset),
        temporal_test_sizes=temporal_test_sizes,
    )

    synchronize_device(device)
    finetune_start_time = time.perf_counter()

    metrics = finetune_graph_classifier(
        pretrained_encoder=pretrained_encoder,
        train_loader=finetune_train_loader,
        train_eval_loader=None,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        num_classes=num_classes,
        epochs=config.finetune_epochs,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        seeds=range(config.num_finetune_runs),
    )

    synchronize_device(device)
    finetune_elapsed_time = time.perf_counter() - finetune_start_time

    shutdown_loader_workers(finetune_train_loader)
    shutdown_loader_workers(val_loader)
    shutdown_test_loaders(test_loader)

    total_elapsed_time = time.perf_counter() - program_start_time

    print_final_results(metrics)

    result_path = save_experiment_result(args=args, config=config, metrics=metrics, script_dir=script_dir)

    print(f"Result saved to: {result_path}")
    print(f"Total pretraining time: {format_elapsed_time(pretrain_elapsed_time)} ({pretrain_elapsed_time:.2f}s)")
    print(f"Total fine-tuning time: {format_elapsed_time(finetune_elapsed_time)} ({finetune_elapsed_time:.2f}s)")
    print(f"Total running time: {format_elapsed_time(total_elapsed_time)} ({total_elapsed_time:.2f}s)")
    print("=" * 70)


if __name__ == "__main__":
    main()
