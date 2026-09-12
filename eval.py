import copy
import random
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score


class GraphClassifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        if not hasattr(encoder, 'embedding_dim'):
            raise AttributeError('The pretrained encoder must expose embedding_dim.')
        if not hasattr(encoder, 'num_features'):
            raise AttributeError('The pretrained encoder must expose num_features.')
        if int(num_classes) < 2:
            raise ValueError(f'num_classes must be at least 2, got {num_classes}.')
        self.encoder = encoder
        self.num_classes = int(num_classes)
        self.classifier = nn.Linear(encoder.embedding_dim, self.num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, data):
        x = getattr(data, "x", None)

        if x is None:
            x = torch.ones((int(data.num_nodes), int(self.encoder.num_features)), dtype=torch.float32, device=data.edge_index.device)
        else:
            x = x.float()
            if x.dim() == 1:
                x = x.unsqueeze(-1)

        if x.dim() != 2:
            raise ValueError(f'Node features must have shape [N, F], got {x.shape}.')
        if x.size(1) != self.encoder.num_features:
            raise ValueError(f'Node-feature dimension mismatch: expected {self.encoder.num_features}, got {x.size(1)}.')

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(int(data.num_nodes), dtype=torch.long, device=data.edge_index.device)

        graph_embeddings, _ = self.encoder(x, data.edge_index.long(), batch)
        logits = self.classifier(graph_embeddings)

        if not torch.isfinite(logits).all():
            raise RuntimeError('The graph classifier produced non-finite logits.')

        return logits


def train_classifier_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_graphs = 0

    for data in loader:
        data = data.to(device, non_blocking=True)
        labels = _prepare_labels(data.y, model.num_classes, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(data)

        if logits.size(0) != labels.numel():
            raise RuntimeError(f'Classifier output count does not match labels: {logits.size(0)} != {labels.numel()}.')

        loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite fine-tuning loss detected: {float(loss.detach())}.')

        loss.backward()
        optimizer.step()

        num_graphs = int(labels.numel())
        total_loss += float(loss.detach()) * num_graphs
        total_graphs += num_graphs

    if total_graphs == 0:
        raise RuntimeError('No graph was processed during fine-tuning.')

    return {"loss": total_loss / total_graphs}


@torch.no_grad()
def evaluate_classifier(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_graphs = 0
    all_labels = []
    all_predictions = []

    for data in loader:
        data = data.to(device, non_blocking=True)
        labels = _prepare_labels(data.y, model.num_classes, device)
        logits = model(data)

        if logits.size(0) != labels.numel():
            raise RuntimeError(f'Classifier output count does not match labels: {logits.size(0)} != {labels.numel()}.')

        loss = F.cross_entropy(logits, labels, reduction="sum")
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite evaluation loss detected: {float(loss)}.')

        predictions = logits.argmax(dim=-1)
        total_loss += float(loss)
        total_graphs += int(labels.numel())
        all_labels.append(labels.detach().cpu())
        all_predictions.append(predictions.detach().cpu())

    if total_graphs == 0:
        raise RuntimeError('Cannot evaluate an empty dataset.')

    labels = torch.cat(all_labels).numpy()
    predictions = torch.cat(all_predictions).numpy()
    class_labels = list(range(model.num_classes))
    macro_f1 = float(f1_score(labels, predictions, labels=class_labels, average="macro", zero_division=0))
    balanced_accuracy = float(balanced_accuracy_score(labels, predictions))

    if model.num_classes == 2:
        recall = float(recall_score(labels, predictions, pos_label=1, zero_division=0))
        precision = float(precision_score(labels, predictions, pos_label=1, zero_division=0))
    else:
        recall = float(recall_score(labels, predictions, labels=class_labels, average="macro", zero_division=0))
        precision = float(precision_score(labels, predictions, labels=class_labels, average="macro", zero_division=0))

    return {"loss": total_loss / total_graphs, "macro_f1": macro_f1, "balanced_accuracy": balanced_accuracy, "recall": recall, "precision": precision}


def finetune_graph_classifier(
    pretrained_encoder,
    train_loader,
    train_eval_loader,
    val_loader,
    test_loader,
    device,
    num_classes,
    epochs,
    lr,
    weight_decay,
    seeds=range(10),
):
    epochs = int(epochs)
    lr = float(lr)
    weight_decay = float(weight_decay)
    num_classes = int(num_classes)
    seeds = tuple(int(seed) for seed in seeds)

    if epochs < 1:
        raise ValueError(f'epochs must be positive, got {epochs}.')
    if lr <= 0.0:
        raise ValueError(f'lr must be positive, got {lr}.')
    if weight_decay < 0.0:
        raise ValueError(f'weight_decay must be non-negative, got {weight_decay}.')
    if not seeds:
        raise ValueError('At least one fine-tuning seed is required.')
    if len(set(seeds)) != len(seeds):
        raise ValueError('Fine-tuning seeds must be unique.')

    del train_eval_loader
    temporal_test = isinstance(test_loader, Mapping)
    recall_name = "Malware Recall" if num_classes == 2 else "Macro Recall"
    precision_name = "Malware Precision" if num_classes == 2 else "Macro Precision"

    if temporal_test:
        if not test_loader:
            raise ValueError('Temporal test loaders cannot be empty.')
        test_names = [str(name) for name in test_loader.keys()]
        for name, loader in test_loader.items():
            if loader is None:
                raise ValueError(f"Temporal test loader '{name}' is None.")
    else:
        if test_loader is None:
            raise ValueError('test_loader cannot be None.')
        test_names = []

    runs = []

    for run_index, seed in enumerate(seeds, start=1):
        _set_random_seed(seed)
        _seed_loader(train_loader, seed)
        _seed_loader(val_loader, seed)

        if temporal_test:
            for loader in test_loader.values():
                _seed_loader(loader, seed)
        else:
            _seed_loader(test_loader, seed)

        result = _finetune_graph_classifier_once(
            pretrained_encoder=pretrained_encoder,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            num_classes=num_classes,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed,
        )
        runs.append(result)

        if temporal_test:
            temporal_text = " | ".join(f"Test {name} F1={result['test_metrics_by_environment'][name]['macro_f1']:.4f} BAcc={result['test_metrics_by_environment'][name]['balanced_accuracy']:.4f} Recall={result['test_metrics_by_environment'][name]['recall']:.4f} Precision={result['test_metrics_by_environment'][name]['precision']:.4f}" for name in test_names)
            print(f"Finetune Run | Run={run_index:02d}/{len(seeds):02d} | Seed={seed} | Best Epoch={result['epoch']:03d} | Val F1={result['val_macro_f1']:.4f} BAcc={result['val_balanced_accuracy']:.4f} Recall={result['val_recall']:.4f} Precision={result['val_precision']:.4f} | {temporal_text} | Avg F1={result['test_macro_f1']:.4f} BAcc={result['test_balanced_accuracy']:.4f} Recall={result['test_recall']:.4f} Precision={result['test_precision']:.4f}")
        else:
            print(f"Finetune Run | Run={run_index:02d}/{len(seeds):02d} | Seed={seed} | Best Epoch={result['epoch']:03d} | Val F1={result['val_macro_f1']:.4f} BAcc={result['val_balanced_accuracy']:.4f} Recall={result['val_recall']:.4f} Precision={result['val_precision']:.4f} | Test F1={result['test_macro_f1']:.4f} BAcc={result['test_balanced_accuracy']:.4f} Recall={result['test_recall']:.4f} Precision={result['test_precision']:.4f}")

    val_macro_f1_mean, val_macro_f1_std = _mean_and_std([run["val_macro_f1"] for run in runs])
    val_balanced_accuracy_mean, val_balanced_accuracy_std = _mean_and_std([run["val_balanced_accuracy"] for run in runs])
    val_recall_mean, val_recall_std = _mean_and_std([run["val_recall"] for run in runs])
    val_precision_mean, val_precision_std = _mean_and_std([run["val_precision"] for run in runs])
    test_macro_f1_mean, test_macro_f1_std = _mean_and_std([run["test_macro_f1"] for run in runs])
    test_balanced_accuracy_mean, test_balanced_accuracy_std = _mean_and_std([run["test_balanced_accuracy"] for run in runs])
    test_recall_mean, test_recall_std = _mean_and_std([run["test_recall"] for run in runs])
    test_precision_mean, test_precision_std = _mean_and_std([run["test_precision"] for run in runs])

    if temporal_test:
        test_metrics_by_environment = {}

        for name in test_names:
            environment_macro_f1_mean, environment_macro_f1_std = _mean_and_std([run["test_metrics_by_environment"][name]["macro_f1"] for run in runs])
            environment_balanced_accuracy_mean, environment_balanced_accuracy_std = _mean_and_std([run["test_metrics_by_environment"][name]["balanced_accuracy"] for run in runs])
            environment_recall_mean, environment_recall_std = _mean_and_std([run["test_metrics_by_environment"][name]["recall"] for run in runs])
            environment_precision_mean, environment_precision_std = _mean_and_std([run["test_metrics_by_environment"][name]["precision"] for run in runs])
            test_metrics_by_environment[name] = {"macro_f1": environment_macro_f1_mean, "macro_f1_std": environment_macro_f1_std, "balanced_accuracy": environment_balanced_accuracy_mean, "balanced_accuracy_std": environment_balanced_accuracy_std, "recall": environment_recall_mean, "recall_std": environment_recall_std, "precision": environment_precision_mean, "precision_std": environment_precision_std}

        print("=" * 110)
        print(f"Validation Macro-F1={val_macro_f1_mean:.4f} +/- {val_macro_f1_std:.4f} | Balanced Acc={val_balanced_accuracy_mean:.4f} +/- {val_balanced_accuracy_std:.4f} | {recall_name}={val_recall_mean:.4f} +/- {val_recall_std:.4f} | {precision_name}={val_precision_mean:.4f} +/- {val_precision_std:.4f}")

        for name in test_names:
            metrics = test_metrics_by_environment[name]
            print(f"Test {name} Macro-F1={metrics['macro_f1']:.4f} +/- {metrics['macro_f1_std']:.4f} | Balanced Acc={metrics['balanced_accuracy']:.4f} +/- {metrics['balanced_accuracy_std']:.4f} | {recall_name}={metrics['recall']:.4f} +/- {metrics['recall_std']:.4f} | {precision_name}={metrics['precision']:.4f} +/- {metrics['precision_std']:.4f}")

        print(f"Average Test Macro-F1={test_macro_f1_mean:.4f} +/- {test_macro_f1_std:.4f} | Balanced Acc={test_balanced_accuracy_mean:.4f} +/- {test_balanced_accuracy_std:.4f} | {recall_name}={test_recall_mean:.4f} +/- {test_recall_std:.4f} | {precision_name}={test_precision_mean:.4f} +/- {test_precision_std:.4f}")
        print("=" * 110)

        print("Results (%)")
        print(f"Validation Macro-F1={100.0 * val_macro_f1_mean:.2f} +/- {100.0 * val_macro_f1_std:.2f} | Balanced Acc={100.0 * val_balanced_accuracy_mean:.2f} +/- {100.0 * val_balanced_accuracy_std:.2f} | {recall_name}={100.0 * val_recall_mean:.2f} +/- {100.0 * val_recall_std:.2f} | {precision_name}={100.0 * val_precision_mean:.2f} +/- {100.0 * val_precision_std:.2f}")

        for name in test_names:
            metrics = test_metrics_by_environment[name]
            print(f"Test {name} Macro-F1={100.0 * metrics['macro_f1']:.2f} +/- {100.0 * metrics['macro_f1_std']:.2f} | Balanced Acc={100.0 * metrics['balanced_accuracy']:.2f} +/- {100.0 * metrics['balanced_accuracy_std']:.2f} | {recall_name}={100.0 * metrics['recall']:.2f} +/- {100.0 * metrics['recall_std']:.2f} | {precision_name}={100.0 * metrics['precision']:.2f} +/- {100.0 * metrics['precision_std']:.2f}")

        print(f"Average Test Macro-F1={100.0 * test_macro_f1_mean:.2f} +/- {100.0 * test_macro_f1_std:.2f} | Balanced Acc={100.0 * test_balanced_accuracy_mean:.2f} +/- {100.0 * test_balanced_accuracy_std:.2f} | {recall_name}={100.0 * test_recall_mean:.2f} +/- {100.0 * test_recall_std:.2f} | {precision_name}={100.0 * test_precision_mean:.2f} +/- {100.0 * test_precision_std:.2f}")
        print("=" * 110)

        return {
            "num_runs": len(runs),
            "seeds": seeds,
            "runs": runs,
            "temporal_test": True,
            "test_names": tuple(test_names),
            "val_macro_f1": val_macro_f1_mean,
            "val_macro_f1_std": val_macro_f1_std,
            "val_balanced_accuracy": val_balanced_accuracy_mean,
            "val_balanced_accuracy_std": val_balanced_accuracy_std,
            "val_recall": val_recall_mean,
            "val_recall_std": val_recall_std,
            "val_precision": val_precision_mean,
            "val_precision_std": val_precision_std,
            "test_macro_f1": test_macro_f1_mean,
            "test_macro_f1_std": test_macro_f1_std,
            "test_balanced_accuracy": test_balanced_accuracy_mean,
            "test_balanced_accuracy_std": test_balanced_accuracy_std,
            "test_recall": test_recall_mean,
            "test_recall_std": test_recall_std,
            "test_precision": test_precision_mean,
            "test_precision_std": test_precision_std,
            "average_test_macro_f1": test_macro_f1_mean,
            "average_test_macro_f1_std": test_macro_f1_std,
            "average_test_balanced_accuracy": test_balanced_accuracy_mean,
            "average_test_balanced_accuracy_std": test_balanced_accuracy_std,
            "average_test_recall": test_recall_mean,
            "average_test_recall_std": test_recall_std,
            "average_test_precision": test_precision_mean,
            "average_test_precision_std": test_precision_std,
            "test_metrics_by_environment": test_metrics_by_environment,
        }

    print("=" * 110)
    print(f"Validation Macro-F1={val_macro_f1_mean:.4f} +/- {val_macro_f1_std:.4f} | Balanced Acc={val_balanced_accuracy_mean:.4f} +/- {val_balanced_accuracy_std:.4f} | {recall_name}={val_recall_mean:.4f} +/- {val_recall_std:.4f} | {precision_name}={val_precision_mean:.4f} +/- {val_precision_std:.4f}")
    print(f"Test Macro-F1={test_macro_f1_mean:.4f} +/- {test_macro_f1_std:.4f} | Balanced Acc={test_balanced_accuracy_mean:.4f} +/- {test_balanced_accuracy_std:.4f} | {recall_name}={test_recall_mean:.4f} +/- {test_recall_std:.4f} | {precision_name}={test_precision_mean:.4f} +/- {test_precision_std:.4f}")
    print("=" * 110)

    print("Results (%)")
    print(f"Validation Macro-F1={100.0 * val_macro_f1_mean:.2f} +/- {100.0 * val_macro_f1_std:.2f} | Balanced Acc={100.0 * val_balanced_accuracy_mean:.2f} +/- {100.0 * val_balanced_accuracy_std:.2f} | {recall_name}={100.0 * val_recall_mean:.2f} +/- {100.0 * val_recall_std:.2f} | {precision_name}={100.0 * val_precision_mean:.2f} +/- {100.0 * val_precision_std:.2f}")
    print(f"Test Macro-F1={100.0 * test_macro_f1_mean:.2f} +/- {100.0 * test_macro_f1_std:.2f} | Balanced Acc={100.0 * test_balanced_accuracy_mean:.2f} +/- {100.0 * test_balanced_accuracy_std:.2f} | {recall_name}={100.0 * test_recall_mean:.2f} +/- {100.0 * test_recall_std:.2f} | {precision_name}={100.0 * test_precision_mean:.2f} +/- {100.0 * test_precision_std:.2f}")
    print("=" * 110)

    return {
        "num_runs": len(runs),
        "seeds": seeds,
        "runs": runs,
        "temporal_test": False,
        "test_names": (),
        "val_macro_f1": val_macro_f1_mean,
        "val_macro_f1_std": val_macro_f1_std,
        "val_balanced_accuracy": val_balanced_accuracy_mean,
        "val_balanced_accuracy_std": val_balanced_accuracy_std,
        "val_recall": val_recall_mean,
        "val_recall_std": val_recall_std,
        "val_precision": val_precision_mean,
        "val_precision_std": val_precision_std,
        "test_macro_f1": test_macro_f1_mean,
        "test_macro_f1_std": test_macro_f1_std,
        "test_balanced_accuracy": test_balanced_accuracy_mean,
        "test_balanced_accuracy_std": test_balanced_accuracy_std,
        "test_recall": test_recall_mean,
        "test_recall_std": test_recall_std,
        "test_precision": test_precision_mean,
        "test_precision_std": test_precision_std,
        "average_test_macro_f1": test_macro_f1_mean,
        "average_test_macro_f1_std": test_macro_f1_std,
        "average_test_balanced_accuracy": test_balanced_accuracy_mean,
        "average_test_balanced_accuracy_std": test_balanced_accuracy_std,
        "average_test_recall": test_recall_mean,
        "average_test_recall_std": test_recall_std,
        "average_test_precision": test_precision_mean,
        "average_test_precision_std": test_precision_std,
        "test_metrics_by_environment": None,
    }


def _finetune_graph_classifier_once(pretrained_encoder, train_loader, val_loader, test_loader, device, num_classes, epochs, lr, weight_decay, seed):
    model = GraphClassifier(copy.deepcopy(pretrained_encoder), num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_epoch = 0
    best_val_macro_f1 = float("-inf")
    best_val_balanced_accuracy = float("-inf")
    best_val_loss = float("inf")
    best_val_metrics = None
    best_state = None

    for epoch in range(1, epochs + 1):
        train_classifier_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate_classifier(model, val_loader, device)
        current_score = (float(val_metrics["macro_f1"]), float(val_metrics["balanced_accuracy"]), -float(val_metrics["loss"]))
        best_score = (best_val_macro_f1, best_val_balanced_accuracy, -best_val_loss)

        if current_score > best_score:
            best_epoch = epoch
            best_val_macro_f1 = float(val_metrics["macro_f1"])
            best_val_balanced_accuracy = float(val_metrics["balanced_accuracy"])
            best_val_loss = float(val_metrics["loss"])
            best_val_metrics = dict(val_metrics)
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None or best_val_metrics is None:
        raise RuntimeError(f'Fine-tuning with seed {seed} did not produce a valid checkpoint.')

    model.load_state_dict(best_state)

    if isinstance(test_loader, Mapping):
        test_metrics_by_environment = {}
        for name, loader in test_loader.items():
            test_metrics_by_environment[str(name)] = evaluate_classifier(model, loader, device)

        average_test_macro_f1 = float(sum(metrics["macro_f1"] for metrics in test_metrics_by_environment.values()) / len(test_metrics_by_environment))
        average_test_balanced_accuracy = float(sum(metrics["balanced_accuracy"] for metrics in test_metrics_by_environment.values()) / len(test_metrics_by_environment))
        average_test_recall = float(sum(metrics["recall"] for metrics in test_metrics_by_environment.values()) / len(test_metrics_by_environment))
        average_test_precision = float(sum(metrics["precision"] for metrics in test_metrics_by_environment.values()) / len(test_metrics_by_environment))

        return {
            "seed": seed,
            "epoch": best_epoch,
            "val_macro_f1": float(best_val_metrics["macro_f1"]),
            "val_balanced_accuracy": float(best_val_metrics["balanced_accuracy"]),
            "val_recall": float(best_val_metrics["recall"]),
            "val_precision": float(best_val_metrics["precision"]),
            "test_macro_f1": average_test_macro_f1,
            "test_balanced_accuracy": average_test_balanced_accuracy,
            "test_recall": average_test_recall,
            "test_precision": average_test_precision,
            "average_test_macro_f1": average_test_macro_f1,
            "average_test_balanced_accuracy": average_test_balanced_accuracy,
            "average_test_recall": average_test_recall,
            "average_test_precision": average_test_precision,
            "test_metrics_by_environment": test_metrics_by_environment,
        }

    test_metrics = evaluate_classifier(model, test_loader, device)

    return {
        "seed": seed,
        "epoch": best_epoch,
        "val_macro_f1": float(best_val_metrics["macro_f1"]),
        "val_balanced_accuracy": float(best_val_metrics["balanced_accuracy"]),
        "val_recall": float(best_val_metrics["recall"]),
        "val_precision": float(best_val_metrics["precision"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_balanced_accuracy": float(test_metrics["balanced_accuracy"]),
        "test_recall": float(test_metrics["recall"]),
        "test_precision": float(test_metrics["precision"]),
        "average_test_macro_f1": float(test_metrics["macro_f1"]),
        "average_test_balanced_accuracy": float(test_metrics["balanced_accuracy"]),
        "average_test_recall": float(test_metrics["recall"]),
        "average_test_precision": float(test_metrics["precision"]),
        "test_metrics_by_environment": None,
    }


def _set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_loader(loader, seed):
    generator = getattr(loader, "generator", None)
    if generator is not None:
        generator.manual_seed(seed)

    sampler = getattr(loader, "sampler", None)
    sampler_generator = getattr(sampler, "generator", None)
    if sampler_generator is not None and sampler_generator is not generator:
        sampler_generator.manual_seed(seed)

    batch_sampler = getattr(loader, "batch_sampler", None)
    batch_sampler_generator = getattr(batch_sampler, "generator", None)
    if batch_sampler_generator is not None and batch_sampler_generator is not generator and (batch_sampler_generator is not sampler_generator):
        batch_sampler_generator.manual_seed(seed)


def _mean_and_std(values):
    values = torch.as_tensor(values, dtype=torch.float64)
    if values.numel() == 0:
        raise ValueError('Cannot summarize an empty metric sequence.')
    return values.mean().item(), values.std(unbiased=True).item() if values.numel() > 1 else 0.0


def _prepare_labels(labels, num_classes, device):
    if labels is None:
        raise ValueError('A downstream graph batch does not contain labels.')

    labels = torch.as_tensor(labels, device=device).view(-1)

    if labels.dtype.is_floating_point:
        rounded = labels.round()
        if not torch.equal(labels, rounded):
            raise ValueError('Downstream classification labels must be integers.')
        labels = rounded.long()
    else:
        labels = labels.long()

    if labels.numel() == 0:
        raise ValueError('Received an empty label tensor.')

    minimum = int(labels.min().item())
    maximum = int(labels.max().item())
    if minimum < 0 or maximum >= int(num_classes):
        raise ValueError(f'Labels must be in [0, {int(num_classes) - 1}], observed [{minimum}, {maximum}].')

    return labels
