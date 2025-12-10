"""
Evaluation script for DeBERTa Multimodal Sentiment Analysis
Supports single model evaluation and multi-seed ensemble
"""

import os
os.environ['USE_TF'] = '0'
os.environ['TRANSFORMERS_NO_TF'] = '1'

import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from scipy.stats import mode

from train import DeBERTaMultimodalModel, MOSEIDataset, compute_metrics


def load_model(checkpoint_path, model_name, hidden_size, num_heads,
               audio_dim, video_dim, num_classes, device):
    """Load a trained DeBERTa model from checkpoint"""
    model = DeBERTaMultimodalModel(
        model_name=model_name,
        hidden_size=hidden_size,
        num_heads=num_heads,
        audio_dim=audio_dim,
        video_dim=video_dim,
        num_classes=num_classes,
        freeze_deberta_layers=20,
        dropout=0.2
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint


def get_predictions(model, dataloader, device):
    """Get logits predictions from a model"""
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio = batch['audio'].to(device)
            video = batch['video'].to(device)
            labels = batch['label']

            outputs = model(input_ids, attention_mask, audio, video)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)


def compute_ensemble_metrics(logits, labels):
    """Compute evaluation metrics for 7-class classification"""
    preds = np.argmax(logits, axis=1)

    # Mult_acc_7
    mult_acc_7 = accuracy_score(labels, preds)

    # Mult_acc_5 (merge extreme classes)
    def to_5_class(x):
        if x <= 1:
            return 0
        elif x >= 5:
            return 4
        else:
            return x - 1

    labels_5 = np.array([to_5_class(l) for l in labels])
    preds_5 = np.array([to_5_class(p) for p in preds])
    mult_acc_5 = accuracy_score(labels_5, preds_5)

    # Binary metrics (class 3 is neutral/0)
    labels_binary = (labels >= 3).astype(int)
    preds_binary = (preds >= 3).astype(int)

    has0_mask = labels == 3
    non0_mask = labels != 3

    has0_acc = accuracy_score(labels_binary[has0_mask], preds_binary[has0_mask]) if has0_mask.sum() > 0 else 0
    non0_acc = accuracy_score(labels_binary[non0_mask], preds_binary[non0_mask]) if non0_mask.sum() > 0 else 0

    return {
        'Mult_acc_7': mult_acc_7,
        'Mult_acc_5': mult_acc_5,
        'Has0_acc_2': has0_acc,
        'Non0_acc_2': non0_acc
    }


def single_model_eval(args):
    """Evaluate a single model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    with open(args.pkl_path, 'rb') as f:
        data = pickle.load(f)

    test_data = data['test']

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Create dataset
    test_dataset = MOSEIDataset(test_data, tokenizer, args.max_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Test samples: {len(test_dataset)}")

    # Load model
    checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
    model, checkpoint = load_model(
        checkpoint_path, args.model_name, args.hidden_size, args.num_heads,
        74, 35, 7, device  # audio_dim=74, video_dim=35
    )

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Get predictions
    logits, labels = get_predictions(model, test_loader, device)

    # Convert labels to class indices
    labels_class = np.clip(np.round(labels + 3), 0, 6).astype(int)

    # Compute metrics
    metrics = compute_ensemble_metrics(logits, labels_class)

    print("\n" + "=" * 50)
    print("Test Results:")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k}: {v*100:.2f}%")


def ensemble_eval(args):
    """Evaluate multi-seed ensemble"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    with open(args.pkl_path, 'rb') as f:
        data = pickle.load(f)

    test_data = data['test']

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Create dataset
    test_dataset = MOSEIDataset(test_data, tokenizer, args.max_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Test samples: {len(test_dataset)}")

    # Load models and get predictions
    print(f"\n{'='*60}")
    print(f"Loading {len(args.checkpoint_dirs)} models for ensemble")
    print(f"{'='*60}")

    all_model_logits = []
    model_info = []

    for ckpt_dir in args.checkpoint_dirs:
        ckpt_path = os.path.join(ckpt_dir, 'best_model.pt')
        if not os.path.exists(ckpt_path):
            print(f"Warning: {ckpt_path} not found, skipping...")
            continue

        print(f"\nLoading model from {ckpt_dir}...")
        model, checkpoint = load_model(
            ckpt_path, args.model_name, args.hidden_size, args.num_heads,
            74, 35, 7, device
        )

        epoch = checkpoint.get('epoch', 'unknown')
        valid_acc = checkpoint.get('best_acc', 0)

        print(f"  Epoch: {epoch}, Valid Mult_acc_7: {valid_acc:.4f}")
        model_info.append({
            'dir': ckpt_dir,
            'epoch': epoch,
            'valid_acc': valid_acc
        })

        # Get predictions
        logits, labels = get_predictions(model, test_loader, device)
        all_model_logits.append(logits)

        # Convert labels to class indices
        labels_class = np.clip(np.round(labels + 3), 0, 6).astype(int)

        # Individual model metrics
        individual_metrics = compute_ensemble_metrics(logits, labels_class)
        print(f"  Individual Mult_acc_7: {individual_metrics['Mult_acc_7']*100:.2f}%")

        del model
        torch.cuda.empty_cache()

    if len(all_model_logits) < 2:
        print("\nNeed at least 2 models for ensemble!")
        return

    # Stack logits
    stacked_logits = np.stack(all_model_logits, axis=0)

    print(f"\n{'='*60}")
    print("ENSEMBLE RESULTS")
    print(f"{'='*60}")

    results = {}

    # 1. Logits averaging
    ensemble_logits_avg = np.mean(stacked_logits, axis=0)
    metrics_avg = compute_ensemble_metrics(ensemble_logits_avg, labels_class)
    results['logits_avg'] = metrics_avg

    # 2. Probability averaging
    probs = np.exp(stacked_logits - np.max(stacked_logits, axis=-1, keepdims=True))
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    ensemble_probs_avg = np.mean(probs, axis=0)
    metrics_probs = compute_ensemble_metrics(ensemble_probs_avg, labels_class)
    results['probs_avg'] = metrics_probs

    # 3. Majority voting
    individual_preds = np.argmax(stacked_logits, axis=2)
    majority_votes = mode(individual_preds, axis=0, keepdims=False)[0]
    majority_acc_7 = accuracy_score(labels_class, majority_votes)
    results['votes'] = {'Mult_acc_7': majority_acc_7}

    # 4. Weighted averaging (by validation accuracy)
    valid_accs = np.array([m['valid_acc'] for m in model_info])
    weights = valid_accs / valid_accs.sum()
    weighted_logits = np.sum(stacked_logits * weights[:, None, None], axis=0)
    metrics_weighted = compute_ensemble_metrics(weighted_logits, labels_class)
    results['weighted_avg'] = metrics_weighted

    # Print results
    print(f"\nEnsemble Methods Comparison:")
    print(f"-" * 50)
    print(f"{'Method':<20} {'Mult_acc_7':>12} {'Mult_acc_5':>12}")
    print(f"-" * 50)

    for method, metrics in results.items():
        mult7 = metrics.get('Mult_acc_7', 0) * 100
        mult5 = metrics.get('Mult_acc_5', 0) * 100 if 'Mult_acc_5' in metrics else 0
        print(f"{method:<20} {mult7:>11.2f}% {mult5:>11.2f}%")

    # Best result
    best_method = max(results.keys(), key=lambda x: results[x].get('Mult_acc_7', 0))
    best_acc = results[best_method]['Mult_acc_7'] * 100

    print(f"\n{'='*60}")
    print(f"BEST ENSEMBLE: {best_method}")
    print(f"Test Mult_acc_7: {best_acc:.2f}%")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate DeBERTa Multimodal Model')
    parser.add_argument('--pkl_path', type=str, default='data/mosei/unaligned_50.pkl')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Single checkpoint directory for single model evaluation')
    parser.add_argument('--checkpoint_dirs', type=str, nargs='+', default=None,
                        help='Multiple checkpoint directories for ensemble evaluation')
    parser.add_argument('--model_name', type=str, default='microsoft/deberta-v3-large')
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    if args.checkpoint_dirs:
        ensemble_eval(args)
    elif args.checkpoint_dir:
        single_model_eval(args)
    else:
        print("Please provide --checkpoint_dir for single model or --checkpoint_dirs for ensemble")


if __name__ == '__main__':
    main()
