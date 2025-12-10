"""
DeBERTa-v3-Large based Multimodal Sentiment Analysis for CMU-MOSEI
7-class sentiment classification with cross-modal attention fusion

Best Results: 56.17% Mult_acc_7 on CMU-MOSEI test set
"""

import os
os.environ['USE_TF'] = '0'
os.environ['TRANSFORMERS_NO_TF'] = '1'

import argparse
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class MOSEIDataset(Dataset):
    """Dataset with raw text for DeBERTa encoding"""

    def __init__(self, data, tokenizer, max_length=128):
        self.raw_text = data['raw_text']
        self.audio = torch.tensor(data['audio'], dtype=torch.float32)
        self.video = torch.tensor(data['vision'], dtype=torch.float32)
        self.labels = torch.tensor(data['regression_labels'], dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = str(self.raw_text[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'audio': self.audio[idx],
            'video': self.video[idx],
            'label': self.labels[idx]
        }


class DeBERTaMultimodalModel(nn.Module):
    """
    DeBERTa-v3-Large + Audio/Video Cross-Modal Attention Fusion

    Architecture:
    - Text: DeBERTa-v3-Large encoder (1024-dim)
    - Audio: Temporal Transformer encoder (74-dim -> hidden_size)
    - Video: Temporal Transformer encoder (35-dim -> hidden_size)
    - Cross-modal attention between modalities
    - Multi-task learning with auxiliary classifiers
    """

    def __init__(
        self,
        model_name='microsoft/deberta-v3-large',
        audio_dim=74,
        video_dim=35,
        hidden_size=512,
        num_heads=8,
        num_classes=7,
        dropout=0.2,
        freeze_deberta_layers=20
    ):
        super().__init__()

        # DeBERTa encoder
        self.deberta = AutoModel.from_pretrained(model_name)
        self.text_dim = self.deberta.config.hidden_size  # 1024 for large

        # Freeze some layers for efficiency
        if freeze_deberta_layers > 0:
            for param in self.deberta.embeddings.parameters():
                param.requires_grad = False
            for i, layer in enumerate(self.deberta.encoder.layer):
                if i < freeze_deberta_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        # Audio encoder (temporal)
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.audio_temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=2
        )

        # Video encoder (temporal)
        self.video_encoder = nn.Sequential(
            nn.Linear(video_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.video_temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=2
        )

        # Project text to hidden_size
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Cross-modal attention
        self.text_to_audio_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.text_to_video_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.audio_to_text_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.video_to_text_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 6, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Classifiers
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Auxiliary classifiers for multi-task learning
        self.text_classifier = nn.Linear(hidden_size, num_classes)
        self.audio_classifier = nn.Linear(hidden_size, num_classes)
        self.video_classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, audio, video):
        # Text encoding with DeBERTa
        text_output = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        text_hidden = text_output.last_hidden_state  # (B, seq_len, 1024)
        text_cls = text_hidden[:, 0]  # CLS token

        # Project text
        text_proj = self.text_proj(text_hidden)  # (B, seq_len, hidden)
        text_cls_proj = text_proj[:, 0]  # (B, hidden)

        # Audio encoding
        audio_hidden = self.audio_encoder(audio)  # (B, 500, hidden)
        audio_hidden = self.audio_temporal(audio_hidden)
        audio_pooled = audio_hidden.mean(dim=1)  # (B, hidden)

        # Video encoding
        video_hidden = self.video_encoder(video)  # (B, 500, hidden)
        video_hidden = self.video_temporal(video_hidden)
        video_pooled = video_hidden.mean(dim=1)  # (B, hidden)

        # Cross-modal attention
        text_to_audio, _ = self.text_to_audio_attn(
            text_proj, audio_hidden, audio_hidden
        )
        text_to_video, _ = self.text_to_video_attn(
            text_proj, video_hidden, video_hidden
        )
        text_to_audio_pooled = text_to_audio[:, 0]
        text_to_video_pooled = text_to_video[:, 0]

        # Audio/Video attend to text
        audio_to_text, _ = self.audio_to_text_attn(
            audio_hidden, text_proj, text_proj,
            key_padding_mask=(attention_mask == 0)
        )
        video_to_text, _ = self.video_to_text_attn(
            video_hidden, text_proj, text_proj,
            key_padding_mask=(attention_mask == 0)
        )

        # Multimodal representation
        multimodal = (audio_to_text.mean(dim=1) + video_to_text.mean(dim=1)) / 2

        # Fusion
        fused = torch.cat([
            text_cls_proj,
            audio_pooled,
            video_pooled,
            text_to_audio_pooled,
            text_to_video_pooled,
            multimodal
        ], dim=-1)

        fused = self.fusion(fused)

        # Classification
        logits = self.classifier(fused)
        text_logits = self.text_classifier(text_cls_proj)
        audio_logits = self.audio_classifier(audio_pooled)
        video_logits = self.video_classifier(video_pooled)

        return logits, text_logits, audio_logits, video_logits


def regression_to_class(pred, num_classes=7):
    """Convert regression prediction to class (0-6)"""
    pred = torch.clamp(pred, -3, 3)
    return torch.round((pred + 3)).long().clamp(0, num_classes - 1)


def compute_metrics(preds, labels, num_classes=7):
    """Compute evaluation metrics"""
    preds = preds.cpu().numpy() if torch.is_tensor(preds) else preds
    labels = labels.cpu().numpy() if torch.is_tensor(labels) else labels

    # Binary accuracy (positive/negative)
    has0_pred = (preds >= 0).astype(int)
    has0_label = (labels >= 0).astype(int)
    has0_acc = (has0_pred == has0_label).mean()
    has0_f1 = f1_score(has0_label, has0_pred, average='weighted')

    # Non-zero binary
    non0_mask = labels != 0
    if non0_mask.sum() > 0:
        non0_pred = (preds[non0_mask] > 0).astype(int)
        non0_label = (labels[non0_mask] > 0).astype(int)
        non0_acc = (non0_pred == non0_label).mean()
        non0_f1 = f1_score(non0_label, non0_pred, average='weighted')
    else:
        non0_acc = 0.0
        non0_f1 = 0.0

    # Multi-class accuracy (5 classes)
    pred_5 = np.clip(np.round(preds + 2), 0, 4).astype(int)
    label_5 = np.clip(np.round(labels + 2), 0, 4).astype(int)
    mult_acc_5 = (pred_5 == label_5).mean()

    # Multi-class accuracy (7 classes)
    pred_7 = np.clip(np.round(preds + 3), 0, 6).astype(int)
    label_7 = np.clip(np.round(labels + 3), 0, 6).astype(int)
    mult_acc_7 = (pred_7 == label_7).mean()

    # MAE and Correlation
    mae = np.abs(preds - labels).mean()
    corr = np.corrcoef(preds, labels)[0, 1] if len(preds) > 1 else 0.0

    return {
        'Has0_acc_2': has0_acc,
        'Has0_F1_score': has0_f1,
        'Non0_acc_2': non0_acc,
        'Non0_F1_score': non0_f1,
        'Mult_acc_5': mult_acc_5,
        'Mult_acc_7': mult_acc_7,
        'MAE': mae,
        'Corr': corr
    }


def train_epoch(model, loader, optimizer, scheduler, device,
                cls_weight=0.7, aux_weight=0.1, mixup_prob=0.5, mixup_alpha=0.4):
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        audio = batch['audio'].to(device)
        video = batch['video'].to(device)
        labels = batch['label'].to(device)

        class_labels = regression_to_class(labels)

        # Mixup augmentation
        if random.random() < mixup_prob:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            idx = torch.randperm(input_ids.size(0))

            # Mixup audio and video (text not mixed)
            audio = lam * audio + (1 - lam) * audio[idx]
            video = lam * video + (1 - lam) * video[idx]

            logits, text_logits, audio_logits, video_logits = model(
                input_ids, attention_mask, audio, video
            )

            loss_main = lam * F.cross_entropy(logits, class_labels) + \
                       (1 - lam) * F.cross_entropy(logits, class_labels[idx])
            loss_text = F.cross_entropy(text_logits, class_labels)
            loss_audio = lam * F.cross_entropy(audio_logits, class_labels) + \
                        (1 - lam) * F.cross_entropy(audio_logits, class_labels[idx])
            loss_video = lam * F.cross_entropy(video_logits, class_labels) + \
                        (1 - lam) * F.cross_entropy(video_logits, class_labels[idx])
        else:
            logits, text_logits, audio_logits, video_logits = model(
                input_ids, attention_mask, audio, video
            )

            loss_main = F.cross_entropy(logits, class_labels)
            loss_text = F.cross_entropy(text_logits, class_labels)
            loss_audio = F.cross_entropy(audio_logits, class_labels)
            loss_video = F.cross_entropy(video_logits, class_labels)

        # Total loss with multi-task weighting
        loss = cls_weight * loss_main + \
               aux_weight * (loss_text + loss_audio + loss_video)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    for batch in tqdm(loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        audio = batch['audio'].to(device)
        video = batch['video'].to(device)
        labels = batch['label'].to(device)

        logits, _, _, _ = model(input_ids, attention_mask, audio, video)

        # Convert logits to regression predictions
        probs = F.softmax(logits, dim=-1)
        class_preds = torch.argmax(probs, dim=-1)
        reg_preds = class_preds.float() - 3  # Map [0,6] back to [-3,3]

        class_labels = regression_to_class(labels)
        loss = F.cross_entropy(logits, class_labels)
        total_loss += loss.item()

        all_preds.append(reg_preds.cpu())
        all_labels.append(labels.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    metrics = compute_metrics(preds, labels)
    metrics['loss'] = total_loss / len(loader)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='DeBERTa Multimodal Sentiment Analysis')
    parser.add_argument('--pkl_path', type=str, required=True,
                        help='Path to CMU-MOSEI pickle file')
    parser.add_argument('--model_name', type=str, default='microsoft/deberta-v3-large')
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--freeze_layers', type=int, default=20,
                        help='Number of DeBERTa layers to freeze')
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--deberta_lr', type=float, default=5e-6)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--early_stop', type=int, default=15)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--mixup_prob', type=float, default=0.5)
    parser.add_argument('--mixup_alpha', type=float, default=0.4)
    parser.add_argument('--cls_weight', type=float, default=0.7)
    parser.add_argument('--aux_weight', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print(f"Loading data from {args.pkl_path}")
    with open(args.pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Create datasets
    train_dataset = MOSEIDataset(data['train'], tokenizer, args.max_length)
    valid_dataset = MOSEIDataset(data['valid'], tokenizer, args.max_length)
    test_dataset = MOSEIDataset(data['test'], tokenizer, args.max_length)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=4, pin_memory=True
    )

    print(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")

    # Create model
    print(f"Creating model with hidden_size={args.hidden_size}")
    model = DeBERTaMultimodalModel(
        model_name=args.model_name,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        dropout=args.dropout,
        freeze_deberta_layers=args.freeze_layers
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer with different learning rates
    deberta_params = list(model.deberta.parameters())
    other_params = [p for n, p in model.named_parameters() if 'deberta' not in n]

    optimizer = torch.optim.AdamW([
        {'params': [p for p in deberta_params if p.requires_grad], 'lr': args.deberta_lr},
        {'params': other_params, 'lr': args.lr}
    ], weight_decay=0.01)

    # Scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    # Training
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    best_acc = 0
    patience = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            cls_weight=args.cls_weight,
            aux_weight=args.aux_weight,
            mixup_prob=args.mixup_prob,
            mixup_alpha=args.mixup_alpha
        )
        print(f"Train Loss: {train_loss:.4f}")

        # Validation
        valid_metrics = evaluate(model, valid_loader, device)
        print(f"Valid Loss: {valid_metrics['loss']:.4f}")
        print(f"Mult_acc_7: {valid_metrics['Mult_acc_7']:.4f} | "
              f"Mult_acc_5: {valid_metrics['Mult_acc_5']:.4f} | "
              f"Has0_acc: {valid_metrics['Has0_acc_2']:.4f}")
        print(f"MAE: {valid_metrics['MAE']:.4f} | Corr: {valid_metrics['Corr']:.4f}")

        # Save best model
        if valid_metrics['Mult_acc_7'] > best_acc:
            best_acc = valid_metrics['Mult_acc_7']
            patience = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'args': args
            }, os.path.join(args.checkpoint_dir, 'best_model.pt'))
            print(f"*** New best model saved! Mult_acc_7: {best_acc:.4f} ***")
        else:
            patience += 1
            if patience >= args.early_stop:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Load best model and evaluate on test
    print("\nLoaded best model for final evaluation")
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])

    print("\n" + "=" * 60)
    print("Final Test Evaluation")
    print("=" * 60)

    test_metrics = evaluate(model, test_loader, device)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print("\nTest Metrics:")
    print("-" * 40)
    for k, v in test_metrics.items():
        if k != 'loss':
            print(f"  {k}: {v:.4f}")
    print("-" * 40)
    print(f"\n*** Final Mult_acc_7: {test_metrics['Mult_acc_7']:.4f} ***")


if __name__ == '__main__':
    main()
