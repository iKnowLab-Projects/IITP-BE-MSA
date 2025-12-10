# DeBERTa Multimodal Sentiment Analysis

DeBERTa-v3-Large based multimodal sentiment analysis model for CMU-MOSEI 7-class classification.

## Results

| Model | Mult_acc_7 | Mult_acc_5 | Has0_acc_2 |
|-------|-----------|-----------|-----------|
| DeBERTa + Multimodal Fusion | **56.17%** | 57.83% | 83.59% |

## Architecture

- **Text Encoder**: DeBERTa-v3-Large (1024-dim, 24 layers)
  - First 20 layers frozen for efficient fine-tuning
- **Audio Encoder**: 2-layer Transformer encoder (74-dim COVAREP features)
- **Video Encoder**: 2-layer Transformer encoder (35-dim OpenFace features)
- **Cross-Modal Attention**: Bidirectional attention between text-audio and text-video
- **Multi-task Learning**: Auxiliary classifiers for each modality

## Data

This model is trained on CMU-MOSEI dataset with:
- 16,326 training samples
- 1,871 validation samples
- 4,659 test samples

Data format: PKL file with `raw_text`, `audio`, `vision`, and `regression_labels` fields.

## Training

### Basic Training

```bash
python train.py \
    --pkl_path data/mosei/unaligned_50.pkl \
    --model_name microsoft/deberta-v3-large \
    --hidden_size 512 \
    --num_heads 8 \
    --freeze_layers 20 \
    --lr 3e-5 \
    --deberta_lr 5e-6 \
    --batch_size 8 \
    --epochs 50 \
    --early_stop 15 \
    --mixup_prob 0.5 \
    --mixup_alpha 0.4 \
    --checkpoint_dir ./checkpoints \
    --seed 42
```

### Best Hyperparameters

| Parameter | Value |
|-----------|-------|
| hidden_size | 512 |
| num_heads | 8 |
| freeze_layers | 20 |
| lr | 3e-5 |
| deberta_lr | 5e-6 |
| batch_size | 8 |
| max_length | 128 |
| mixup_prob | 0.5 |
| mixup_alpha | 0.4 |
| cls_weight | 0.7 |
| aux_weight | 0.1 |
| dropout | 0.2 |

## Evaluation

### Single Model Evaluation

```bash
python eval.py \
    --pkl_path data/mosei/unaligned_50.pkl \
    --checkpoint_dir ./checkpoints
```

### Multi-Seed Ensemble

Train multiple models with different seeds and ensemble:

```bash
# Train with different seeds
for seed in 42 123 456 789; do
    python train.py \
        --pkl_path data/mosei/unaligned_50.pkl \
        --checkpoint_dir ./checkpoints_seed${seed} \
        --seed ${seed}
done

# Ensemble evaluation
python eval.py \
    --pkl_path data/mosei/unaligned_50.pkl \
    --checkpoint_dirs ./checkpoints_seed42 ./checkpoints_seed123 ./checkpoints_seed456 ./checkpoints_seed789
```

## Key Features

1. **DeBERTa-v3-Large**: State-of-the-art language model with disentangled attention
2. **Cross-Modal Attention**: Learns interactions between text, audio, and video modalities
3. **Mixup Augmentation**: Applied to audio/video features for better generalization
4. **Multi-task Learning**: Auxiliary losses from each modality branch
5. **Layer Freezing**: Efficient fine-tuning by freezing early DeBERTa layers

## Requirements

```
torch>=2.0.0
transformers>=4.35.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.11.0
tqdm>=4.66.0
```

## Citation

If you use this code, please cite:

```bibtex
@misc{deberta-mosei-2024,
  title={DeBERTa Multimodal Sentiment Analysis for CMU-MOSEI},
  author={IITP Butterfly Effect Project},
  year={2024}
}
```

## License

MIT License
