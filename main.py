"""
Entry point for ChimeraAI implementation.

This script provides a command-line interface to run training, inference, and evaluation
for the ChimeraAI human image animation system with hybrid guidance.
"""

import os
import argparse
import torch
from typing import Dict, Any, Optional
import logging

from utils.config import load_config
from utils.logging import init_logger
from training.trainer import ChimeraAITrainer
from inference.inference import ChimeraAIInference
from evaluation.metrics import evaluate_animation


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="ChimeraAI Implementation")
    
    # Common arguments
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to configuration file")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on (cuda or cpu)")
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Mode to run")
    
    # Training subparser
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    train_parser.add_argument("--stage", type=int, default=0, help="Training stage to start from")
    train_parser.add_argument("--data_dir", type=str, help="Directory containing processed data")
    train_parser.add_argument("--batch_size", type=int, help="Override batch size in config")
    train_parser.add_argument("--epochs", type=int, help="Override number of epochs in config")
    train_parser.add_argument("--lr", type=float, help="Override learning rate in config")
    train_parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    train_parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    
    # Inference subparser
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    infer_parser.add_argument("--reference", type=str, required=True, help="Path to reference image")
    infer_parser.add_argument("--motion", type=str, required=True, help="Path to motion source (video or directory)")
    infer_parser.add_argument("--output", type=str, required=True, help="Path to save output animation")
    infer_parser.add_argument("--num_frames", type=int, default=73, help="Number of frames to generate")
    infer_parser.add_argument("--guidance_scale", type=float, default=2.5, help="Classifier-free guidance scale")
    infer_parser.add_argument("--steps", type=int, default=50, help="Number of diffusion steps")
    
    # Evaluation subparser
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument("--reference", type=str, required=True, help="Path to reference image")
    eval_parser.add_argument("--source", type=str, required=True, help="Path to source motion video")
    eval_parser.add_argument("--generated", type=str, required=True, help="Path to generated animation")
    eval_parser.add_argument("--output", type=str, help="Path to save evaluation results")
    
    # Data preprocessing subparser
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess data")
    preprocess_parser.add_argument("--raw_dir", type=str, help="Directory containing raw data")
    preprocess_parser.add_argument("--output_dir", type=str, help="Directory to save processed data")
    
    return parser.parse_args()


def train(args):
    """Run training."""
    print("Starting training...")
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.data_dir:
        config['data']['processed_data_dir'] = args.data_dir
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.fp16:
        config['training']['fp16'] = args.fp16
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(args.log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    config['training']['checkpoint_dir'] = checkpoint_dir
    
    # Initialize logger
    init_logger(
        name="ChimeraAI",
        log_dir=args.log_dir,
        use_tensorboard=True,
        use_wandb=args.wandb,
        wandb_project=config.get('logging', {}).get('project_name', "ChimeraAI"),
        wandb_config=config
    )
    
    # Initialize trainer
    trainer = ChimeraAITrainer(
        config_path=args.config,
        resume_from=args.resume,
        device=args.device
    )
    
    # Start training
    trainer.train()


def infer(args):
    """Run inference."""
    print("Starting inference...")
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize inference engine
    inference = ChimeraAIInference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # Generate animation
    output_path = inference.generate_animation(
        reference_image_path=args.reference,
        motion_source_path=args.motion,
        output_path=args.output,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        num_steps=args.steps
    )
    
    print(f"Animation saved to: {output_path}")


def eval(args):
    """Run evaluation."""
    print("Starting evaluation...")
    
    # Run evaluation
    metrics = evaluate_animation(
        reference_image_path=args.reference,
        source_video_path=args.source,
        generated_video_path=args.generated,
        output_path=args.output
    )
    
    # Print metrics
    print("\nEvaluation Results:")
    print("-" * 30)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print("-" * 30)


def preprocess(args):
    """Run data preprocessing."""
    print("Starting data preprocessing...")
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.raw_dir:
        config['data']['raw_data_dir'] = args.raw_dir
    if args.output_dir:
        config['data']['processed_data_dir'] = args.output_dir
    
    # Import preprocess function here to avoid circular imports
    from data.preprocess import preprocess_dataset
    
    # Run preprocessing
    preprocess_dataset(args.config)
    
    print("Data preprocessing completed")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check for existence of config file
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    
    # Run the appropriate mode
    if args.mode == "train":
        train(args)
    elif args.mode == "infer":
        infer(args)
    elif args.mode == "eval":
        eval(args)
    elif args.mode == "preprocess":
        preprocess(args)
    else:
        print("Please specify a mode: train, infer, eval, or preprocess")


if __name__ == "__main__":
    main()
