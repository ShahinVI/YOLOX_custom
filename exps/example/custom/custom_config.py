#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Custom YOLOX Experiment Configuration
# This file provides complete control over all training parameters

import os
from yolox.exp import Exp as MyExp

# ========================================================================
# MODEL CONFIGURATION
# ========================================================================
# Name of the model architecture
# Options: "yolox_nano", "yolox_tiny", "yolox_s", "yolox_m", "yolox_l", "yolox_x"
MODEL_NAME = "yolox_s"  # Change to your model name if needed

# =======================================================================
# Activation function used in the model
# explanation for each option:
# "silu" (default): Smooth and efficient, works well in most cases
# "relu": Simple and fast, but can cause dead neurons
# "lrelu": Leaky ReLU, helps with dead neurons but can be less smooth
# "swish": Smooth and non-monotonic, can improve performance in some cases
ACTIVATION_FUNCTION = "silu"  # Change to "silu", "relu", "lrelu", or "swish" if needed

# ========================================================================
# Dataset Configuration
# ========================================================================
# Path to your dataset directory
DATASET_DIR = "datasets/COCO"  # Change to your dataset path
# Annotation files in COCO format
TRAIN_ANN = "instances_train2017.json"  # Training annotations file
VAL_ANN = "instances_val2017.json"      # Validation annotations file
TEST_ANN = "instances_test2017.json"    # Test annotations file (optional)

# classes and number of classes
CLASSES = ("cat", "dog")  # Change to your dataset classes
NUM_CLASSES = len(CLASSES)  # Number of classes in your dataset

# ========================================================================
# Model training parameters
# ========================================================================
MULTISCALE_RANGE = 0 # Range for multi-scale training (0 to disable)
NUMBER_OF_WORKERS = 4  # Number of data loading workers, explained below
# Explanation: More workers = faster data loading, but higher RAM usage
# Recommended: 4-8 for most systems, 0 for debugging (single-threaded)
MAX_EPOCH = 200  # Total number of training epochs
WARMUP_EPOCHS = 5  # Number of warmup epochs for learning rate
# Learning rate per image (base LR for batch size of 64)
BASIC_LR_PER_IMG = 0.01 / 64.0  # Base learning rate per image
WARMUP_LR = 0  # Learning rate during warmup (0 = linear warmup)
MIN_LR_RATIO = 0.05  # Minimum learning rate as a fraction of max LR
# Scheduler type for learning rate decay
SCHEDULER = "yoloxwarmcos"  # Options: "yoloxwarmcos", "step", "exp", explanation below
# Explanation: "yoloxwarmcos" is recommended for most cases, it combines warmup and cosine decay
# "step" is a simple step decay, while "exp" is exponential decay.
# "exp" is less commonly used but can be effective in some cases. 
# like "yoloxwarmcos", it combines warmup and exponential decay.
WEIGHT_DECAY = 5e-4  # L2 regularization strength, explained below
# Explanation: Helps prevent overfitting by penalizing large weights
MOMENTUM = 0.9  # Momentum for SGD optimizer, explained below
# Explanation: Helps accelerate SGD in the relevant direction and dampens oscillations
EMA = True  # Enable Exponential Moving Average of model weights, explained below
# Explanation: EMA helps stabilize training and often improves final performance
# ========================================================================

# ========================================================================
# Data augmentation parameters
# ========================================================================
# Enable or disable data augmentation
AUGMENTATION = True  # Set to False to disable all augmentations
if AUGMENTATION:
    # Mosaic augmentation probability (combines 4 images into 1)
    MOSAIC_PROB = 0.5  # Probability of applying mosaic augmentation, to disable set to 0.0
    # Mixup augmentation probability (blends two images)
    MIXUP_PROB = 0.5  # Probability of applying mixup augmentation, to disable set to 0.0
    # Enable or disable mixup entirely
    ENABLE_MIXUP = True  # Set to False to disable mixup augmentation
    # HSV augmentation probability (color space changes)
    HSV_PROB = 0.5  # Probability of applying HSV augmentation, to disable set to 0.0
    # Horizontal flip probability
    HFLIP_PROB = 0.5  # Probability of applying horizontal flip augmentation, to disable set to 0.0
    # Geometric augmentation ranges
    DEGREES = 10.0  # Rotation range in degrees (±10°)
    TRANSLATE = 0.1  # Translation range as fraction of image size (±10%)
    SHEAR = 2.0  # Shear range in degrees (±2°)
else:
    # If augmentation is disabled, set all probabilities to 0
    MOSAIC_PROB = 0.0
    MIXUP_PROB = 0.0
    ENABLE_MIXUP = False
    HSV_PROB = 0.0
    HFLIP_PROB = 0.0
    DEGREES = 0.0
    TRANSLATE = 0.0
    SHEAR = 0.0
# Mosaic scaling range (how much to scale images in mosaic)
MOSAIC_SCALE = (0.5, 1.5)  # Scale range for mosaic augmentation (50% to 150%)
# Mixup scaling range
MIXUP_SCALE = (0.5, 1.5)  # Scale range for mixup augmentation (50% to 150%)
# Number of epochs to disable augmentation at the end
NO_AUG_EPOCHS = 15  # Number of epochs to disable augmentation at the end
# ========================================================================

# Model parameters for different YOLOX variants
MODEL_PARAMS = {"yolox_nano": (0.33, 0.25,(416, 416)), 
                "yolox_tiny": (0.33, 0.375,(416, 416)), 
                "yolox_s": (0.33, 0.50,(640, 640)), 
                "yolox_m": (0.67, 0.75,(640, 640)), 
                "yolox_l": (1.00, 1.00,(640, 640)), 
                "yolox_x": (1.33, 1.25,(640, 640))}

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        
        # ========================================================================
        # EXPERIMENT IDENTIFICATION
        # ========================================================================
        # Name of this experiment - used for logging and saving checkpoints
        self.exp_name = "my_custom_yolox_experiment"
        
        # ========================================================================
        # MODEL ARCHITECTURE CONFIGURATION
        # ========================================================================
        # Number of classes in your dataset (MOST IMPORTANT SETTING)
        # COCO has 80 classes, change this to match your dataset
        self.num_classes = NUM_CLASSES  # Example: cat and dog detection
        # class names for logging and visualization
        self.class_names = CLASSES  # List of class names, e.g., ["cat", "dog"]
        
        # Model size factors (determines model capacity)
        # YOLOX-Nano: depth=0.33, width=0.25
        # YOLOX-Tiny: depth=0.33, width=0.375  
        # YOLOX-S:    depth=0.33, width=0.50
        # YOLOX-M:    depth=0.67, width=0.75
        # YOLOX-L:    depth=1.00, width=1.00
        # YOLOX-X:    depth=1.33, width=1.25
        try:
            self.depth, self.width, _ = MODEL_PARAMS[MODEL_NAME]
        except Exception as e:
            print(f"Error setting model parameters for {MODEL_NAME}: {e}")
            return 0

        # Activation function used throughout the model
        # Options: "silu" (default), "relu", "lrelu", "swish"
        if ACTIVATION_FUNCTION not in ["silu", "relu", "lrelu", "swish"]:
            raise ValueError(f"Invalid activation function: {ACTIVATION_FUNCTION}. "
                             "Choose from 'silu', 'relu', 'lrelu', 'swish'.")
        self.act = ACTIVATION_FUNCTION
        
        # ========================================================================
        # DATASET CONFIGURATION
        # ========================================================================
        # Root directory containing your dataset
        # Structure should be: data_dir/images/ and data_dir/annotations/
        self.data_dir = DATASET_DIR
        
        # Annotation file names (must be in COCO JSON format)
        self.train_ann = TRAIN_ANN   # Training annotations
        self.val_ann = VAL_ANN       # Validation annotations
        self.test_ann = TEST_ANN     # Test annotations (optional)
        
        # ========================================================================
        # INPUT/OUTPUT CONFIGURATION
        # ========================================================================
        # Input image size for training (height, width)
        # Must be multiples of 32. Common sizes: (416,416), (640,640), (832,832)
        # Larger = better accuracy but slower training
        self.input_size = MODEL_PARAMS[MODEL_NAME][2]

        # Output image size for evaluation/testing
        # Usually same as input_size, but can be different for speed/accuracy trade-off
        self.test_size = MODEL_PARAMS[MODEL_NAME][2]  # Use same size for testing
        
        # Multi-scale training range (0 to disable)
        # Randomly varies input size by ±multiscale_range*32 pixels during training
        # Helps model generalize to different image sizes
        self.multiscale_range = MULTISCALE_RANGE  # Will train on sizes from 480x480 to 800x800
        
        # Alternative: manually specify size range (uncomment to use)
        # self.random_size = (14, 26)  # Size range in multiples of 32
        
        # ========================================================================
        # DATA LOADING CONFIGURATION
        # ========================================================================
        # Number of parallel workers for data loading
        # Higher = faster data loading but more RAM usage
        # Set to 0 for single-threaded loading (useful for debugging)
        self.data_num_workers = NUMBER_OF_WORKERS
        
        # ========================================================================
        # TRAINING HYPERPARAMETERS
        # ========================================================================
        # Total number of training epochs
        self.max_epoch = MAX_EPOCH
        
        # Warmup epochs - gradual learning rate increase at start
        # Helps stabilize training, especially with large batch sizes
        self.warmup_epochs = WARMUP_EPOCHS

        # Learning rate configuration
        self.basic_lr_per_img = BASIC_LR_PER_IMG  # Base LR per image (will be scaled by batch size)
        self.warmup_lr = WARMUP_LR                # Learning rate during warmup (0 = linear warmup)
        self.min_lr_ratio = MIN_LR_RATIO          # Minimum LR as fraction of max LR

        # Learning rate scheduler type
        # Options: "yoloxwarmcos", "step", "exp"
        self.scheduler = SCHEDULER  # Recommended: "yoloxwarmcos" for most cases

        # Optimizer settings
        self.weight_decay = WEIGHT_DECAY  # L2 regularization strength
        self.momentum = MOMENTUM       # SGD momentum

        # Exponential Moving Average (EMA) of model weights
        # Helps stabilize training and often improves final performance
        self.ema = EMA
        
        # ========================================================================
        # DATA AUGMENTATION CONFIGURATION
        # ========================================================================
        # Mosaic augmentation probability (combines 4 images into 1)
        # Very effective for object detection, but can be disabled for fine-tuning
        self.mosaic_prob = MOSAIC_PROB  # Set to 0.0 to disable mosaic augmentation
        
        # Mixup augmentation probability (blends two images)
        # Works well with mosaic, helps with generalization
        self.mixup_prob = MIXUP_PROB
        
        # Enable/disable mixup entirely
        self.enable_mixup = ENABLE_MIXUP
        
        # HSV augmentation probability (color space changes)
        # Helps model become invariant to lighting conditions
        self.hsv_prob = HSV_PROB
        
        # Horizontal flip probability
        # Set to 0.0 if your objects have orientation (e.g., text detection)
        self.flip_prob = HFLIP_PROB
        
        # Geometric augmentation ranges
        self.degrees = DEGREES      # Rotation range in degrees (±10°)
        self.translate = TRANSLATE     # Translation range as fraction of image size (±10%)
        self.shear = SHEAR         # Shear range in degrees (±2°)
        
        # Mosaic scaling range (how much to scale images in mosaic)
        self.mosaic_scale = MOSAIC_SCALE  # 10% to 200% of original size
        
        # Mixup scaling range
        self.mixup_scale = MIXUP_SCALE  # 50% to 150% of original size

        # Number of epochs to disable augmentation at the end
        # Helps model converge better on clean images
        self.no_aug_epochs = NO_AUG_EPOCHS
        
        # ========================================================================
        # EVALUATION/TESTING CONFIGURATION  
        # ========================================================================
        # Confidence threshold for evaluation
        # Predictions below this confidence are filtered out
        self.test_conf = 0.01  # Low threshold to catch all potential detections
        
        # Non-Maximum Suppression (NMS) threshold
        # Higher values = more boxes kept, lower values = more aggressive filtering
        self.nmsthre = 0.65
        
        # ========================================================================
        # LOGGING AND SAVING CONFIGURATION
        # ========================================================================
        # How often to print training logs (every N iterations)
        self.print_interval = 10
        
        # How often to run evaluation (every N epochs)
        # Set to 1 for evaluation every epoch (slower but more monitoring)
        self.eval_interval = 1
        
        # Whether to save checkpoint history
        # True = save all epoch checkpoints, False = only save best and latest
        self.save_history_ckpt = False  # Set to False to save disk space
        
        # ========================================================================
        # ADVANCED CONFIGURATION (Usually don't need to change)
        # ========================================================================
        # Random seed for reproducibility (None = random seed each run)
        self.seed = 42
        
        # Enable automatic mixed precision training (saves memory, faster on modern GPUs)
        self.enable_amp = True
        
        # Gradient clipping value (prevents exploding gradients)
        self.grad_clip = None  # None = no clipping, or set to value like 10.0
        
    # ========================================================================
    # CUSTOM METHODS (Override parent methods for advanced customization)
    # ========================================================================
    
    def get_model(self):
        """
        Override this method to customize model architecture
        """
        # Use the default YOLOX model
        model = super().get_model()
        
        # Example customizations:
        # - Modify backbone (e.g., use different ResNet variant)
        # - Change head configuration
        # - Add custom layers
        
        return model
    
    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        """
        Override this method to use custom dataset class
        Args:
            cache: Whether to cache images in memory (faster training)
            cache_type: "ram" or "disk" caching
        """
        # Use default COCO dataset format
        dataset = super().get_dataset(cache, cache_type)
        
        # Example customizations:
        # - Use custom dataset class
        # - Add custom preprocessing
        # - Filter certain classes
        
        return dataset
    
    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img: str = None):
        """
        Override this method to customize data loading
        Args:
            batch_size: Training batch size
            is_distributed: Whether using distributed training
            no_aug: Whether to disable augmentation
            cache_img: Image caching type ("ram", "disk", or None)
        """
        # Use default data loader
        dataloader = super().get_data_loader(batch_size, is_distributed, no_aug, cache_img)
        
        # Example customizations:
        # - Custom batch sampler
        # - Different augmentation pipeline
        # - Custom collate function
        
        return dataloader
    
    def get_optimizer(self, batch_size):
        """
        Override this method to use different optimizer
        """
        # Default uses SGD optimizer
        optimizer = super().get_optimizer(batch_size)
        
        # Example: Use Adam optimizer instead
        # import torch.optim as optim
        # lr = self.basic_lr_per_img * batch_size
        # optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.weight_decay)
        
        return optimizer
    
    def get_lr_scheduler(self, lr, iters_per_epoch):
        """
        Override this method to use custom learning rate schedule
        """
        # Use default cosine annealing with warmup
        scheduler = super().get_lr_scheduler(lr, iters_per_epoch)
        
        # Example: Use step decay instead
        # from torch.optim.lr_scheduler import StepLR
        # scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)
        
        return scheduler
    
    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        """
        Override this method to use custom evaluation metrics
        """
        # Use default COCO evaluator
        evaluator = super().get_evaluator(batch_size, is_distributed, testdev, legacy)
        
        # Example customizations:
        # - Use custom evaluation metrics
        # - Add additional evaluation datasets
        # - Modify evaluation frequency
        
        return evaluator

# ========================================================================
# EXAMPLE CONFIGURATIONS FOR DIFFERENT USE CASES
# ========================================================================

# example of configurations based on your needs:

# # Configuration for small dataset (< 1000 images)
# class SmallDatasetExp(Exp):
#     def __init__(self):
#         super().__init__()
#         self.max_epoch = 200
#         self.warmup_epochs = 10
#         self.basic_lr_per_img = 0.005 / 64.0  # Lower learning rate
#         self.mosaic_prob = 0.5                # Reduce augmentation
#         self.mixup_prob = 0.3
#         self.no_aug_epochs = 30              # More epochs without augmentation

# # Configuration for large dataset (> 100,000 images)
# class LargeDatasetExp(Exp):
#     def __init__(self):
#         super().__init__()
#         self.max_epoch = 50                  # Fewer epochs needed
#         self.eval_interval = 5               # Less frequent evaluation
#         self.data_num_workers = 8            # More workers for faster loading
#         self.save_history_ckpt = False       # Save disk space

# # Configuration for high accuracy (slower training)
# class HighAccuracyExp(Exp):
#     def __init__(self):
#         super().__init__()
#         self.input_size = (832, 832)         # Larger input size
#         self.test_size = (832, 832)
#         self.depth = 1.00                    # YOLOX-L model
#         self.width = 1.00
#         self.multiscale_range = 8            # More scale variation
#         self.no_aug_epochs = 30              # More fine-tuning epochs

# # Configuration for fast training (lower accuracy)
# class FastTrainingExp(Exp):
#     def __init__(self):
#         super().__init__()
#         self.input_size = (416, 416)         # Smaller input size
#         self.test_size = (416, 416)
#         self.depth = 0.33                    # YOLOX-S model
#         self.width = 0.375                   # Even smaller (YOLOX-Tiny)
#         self.mosaic_prob = 0.8               # Less augmentation
#         self.mixup_prob = 0.5
#         self.max_epoch = 50                  # Fewer epochs