"""
Evaluation metrics for ChimeraAI.

This module implements various metrics for assessing the quality of generated animations,
including FID, PSNR, SSIM, and custom metrics for temporal coherence and expression accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import os

from torchvision.models import inception_v3, Inception_V3_Weights
from scipy import linalg


class FrechetInceptionDistance:
    """
    Fréchet Inception Distance (FID) implementation.
    
    FID is a measure of similarity between two datasets of images.
    It is calculated by computing the Fréchet distance between two
    Gaussians fitted to feature representations of the Inception network.
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize FID calculator.
        
        Args:
            device: Device to run the model on
        """
        self.device = device
        
        # Load Inception model
        self.inception_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        self.inception_model.fc = nn.Identity()  # Remove classification layer
        self.inception_model.eval()
        self.inception_model.to(device)
        
        # Register hook to get features
        self.features = None
        def hook_fn(module, input, output):
            self.features = output.detach()
        
        self.inception_model.avgpool.register_forward_hook(hook_fn)
    
    def _get_activations(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get Inception activations for a batch of images.
        
        Args:
            images: Batch of images [N, 3, 299, 299]
            
        Returns:
            Inception activations [N, 2048]
        """
        # Ensure images are the right size
        if images.shape[2] != 299 or images.shape[3] != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Forward pass
        with torch.no_grad():
            self.inception_model(images)
        
        # Get features from hook
        activations = self.features.squeeze()
        
        return activations
    
    def _calculate_statistics(self, activations: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate mean and covariance statistics.
        
        Args:
            activations: Inception activations
            
        Returns:
            Tuple of (mean, covariance)
        """
        activations = activations.cpu().numpy()
        
        # Calculate mean and covariance
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        
        return mu, sigma
    
    def _calculate_frechet_distance(
        self,
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray
    ) -> float:
        """
        Calculate Fréchet distance between two distributions.
        
        Args:
            mu1: Mean of first distribution
            sigma1: Covariance of first distribution
            mu2: Mean of second distribution
            sigma2: Covariance of second distribution
            
        Returns:
            Fréchet distance
        """
        # Calculate squared difference between means
        diff = mu1 - mu2
        
        # Product of covariances
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
        
        # Check for numerical errors
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
        
        # Ensure covmean is real
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        # Calculate FID
        tr_covmean = np.trace(covmean)
        
        return diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    
    def calculate_fid(
        self,
        real_images: torch.Tensor,
        generated_images: torch.Tensor
    ) -> float:
        """
        Calculate FID between real and generated images.
        
        Args:
            real_images: Real images [N, 3, H, W]
            generated_images: Generated images [N, 3, H, W]
            
        Returns:
            FID score (lower is better)
        """
        # Get activations
        real_activations = self._get_activations(real_images)
        gen_activations = self._get_activations(generated_images)
        
        # Calculate statistics
        real_mu, real_sigma = self._calculate_statistics(real_activations)
        gen_mu, gen_sigma = self._calculate_statistics(gen_activations)
        
        # Calculate FID
        fid = self._calculate_frechet_distance(real_mu, real_sigma, gen_mu, gen_sigma)
        
        return float(fid)


class PSNR:
    """Peak Signal-to-Noise Ratio for image quality assessment."""
    
    def __init__(self, max_value: float = 1.0):
        """
        Initialize PSNR calculator.
        
        Args:
            max_value: Maximum value of the signal
        """
        self.max_value = max_value
    
    def __call__(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Calculate PSNR between two images.
        
        Args:
            img1: First image tensor [3, H, W]
            img2: Second image tensor [3, H, W]
            
        Returns:
            PSNR value in dB (higher is better)
        """
        # Calculate MSE
        mse = F.mse_loss(img1, img2)
        
        # Avoid division by zero
        if mse == 0:
            return float('inf')
        
        # Calculate PSNR
        psnr = 20 * torch.log10(self.max_value / torch.sqrt(mse))
        
        return psnr.item()


class SSIM:
    """Structural Similarity Index for image quality assessment."""
    
    def __init__(
        self,
        window_size: int = 11,
        channel: int = 3,
        sigma: float = 1.5
    ):
        """
        Initialize SSIM calculator.
        
        Args:
            window_size: Size of the window for SSIM calculation
            channel: Number of channels in the images
            sigma: Standard deviation of the Gaussian window
        """
        self.window_size = window_size
        self.channel = channel
        self.sigma = sigma
        
        # Values for stabilization
        self.C1 = (0.01 * 1.0) ** 2
        self.C2 = (0.03 * 1.0) ** 2
    
    def _create_window(self, window_size: int) -> torch.Tensor:
        """Create a Gaussian window for SSIM."""
        # Create 1D Gaussian window
        sigma = self.sigma
        gauss = torch.exp(
            -torch.arange(-window_size//2, window_size//2+1)**2 / (2*sigma**2)
        )
        gauss = gauss / gauss.sum()
        
        # Convert to 2D window
        window_1d = gauss.unsqueeze(1)
        window_2d = window_1d @ window_1d.t()
        
        # Expand to channel dimension
        window = window_2d.expand(self.channel, 1, window_size, window_size)
        
        return window
    
    def __call__(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Calculate SSIM between two images.
        
        Args:
            img1: First image tensor [3, H, W]
            img2: Second image tensor [3, H, W]
            
        Returns:
            SSIM value (higher is better)
        """
        # Check shapes
        if img1.shape != img2.shape:
            raise ValueError(f"Images must have the same shape: {img1.shape} vs {img2.shape}")
        
        # Ensure 4D tensor: [1, channel, height, width]
        if len(img1.shape) == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
        
        device = img1.device
        
        # Create Gaussian window
        window = self._create_window(self.window_size).to(device)
        
        # Calculate means
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=self.channel)
        
        # Calculate squares
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Calculate variances and covariance
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=self.channel) - mu1_mu2
        
        # Calculate SSIM
        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))
        
        # Average over spatial dimensions
        ssim_value = ssim_map.mean()
        
        return ssim_value.item()


class TemporalConsistencyMetric:
    """
    Metric for evaluating temporal consistency in animations.
    
    This metric measures how smoothly the animation transitions between frames,
    detecting any sudden changes or jitter.
    """
    
    def __init__(self, use_flow: bool = True):
        """
        Initialize temporal consistency metric.
        
        Args:
            use_flow: Whether to use optical flow for evaluation
        """
        self.use_flow = use_flow
    
    def _compute_flow(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Compute optical flow between two frames.
        
        Args:
            img1: First frame (BGR format)
            img2: Second frame (BGR format)
            
        Returns:
            Optical flow field
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        return flow
    
    def _compute_warped_error(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        flow: np.ndarray
    ) -> np.ndarray:
        """
        Compute error between second image and warped first image.
        
        Args:
            img1: First frame
            img2: Second frame
            flow: Optical flow from first to second frame
            
        Returns:
            Error map
        """
        h, w = flow.shape[:2]
        
        # Create sampling grid
        grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Apply flow to grid
        grid_x = grid_x + flow[..., 0]
        grid_y = grid_y + flow[..., 1]
        
        # Warp image
        warped = cv2.remap(img1, grid_x, grid_y, cv2.INTER_LINEAR)
        
        # Compute error
        error = np.abs(warped.astype(np.float32) - img2.astype(np.float32))
        
        return error
    
    def __call__(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """
        Calculate temporal consistency metrics for a sequence of frames.
        
        Args:
            frames: List of frames in RGB format
            
        Returns:
            Dictionary of temporal consistency metrics
        """
        if len(frames) < 2:
            raise ValueError("Need at least 2 frames to calculate temporal consistency")
        
        # Convert to BGR for OpenCV
        frames_bgr = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]
        
        # Calculate metrics
        direct_diff_values = []
        warped_diff_values = []
        
        for i in range(1, len(frames_bgr)):
            # Calculate direct pixel difference
            direct_diff = np.abs(frames_bgr[i].astype(np.float32) - frames_bgr[i-1].astype(np.float32))
            direct_diff_mean = np.mean(direct_diff)
            direct_diff_values.append(direct_diff_mean)
            
            # Calculate flow-based difference if needed
            if self.use_flow:
                flow = self._compute_flow(frames_bgr[i-1], frames_bgr[i])
                warped_diff = self._compute_warped_error(frames_bgr[i-1], frames_bgr[i], flow)
                warped_diff_mean = np.mean(warped_diff)
                warped_diff_values.append(warped_diff_mean)
        
        # Calculate averages
        avg_direct_diff = np.mean(direct_diff_values)
        
        metrics = {
            'direct_diff': float(avg_direct_diff),
        }
        
        if self.use_flow:
            avg_warped_diff = np.mean(warped_diff_values)
            metrics['warped_diff'] = float(avg_warped_diff)
        
        return metrics


class ExpressionAccuracyMetric:
    """
    Metric for evaluating facial expression accuracy.
    
    This metric measures how well the generated animations capture
    the facial expressions from the motion source.
    """
    
    def __init__(self, facial_landmark_detector=None):
        """
        Initialize expression accuracy metric.
        
        Args:
            facial_landmark_detector: Optional facial landmark detector model
        """
        self.detector = facial_landmark_detector
        
        # If no detector provided, try to initialize one (placeholder)
        if self.detector is None:
            try:
                import mediapipe as mp
                self.mp = mp
                self.mp_face_mesh = mp.solutions.face_mesh
                self.detector = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    min_detection_confidence=0.5
                )
                self.has_detector = True
            except ImportError:
                print("MediaPipe not available for expression accuracy metric")
                self.has_detector = False
    
    def _extract_landmarks(self, image: np.ndarray) -> np.ndarray:
        """
        Extract facial landmarks from an image.
        
        Args:
            image: RGB image
            
        Returns:
            Array of facial landmarks
        """
        if not self.has_detector:
            # Return placeholder if no detector
            return np.zeros((68, 2))
        
        # Process image with MediaPipe
        results = self.detector.process(image)
        
        # Check if face detected
        if not results.multi_face_landmarks:
            return np.zeros((68, 2))
        
        # Extract landmark coordinates
        h, w = image.shape[:2]
        landmarks = []
        
        for landmark in results.multi_face_landmarks[0].landmark:
            landmarks.append([landmark.x * w, landmark.y * h])
        
        # Convert to numpy array
        landmarks = np.array(landmarks)
        
        # Select subset of landmarks for compatibility (68 points)
        # This is a simplified mapping - in a real implementation, a proper mapping would be used
        if len(landmarks) > 68:
            indices = np.linspace(0, len(landmarks) - 1, 68).astype(int)
            landmarks = landmarks[indices]
        
        return landmarks
    
    def _calculate_expression_difference(
        self,
        source_landmarks: np.ndarray,
        generated_landmarks: np.ndarray
    ) -> float:
        """
        Calculate difference between expressions based on landmarks.
        
        Args:
            source_landmarks: Landmarks from source frame
            generated_landmarks: Landmarks from generated frame
            
        Returns:
            Expression difference score (lower is better)
        """
        # Normalize landmarks to remove effects of translation and scale
        def normalize_landmarks(landmarks):
            # Center landmarks
            center = np.mean(landmarks, axis=0)
            centered = landmarks - center
            
            # Scale landmarks
            scale = np.max(np.abs(centered))
            if scale > 0:
                normalized = centered / scale
            else:
                normalized = centered
            
            return normalized
        
        source_norm = normalize_landmarks(source_landmarks)
        generated_norm = normalize_landmarks(generated_landmarks)
        
        # Calculate Euclidean distance between corresponding landmarks
        distances = np.sqrt(np.sum((source_norm - generated_norm) ** 2, axis=1))
        
        # Take mean distance as the difference score
        mean_distance = np.mean(distances)
        
        return float(mean_distance)
    
    def __call__(
        self,
        source_frames: List[np.ndarray],
        generated_frames: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Calculate expression accuracy metrics.
        
        Args:
            source_frames: List of source frames in RGB format
            generated_frames: List of generated frames in RGB format
            
        Returns:
            Dictionary of expression accuracy metrics
        """
        if len(source_frames) != len(generated_frames):
            raise ValueError("Source and generated frame sequences must have the same length")
        
        if not self.has_detector:
            return {'expression_error': float('nan')}
        
        # Extract landmarks for all frames
        source_landmarks = [self._extract_landmarks(frame) for frame in source_frames]
        generated_landmarks = [self._extract_landmarks(frame) for frame in generated_frames]
        
        # Calculate expression differences
        differences = [
            self._calculate_expression_difference(s, g)
            for s, g in zip(source_landmarks, generated_landmarks)
        ]
        
        # Calculate metrics
        mean_diff = np.mean(differences)
        max_diff = np.max(differences)
        std_diff = np.std(differences)
        
        return {
            'expression_error_mean': float(mean_diff),
            'expression_error_max': float(max_diff),
            'expression_error_std': float(std_diff)
        }


class IdentityPreservationMetric:
    """
    Metric for evaluating identity preservation in animations.
    
    This metric measures how well the generated animations maintain
    the identity of the reference image.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize identity preservation metric.
        
        Args:
            model_path: Optional path to a pre-trained face recognition model
        """
        # In a real implementation, this would load a proper face recognition model
        # For simplicity, we'll use a placeholder ResNet18
        self.model = torch.nn.Sequential(
            # Use a pre-trained ResNet18 without the classifier
            torch.nn.Sequential(*list(torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).children())[:-1]),
            # Add a flattening layer
            torch.nn.Flatten()
        )
        
        # Set to evaluation mode
        self.model.eval()
        
        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def _extract_face(self, image: np.ndarray) -> np.ndarray:
        """
        Extract face from an image.
        
        In a real implementation, this would use a face detector.
        For simplicity, we'll assume the face is centered.
        
        Args:
            image: RGB image
            
        Returns:
            Face image
        """
        # Simple center crop (this is a placeholder)
        h, w = image.shape[:2]
        min_dim = min(h, w)
        
        # Get center crop
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        face = image[top:top+min_dim, left:left+min_dim]
        
        # Resize to 224x224 for the model
        face = cv2.resize(face, (224, 224))
        
        return face
    
    def _extract_identity_features(self, image: np.ndarray) -> torch.Tensor:
        """
        Extract identity features from an image.
        
        Args:
            image: RGB image
            
        Returns:
            Identity feature vector
        """
        # Extract face
        face = self._extract_face(image)
        
        # Convert to tensor
        face_tensor = torch.from_numpy(face).float().permute(2, 0, 1) / 255.0
        face_tensor = face_tensor.unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(face_tensor)
        
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        return features
    
    def __call__(
        self,
        reference_image: np.ndarray,
        generated_frames: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Calculate identity preservation metrics.
        
        Args:
            reference_image: Reference image in RGB format
            generated_frames: List of generated frames in RGB format
            
        Returns:
            Dictionary of identity preservation metrics
        """
        # Extract features from reference image
        reference_features = self._extract_identity_features(reference_image)
        
        # Extract features from each generated frame
        frame_features = [self._extract_identity_features(frame) for frame in generated_frames]
        
        # Calculate cosine similarity between reference and each frame
        similarities = [F.cosine_similarity(reference_features, frame_feat).item() 
                       for frame_feat in frame_features]
        
        # Calculate metrics
        mean_sim = np.mean(similarities)
        min_sim = np.min(similarities)
        std_sim = np.std(similarities)
        
        return {
            'identity_similarity_mean': float(mean_sim),
            'identity_similarity_min': float(min_sim),
            'identity_similarity_std': float(std_sim)
        }


def evaluate_animation(
    reference_image_path: str,
    source_video_path: str,
    generated_video_path: str,
    output_path: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate an animation using multiple metrics.
    
    Args:
        reference_image_path: Path to the reference image
        source_video_path: Path to the source motion video
        generated_video_path: Path to the generated animation
        output_path: Optional path to save evaluation results
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Load reference image
    reference_image = cv2.imread(reference_image_path)
    reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
    
    # Load source video frames
    source_frames = []
    cap = cv2.VideoCapture(source_video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        source_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    # Load generated video frames
    generated_frames = []
    cap = cv2.VideoCapture(generated_video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        generated_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    # Ensure same number of frames
    min_frames = min(len(source_frames), len(generated_frames))
    source_frames = source_frames[:min_frames]
    generated_frames = generated_frames[:min_frames]
    
    # Initialize metrics
    psnr = PSNR()
    ssim = SSIM()
    temporal_metric = TemporalConsistencyMetric()
    expression_metric = ExpressionAccuracyMetric()
    identity_metric = IdentityPreservationMetric()
    
    # Calculate FID (using a subset of frames)
    try:
        fid_calculator = FrechetInceptionDistance()
        # Sample frames for FID calculation (to save memory)
        frame_indices = np.linspace(0, min_frames - 1, min(min_frames, 50)).astype(int)
        
        # Convert frames to tensors
        source_tensors = []
        generated_tensors = []
        
        for idx in frame_indices:
            # Source frame
            s_frame = torch.from_numpy(source_frames[idx]).float().permute(2, 0, 1) / 255.0
            source_tensors.append(s_frame)
            
            # Generated frame
            g_frame = torch.from_numpy(generated_frames[idx]).float().permute(2, 0, 1) / 255.0
            generated_tensors.append(g_frame)
        
        source_batch = torch.stack(source_tensors).to(fid_calculator.device)
        generated_batch = torch.stack(generated_tensors).to(fid_calculator.device)
        
        fid_score = fid_calculator.calculate_fid(source_batch, generated_batch)
        has_fid = True
    except Exception as e:
        print(f"Error calculating FID: {e}")
        fid_score = float('nan')
        has_fid = False
    
    # Calculate PSNR and SSIM for each frame
    psnr_values = []
    ssim_values = []
    
    for s_frame, g_frame in zip(source_frames, generated_frames):
        # Convert to tensors
        s_tensor = torch.from_numpy(s_frame).float().permute(2, 0, 1) / 255.0
        g_tensor = torch.from_numpy(g_frame).float().permute(2, 0, 1) / 255.0
        
        # Calculate metrics
        psnr_values.append(psnr(s_tensor, g_tensor))
        ssim_values.append(ssim(s_tensor, g_tensor))
    
    # Calculate temporal consistency
    temporal_metrics = temporal_metric(generated_frames)
    
    # Calculate expression accuracy
    expression_metrics = expression_metric(source_frames, generated_frames)
    
    # Calculate identity preservation
    identity_metrics = identity_metric(reference_image, generated_frames)
    
    # Compile all metrics
    metrics = {
        'fid': fid_score if has_fid else float('nan'),
        'psnr_mean': float(np.mean(psnr_values)),
        'psnr_min': float(np.min(psnr_values)),
        'ssim_mean': float(np.mean(ssim_values)),
        'ssim_min': float(np.min(ssim_values)),
        **temporal_metrics,
        **expression_metrics,
        **identity_metrics
    }
    
    # Save metrics to file if requested
    if output_path:
        import json
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate animations with ChimeraAI metrics")
    parser.add_argument("--reference", type=str, required=True, help="Path to reference image")
    parser.add_argument("--source", type=str, required=True, help="Path to source motion video")
    parser.add_argument("--generated", type=str, required=True, help="Path to generated animation")
    parser.add_argument("--output", type=str, help="Path to save evaluation results")
    
    args = parser.parse_args()
    
    metrics = evaluate_animation(
        args.reference,
        args.source,
        args.generated,
        args.output
    )
    
    # Print metrics
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
