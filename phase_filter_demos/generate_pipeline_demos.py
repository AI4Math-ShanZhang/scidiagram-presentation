"""
Phase-Selective Filtering Pipeline Demo
Generates pipeline visualization: Image → Kernels → Response → Peaks → Tokens

This is standard convolution-based detection (position-invariant),
which correctly detects primitives regardless of position.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter
import os
import glob


# ============================================================
# Kernel Design (Position-Invariant Filters)
# ============================================================

def create_line_kernel(size=11, angle=0):
    """
    Create a line detection kernel at a specific angle.
    Positive along the line, negative perpendicular.
    """
    kernel = np.zeros((size, size))
    center = size // 2

    for i in range(size):
        for j in range(size):
            x = (j - center) * np.cos(angle) + (i - center) * np.sin(angle)
            y = -(j - center) * np.sin(angle) + (i - center) * np.cos(angle)
            kernel[i, j] = np.exp(-y**2 / 2) - 0.5 * np.exp(-y**2 / 8)

    kernel = kernel - kernel.mean()
    kernel = kernel / (np.abs(kernel).sum() + 1e-8)
    return kernel


def create_circle_kernel(size=15, radius=5):
    """
    Create a circle/arc detection kernel.
    Responds to circular patterns.
    """
    kernel = np.zeros((size, size))
    center = size // 2

    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            kernel[i, j] = np.exp(-(dist - radius)**2 / 2) - 0.3 * np.exp(-(dist - radius)**2 / 8)

    kernel = kernel - kernel.mean()
    kernel = kernel / (np.abs(kernel).sum() + 1e-8)
    return kernel


# ============================================================
# Detection (Standard Convolution - Position Invariant)
# ============================================================

def detect_primitives(image, kernel):
    """
    Standard convolution - same kernel response for same content
    regardless of position. This is CORRECT for detection.
    """
    response = convolve2d(image, kernel, mode='same', boundary='symm')
    return response


def find_peaks(response, threshold_ratio=0.4, min_distance=15):
    """Find local maxima in response map."""
    local_max = maximum_filter(response, size=min_distance)
    threshold = threshold_ratio * response.max()
    peaks = (response == local_max) & (response > threshold)

    peak_coords = np.where(peaks)
    peak_values = response[peaks]

    if len(peak_values) == 0:
        return []

    sorted_idx = np.argsort(peak_values)[::-1]
    return list(zip(peak_coords[0][sorted_idx], peak_coords[1][sorted_idx]))


# ============================================================
# Pipeline Visualization
# ============================================================

def visualize_pipeline(image_path, output_path):
    """
    Generate pipeline visualization:
    Image → Kernels → Response Maps → Peaks → Semantic Tokens
    """
    # Load image
    img = Image.open(image_path).convert('L')
    img_array = np.array(img, dtype=float) / 255.0

    # Invert if background is white
    if img_array.mean() > 0.5:
        img_array = 1 - img_array

    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 5, hspace=0.3, wspace=0.25)

    # Define kernels
    kernels = {
        'Line (H)': create_line_kernel(11, 0),
        'Line (V)': create_line_kernel(11, np.pi/2),
        'Line (D)': create_line_kernel(11, np.pi/4),
        'Circle': create_circle_kernel(15, 5),
    }
    colors = ['red', 'blue', 'green', 'orange']
    markers = ['s', 's', 's', 'o']

    # Original image (large, spans 2 rows)
    ax_orig = fig.add_subplot(gs[0:2, 0])
    ax_orig.imshow(1 - img_array, cmap='gray')
    ax_orig.set_title('Input Diagram', fontsize=14, fontweight='bold')
    ax_orig.axis('off')

    # Process each kernel
    all_peaks = []

    for idx, (name, kernel) in enumerate(kernels.items()):
        # Show kernel
        ax_k = fig.add_subplot(gs[0, idx+1])
        ax_k.imshow(kernel, cmap='RdBu', vmin=-abs(kernel).max(), vmax=abs(kernel).max())
        ax_k.set_title(f'Kernel: {name}', fontsize=10)
        ax_k.axis('off')

        # Compute response (standard convolution - position invariant)
        response = detect_primitives(img_array, kernel)

        # Show response with peaks
        ax_r = fig.add_subplot(gs[1, idx+1])
        ax_r.imshow(response, cmap='hot')

        # Find and mark peaks
        peaks = find_peaks(response, threshold_ratio=0.35, min_distance=20)[:6]
        for (py, px) in peaks:
            ax_r.plot(px, py, 'c+', markersize=12, markeredgewidth=2)
            all_peaks.append((py, px, name, colors[idx], markers[idx]))

        ax_r.set_title('Response + Peaks', fontsize=10)
        ax_r.axis('off')

    # Final result: all detected tokens on original image
    ax_final = fig.add_subplot(gs[2, :])
    ax_final.imshow(1 - img_array, cmap='gray')

    # Plot all peaks with different colors for different primitives
    plotted_labels = set()
    for (py, px, name, color, marker) in all_peaks:
        label = name if name not in plotted_labels else None
        ax_final.scatter(px, py, c=color, s=120, marker=marker,
                        edgecolors='white', linewidths=2, label=label)
        plotted_labels.add(name)

    # Legend
    ax_final.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax_final.set_title('Detected Primitives → Semantic Tokens\n'
                      '(Each point = token with position + primitive type)',
                      fontsize=12, fontweight='bold')
    ax_final.axis('off')

    # Main title
    plt.suptitle('Pipeline: Image → Kernels → Response → Peaks → Tokens\n'
                 '(Standard convolution - position invariant detection)',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


def process_all_images(input_dir, output_dir):
    """Process all images in input directory."""
    # Get all PNG images
    image_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))

    print(f"Found {len(image_files)} images")
    print("=" * 60)

    for img_path in image_files:
        img_name = os.path.basename(img_path)
        output_name = f"pipeline_{img_name}"
        output_path = os.path.join(output_dir, output_name)

        try:
            visualize_pipeline(img_path, output_path)
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    input_dir = "/home/ohio_user/code/GPSToken/test_images"
    output_dir = "/home/ohio_user/code/paper_discussion/scale_rope_presentation/phase_filter_demos"

    os.makedirs(output_dir, exist_ok=True)
    process_all_images(input_dir, output_dir)
