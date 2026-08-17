import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_intensities(image_paths, max_samples=500):
    np.random.seed(42)
    if len(image_paths) > max_samples:
        paths = np.random.choice(image_paths, max_samples, replace=False)
    else:
        paths = image_paths
        
    intensities = []
    for p in tqdm(paths, desc="Loading intensities"):
        img = np.array(Image.open(p).convert('L'))
        # Exclude pure black background (0) for a better brain tissue distribution
        brain_pixels = img[img > 5] 
        if len(brain_pixels) > 0:
            # Subsample pixels to save memory
            sub = np.random.choice(brain_pixels, min(1000, len(brain_pixels)), replace=False)
            intensities.extend(sub.tolist())
    return np.array(intensities)

def main():
    brisc_dir = r"D:\Research\Multi-View Fusion Network\datasets\brisc2025\**\*.jpg"
    pmram_dir = r"D:\Research\Multi-View Fusion Network\datasets\PMRAM Bangladeshi Brain Cancer - MRI Dataset\Raw\**\*.jpg"
    ajbds_dir = r"D:\Research\Multi-View Fusion Network\datasets\AJBDS-2023_extracted\AJBDS-2023\Training\**\input\*.jpg"
    
    brisc_paths = glob.glob(brisc_dir, recursive=True)
    pmram_paths = glob.glob(pmram_dir, recursive=True)
    ajbds_paths = glob.glob(ajbds_dir, recursive=True)
    
    print(f"Found {len(brisc_paths)} BRISC2025, {len(pmram_paths)} PMRAM, {len(ajbds_paths)} AJBDS images.")
    
    i_brisc = get_intensities(brisc_paths)
    i_pmram = get_intensities(pmram_paths)
    i_ajbds = get_intensities(ajbds_paths)
    
    os.makedirs('figures/external', exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.hist(i_brisc, bins=50, alpha=0.5, density=True, label=f'BRISC2025 (mean={i_brisc.mean():.1f})', color='blue')
    plt.hist(i_pmram, bins=50, alpha=0.5, density=True, label=f'PMRAM (mean={i_pmram.mean():.1f})', color='red')
    plt.hist(i_ajbds, bins=50, alpha=0.5, density=True, label=f'AJBDS (mean={i_ajbds.mean():.1f})', color='green')
    
    plt.title('Domain Shift: Brain Tissue Intensity Distribution Across Datasets')
    plt.xlabel('Pixel Intensity (Grayscale)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/external/domain_shift_intensity.png', dpi=150)
    plt.close()
    
    # Save a quick markdown report
    with open('reports/domain_shift_analysis.md', 'w') as f:
        f.write("# Domain Shift Analysis\n\n")
        f.write("Analysis of raw grayscale pixel intensities (excluding pure black background > 5).\n\n")
        f.write(f"- BRISC2025: Mean={i_brisc.mean():.2f}, Std={i_brisc.std():.2f}\n")
        f.write(f"- PMRAM: Mean={i_pmram.mean():.2f}, Std={i_pmram.std():.2f}\n")
        f.write(f"- AJBDS-2023: Mean={i_ajbds.mean():.2f}, Std={i_ajbds.std():.2f}\n\n")
        f.write("A severe intensity distribution shift is visible, likely due to different MRI scanner protocols and normalization procedures at the respective institutions.\n")
        
    print("Domain shift analysis complete.")

if __name__ == "__main__":
    main()
