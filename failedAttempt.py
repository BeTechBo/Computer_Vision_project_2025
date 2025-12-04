import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt


# Display function for Colab
def display_saved_step(img, title):
    """Display an image in Colab using matplotlib"""
    plt.figure(figsize=(5, 5))
    if len(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


star_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)


class ImageCache:
    """Cache for reference images to avoid redundant I/O"""
    def __init__(self):
        self._ref_image = None

    def get_reference(self, ref_path):
        """Load and preprocess reference image once"""
        if self._ref_image is None and ref_path and os.path.exists(ref_path):
            ref_img = cv2.imread(ref_path)
            if ref_img is not None:
                self._ref_image = self._preprocess_reference(ref_img)
        return self._ref_image

    def _preprocess_reference(self, img):
        """Preprocess reference image for histogram matching"""
        img = resize(img)
        img = to_grayscale(img)
        img, _ = ensure_white_background(img)
        return img

    def reset(self):
        """Clear cache"""
        self._ref_image = None


image_cache = ImageCache()


def resize(img, max_side=640):
    """Resizes the image's largest side to 640 and the other proportionally."""
    factor = max_side / max(img.shape)
    img = cv2.resize(img.copy(), None, fx=factor, fy=factor,
                     interpolation=cv2.INTER_AREA)
    return img


def to_grayscale(img):
    """Converts a colored image to grayscale"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def is_inverted(image):
    """Detect if image is inverted by analyzing edge pixels strategically."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = image.shape
    sample_points = [
        image[0, 0], image[0, width // 2], image[0, -1],
        image[height // 2, 0], image[height // 2, -1],
        image[-1, 0], image[-1, width // 2], image[-1, -1]
    ]
    white_count = sum(1 for p in sample_points if p > 128)
    white_ratio = white_count / len(sample_points)
    return white_ratio < 0.5


def ensure_white_background(image):
    """Ensure image has white background (normal orientation)"""
    was_inverted = is_inverted(image)
    if was_inverted:
        print("  Detected inverted image - auto-correcting...")
        return 255 - image, True
    return image, False


def match_histogram(source, template):
    """Match histogram - VECTORIZED version (50-100x faster)"""
    if len(source.shape) == 3:
        source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    if len(template.shape) == 3:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    src_hist = cv2.calcHist([source], [0], None, [256], [0, 256])
    tpl_hist = cv2.calcHist([template], [0], None, [256], [0, 256])

    eps = 1e-10
    src_hist = src_hist / (src_hist.sum() + eps)
    tpl_hist = tpl_hist / (tpl_hist.sum() + eps)

    src_cdf = np.cumsum(src_hist).ravel()
    tpl_cdf = np.cumsum(tpl_hist).ravel()

    src_cdf = (src_cdf - src_cdf.min()) * 255 / (src_cdf.max() - src_cdf.min() + eps)
    tpl_cdf = (tpl_cdf - tpl_cdf.min()) * 255 / (tpl_cdf.max() - tpl_cdf.min() + eps)

    lookup_table = np.argmin(
        np.abs(tpl_cdf.reshape(-1, 1) - src_cdf.reshape(1, -1)), 
        axis=0
    ).astype(np.uint8)

    matched = cv2.LUT(source, lookup_table)
    return matched


def normalize_brightness_contrast(image, reference_image):
    """Normalize brightness and contrast to match reference image"""
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image.copy()

    if len(reference_image.shape) == 3:
        reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    else:
        reference_gray = reference_image.copy()

    if reference_gray.shape != image_gray.shape:
        reference_gray = cv2.resize(reference_gray, 
                                    (image_gray.shape[1], image_gray.shape[0]))

    normalized = match_histogram(image_gray, reference_gray)
    return normalized


def threshold(img, blur_size=13):
    """Applies a blur and thresholding to a grayscale image."""
    img = cv2.GaussianBlur(img, ksize=(blur_size, blur_size), sigmaX=0)
    img = cv2.adaptiveThreshold(
        img, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        thresholdType=cv2.THRESH_BINARY, blockSize=11, C=3)
    img = cv2.dilate(img, kernel=star_kernel)
    return img


def denoise_salt_pepper(img, method='median', strength=5):
    """Remove salt-and-pepper noise from binary image"""
    if strength % 2 == 0:
        strength += 1

    if method == 'median':
        denoised = cv2.medianBlur(img, strength)
    elif method == 'morphological':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (strength, strength))
        denoised = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    else:
        denoised = img

    return denoised


def remove_periodic_noise(img, block_size=5):
    """Remove periodic noise using FFT"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    f_transform = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f_transform)

    rows, cols = img.shape
    u = np.arange(rows)
    v = np.arange(cols)
    u = np.where(u < rows / 2, u, u - rows)
    v = np.where(v < cols / 2, v, v - cols)
    u, v = np.meshgrid(u, v, indexing='ij')

    sigma = 30
    lpf = np.exp(-(u**2 + v**2) / (2 * sigma**2))
    hpf = 1 - lpf

    f_filtered = f_shift * (0.7 + 0.3 * hpf)
    f_ishift = np.fft.ifftshift(f_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = np.clip(img_back, 0, 255).astype(np.uint8)

    return img_back


def enhance_low_contrast_regions(img, clip_limit=2.0, tile_size=8):
    """Enhance visibility using CLAHE for barely visible numbers"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(img)

    return enhanced


def correct_illumination_uneven(img, reference_img=None):
    """Correct uneven illumination (one side brighter than other)"""
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (80, 80))
    background = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (99, 99), 0)
    background = np.maximum(background, 1)

    corrected = (img_gray.astype(np.float32) / background.astype(np.float32) * 127).astype(np.uint8)
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)

    return corrected


def detect_rotation_angle(img):
    """Detect rotation angle using Hough line transform"""
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()

    edges = cv2.Canny(img_gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    if lines is None or len(lines) < 2:
        return 0

    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = np.degrees(theta)
        if angle > 90:
            angle = angle - 180
        angles.append(angle)

    horizontal_angles = [a for a in angles if abs(a) < 45]

    if not horizontal_angles:
        return 0

    dominant_angle = np.median(horizontal_angles)
    return dominant_angle


def rotate_image(img, angle):
    """Rotate image by specified angle"""
    if angle == 0:
        return img

    height, width = img.shape[:2]
    center = (width // 2, height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    rotated = cv2.warpAffine(img, rotation_matrix, (new_width, new_height),
                            borderMode=cv2.BORDER_CONSTANT, 
                            borderValue=255 if len(img.shape) == 2 else (255, 255, 255))

    return rotated


def extract_grid_only(img, min_area_ratio=0.5):
    """Extract only main grid, avoiding sub-grids"""
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()

    contours, _ = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img_gray

    image_area = img_gray.shape[0] * img_gray.shape[1]
    min_area = image_area * min_area_ratio

    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h != 0 else 0
            if 0.8 < aspect_ratio < 1.2:
                valid_contours.append((area, contour))

    if not valid_contours:
        return img_gray

    largest_area, largest_contour = max(valid_contours, key=lambda x: x[0])

    output = np.zeros_like(img_gray)
    cv2.drawContours(output, [largest_contour], 0, 255, -1)

    return output


def remove_small_components_advanced(img, min_area=25, connectivity=8):
    """Remove small components (noise)"""
    if len(img.shape) == 3:
        img_Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_Gray = img.copy()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        img_Gray, connectivity=connectivity)

    output = np.zeros_like(img_Gray)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            output[labels == i] = 255

    return output


def remove_small_components_adaptive(img, percentile=10, connectivity=8):
    """Remove small components adaptively"""
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        img_gray, connectivity=connectivity)

    areas = stats[1:, cv2.CC_STAT_AREA]

    if len(areas) == 0:
        return img_gray

    min_area = np.percentile(areas, percentile)

    output = np.zeros_like(img_gray)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            output[labels == i] = 255

    return output


def expose_grid(img):
    """Find the biggest connected part (grid) and remove everything else"""
    inverse = 255 - img.copy()
    mask = get_mask_like(img)
    size = min(inverse.shape)
    biggest_area = 0
    point = (0, 0)

    for x in range(size // 4, 3 * (size // 4)):
        if inverse[x, x] == 0:
            area = cv2.floodFill(inverse, mask, seedPoint=(x + 1, x + 1), newVal=64)[0]
            if area > biggest_area:
                biggest_area = area
                point = (x, x)

    inverse[inverse != 64] = 0
    mask = get_mask_like(inverse)
    cv2.floodFill(inverse, mask, seedPoint=point, newVal=255)
    inverse[inverse != 255] = 0

    result = cv2.erode(inverse, kernel=star_kernel)
    return result


def find_corners(img):
    """Find corners of the grid"""
    height, width = img.shape[0] - 1, img.shape[1] - 1
    corners = np.zeros((4, 2), np.int32)
    LU_not, RU_not = True, True
    LD_not, RD_not = True, True

    for dist in range(min(img.shape)):
        for x in range(dist + 1):
            y = dist - x
            if img[y, x] == 255 and LU_not:
                corners[0] = (x, y)
                LU_not = False
            if img[y, width - x] == 255 and RU_not:
                corners[1] = (width - x, y)
                RU_not = False
            if img[height - y, x] == 255 and LD_not:
                corners[2] = (x, height - y)
                LD_not = False
            if img[height - y, width - x] == 255 and RD_not:
                corners[3] = ((width - x), (height - y))
                RD_not = False
        if not (LU_not or RU_not or LD_not or RD_not):
            break

    return corners


def transform(img, corners, size=288):
    """Apply perspective transform"""
    assert (size % 9) == 0
    boundary = np.float32([[0, 0], [size, 0], [0, size], [size, size]])
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), boundary)
    img = cv2.warpPerspective(img, M, (size, size))
    return img


def get_mask_like(img):
    """Create mask with padding"""
    return np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)


def save_step_image(img, step_name, output_dir, prefix_fileN):
    """Save step image"""
    if img is None:
        return

    base_name = os.path.splitext(prefix_fileN)[0]
    out_fil_name = f"{base_name}_{step_name}.png"
    output_path = os.path.join(output_dir, out_fil_name)

    cv2.imwrite(output_path, img)
    return output_path


def preprocess_sudoku_image(image_path, output_size=288, save_steps=False, 
                           s_out_dir=None, prefix_fileN=None, reference_img=None,
                           handle_rotation=True, fix_illumination=True, 
                           denoise_method='median', enhance_contrast=True):
    """
    Process a single sudoku image through the preprocessing pipeline.
    Enhanced version with rotation, illumination correction, and advanced denoising.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None

    if save_steps and s_out_dir:
        save_step_image(img, "00_original", s_out_dir, prefix_fileN)
        display_saved_step(img, "Step 00: Original")

    img_resized = resize(img)
    if save_steps and s_out_dir:
        save_step_image(img_resized, "01_resized", s_out_dir, prefix_fileN)
        display_saved_step(img_resized, "Step 01: Resized")

    img_Gray = to_grayscale(img_resized)
    if save_steps and s_out_dir:
        save_step_image(img_Gray, "02_grayscale", s_out_dir, prefix_fileN)
        display_saved_step(img_Gray, "Step 02: Grayscale")

    if fix_illumination:
        img_Gray = correct_illumination_uneven(img_Gray)
        if save_steps and s_out_dir:
            save_step_image(img_Gray, "02a_illumination_corrected", s_out_dir, prefix_fileN)
            display_saved_step(img_Gray, "Step 02a: Illumination Corrected")

    if enhance_contrast:
        img_Gray = enhance_low_contrast_regions(img_Gray, clip_limit=2.5, tile_size=8)
        if save_steps and s_out_dir:
            save_step_image(img_Gray, "02b_contrast_enhanced", s_out_dir, prefix_fileN)
            display_saved_step(img_Gray, "Step 02b: Contrast Enhanced")

    if reference_img is not None:
        img_normalized = normalize_brightness_contrast(img_Gray, reference_img)
        if save_steps and s_out_dir:
            save_step_image(img_normalized, "02c_normalized", s_out_dir, prefix_fileN)
            display_saved_step(img_normalized, "Step 02c: Normalized")
        img_Gray = img_normalized

    img_corrected, was_inverted = ensure_white_background(img_Gray)
    if save_steps and s_out_dir:
        save_step_image(img_corrected, "02d_inversion_corrected", s_out_dir, prefix_fileN)
        display_saved_step(img_corrected, "Step 02d: Inversion Corrected")

    img_threshold = threshold(img_corrected)
    if save_steps and s_out_dir:
        save_step_image(img_threshold, "03_threshold", s_out_dir, prefix_fileN)
        display_saved_step(img_threshold, "Step 03: Threshold")

    if denoise_method == 'median':
        img_denoised = denoise_salt_pepper(img_threshold, method='median', strength=5)
    elif denoise_method == 'morphological':
        img_denoised = denoise_salt_pepper(img_threshold, method='morphological', strength=5)
    elif denoise_method == 'combined':
        img_denoised = denoise_salt_pepper(img_threshold, method='median', strength=5)
        img_denoised = remove_periodic_noise(img_denoised, block_size=5)
    else:
        img_denoised = img_threshold

    if save_steps and s_out_dir:
        save_step_image(img_denoised, "03a_denoised", s_out_dir, prefix_fileN)
        display_saved_step(img_denoised, "Step 03a: Denoised")

    img_good = remove_small_components_adaptive(img_denoised, percentile=15, connectivity=8)
    if save_steps and s_out_dir:
        save_step_image(img_good, "03b_cleaned", s_out_dir, prefix_fileN)
        display_saved_step(img_good, "Step 03b: Cleaned")

    img_grid = expose_grid(img_good)
    img_grid = extract_grid_only(img_grid, min_area_ratio=0.5)
    if save_steps and s_out_dir:
        save_step_image(img_grid, "04_grid_extracted", s_out_dir, prefix_fileN)
        display_saved_step(img_grid, "Step 04: Grid Extracted")

    if handle_rotation:
        rotation_angle = detect_rotation_angle(img_grid)
        if abs(rotation_angle) > 2:
            print(f"  Detected rotation: {rotation_angle:.1f}° - correcting...")
            img_grid = rotate_image(img_grid, rotation_angle)
            if save_steps and s_out_dir:
                save_step_image(img_grid, "04a_rotation_corrected", s_out_dir, prefix_fileN)
                display_saved_step(img_grid, "Step 04a: Rotation Corrected")

    corners = find_corners(img_grid)

    if save_steps and s_out_dir:
        corners_vis = cv2.cvtColor(img_grid.copy(), cv2.COLOR_GRAY2BGR)
        for i, corner in enumerate(corners):
            cv2.circle(corners_vis, tuple(corner), 5, (0, 0, 255), -1)
            cv2.putText(corners_vis, str(i), tuple(corner),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        save_step_image(corners_vis, "05_corners_detected", s_out_dir, prefix_fileN)
        display_saved_step(corners_vis, "Step 05: Corners Detected")

    img_transformed = transform(img_good, corners, output_size)
    if save_steps and s_out_dir:
        save_step_image(img_transformed, "06_final_transformed", s_out_dir, prefix_fileN)
        display_saved_step(img_transformed, "Step 06: Final Transformed")

    return img_transformed


def process_images_folder(input_folder="images", output_folder="trial", 
                         save_steps=True, reference_image_path=None,
                         handle_rotation=True, fix_illumination=True,
                         denoise_method='combined', enhance_contrast=True):
    """
    Process all images in the input folder and save preprocessed grids.
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    steps_dir = None
    if save_steps:
        steps_dir = Path(output_folder) / "processing_steps"
        steps_dir.mkdir(parents=True, exist_ok=True)

    reference_img = None
    if reference_image_path and os.path.exists(reference_image_path):
        reference_img = image_cache.get_reference(reference_image_path)
        if reference_img is not None:
            print(f"Using reference image: {reference_image_path}")
        else:
            print(f"Warning: Could not load reference image {reference_image_path}")
    else:
        print("No reference image provided - skipping brightness/contrast normalization")

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in os.listdir(input_folder)
                  if os.path.splitext(f)[1].lower() in image_extensions]

    image_files.sort()

    if not image_files:
        print(f"No images found in {input_folder} folder")
        return

    print(f"Found {len(image_files)} images to process")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    if save_steps:
        print(f"Steps folder: {steps_dir}")

    successful = 0
    failed = 0

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(input_folder, image_file)
        print(f"\nProcessing {image_file} ({i+1}/{len(image_files)})")

        try:
            processed_grid = preprocess_sudoku_image(
                image_path,
                save_steps=save_steps,
                s_out_dir=str(steps_dir) if steps_dir else None,
                prefix_fileN=image_file,
                reference_img=reference_img,
                handle_rotation=handle_rotation,
                fix_illumination=fix_illumination,
                denoise_method=denoise_method,
                enhance_contrast=enhance_contrast
            )

            if processed_grid is not None:
                output_path = Path(output_folder) / image_file
                cv2.imwrite(str(output_path), processed_grid)
                print(f"  ✓ Successfully processed and saved: {output_path}")
                if save_steps:
                    print(f"  ✓ Intermediate steps saved to: {steps_dir}")
                successful += 1
            else:
                print(f"  ✗ Failed to process {image_file}")
                failed += 1

        except Exception as e:
            print(f"  ✗ Error processing {image_file}: {str(e)}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Preprocessing complete!")
    print(f"Successful: {successful}, Failed: {failed}")
    print(f"Preprocessed grids saved to: {output_folder}")
    if save_steps:
        print(f"Processing steps saved to: {steps_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    INPUT_FOLDER = "images"
    OUTPUT_FOLDER = "trial"
    REFERENCE_IMAGE = "images/01.jpg"
    SAVE_STEPS = True

    HANDLE_ROTATION = True
    FIX_ILLUMINATION = True
    DENOISE_METHOD = 'combined'
    ENHANCE_CONTRAST = True

    process_images_folder(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        save_steps=SAVE_STEPS,
        reference_image_path=REFERENCE_IMAGE,
        handle_rotation=HANDLE_ROTATION,
        fix_illumination=FIX_ILLUMINATION,
        denoise_method=DENOISE_METHOD,
        enhance_contrast=ENHANCE_CONTRAST
    )