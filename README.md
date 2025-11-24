# Computer_Vision_project_2025_MileStone_1
# Team members: 
# Ebram Thabet
# Nour Abdelghani
# Adham Khalil
# Eman Abdelhady

Sudoku Grid Preprocessor
A robust preprocessing pipeline for Sudoku puzzle images that transforms raw photos into clean, normalized, perspective-corrected grid images ready for digit recognition.

Features
📸 Handles real-world photos from phones or cameras
🔁 Automatically corrects inverted images (dark background)
🌓 Normalizes brightness and contrast using a reference image
🧹 Removes noise and small disconnected components
🧭 Detects and isolates the Sudoku grid
📐 Applies perspective correction to produce a top-down view
👁️ Visualizes every processing step when run in Google Colab
💾 Saves both final grids and intermediate steps for inspection
Requirements
Python 3.7 or higher
OpenCV (opencv-python)
NumPy
Matplotlib (only for visualization in Colab)
Install dependencies with:
pip install opencv-python numpy matplotlib

How to Add Your Images
This script expects all input images to be in a folder named images in the same directory as the script.

In Google Colab
You have three options to populate the images folder:

Option 1: Upload Files Directly
Run a small setup snippet at the top of your notebook that uses files.upload() to let you select and upload your Sudoku images. Make sure at least one image is named 01.jpg — this will be used as the reference for brightness normalization.

Option 2: Use Google Drive
Mount your Google Drive and create a symbolic link from your Drive image folder to the local images directory. This is ideal if you have many images or want to reuse them across sessions.

Option 3: Download from URLs
If your images are hosted online, you can download them directly into the images folder using wget or requests.

Locally (on your computer)
Create a folder named images in the same directory as your script and place your Sudoku image files inside it (e.g., 01.jpg, puzzle.png, etc.). The script expects images/01.jpg to exist by default, as it is used for histogram matching to normalize lighting across all images.

You can organize your files using your file browser or terminal — just ensure the folder structure matches what the script expects.

Running the Script
Once your images folder is ready:

In Colab: Run the full script. It will process all images, display each step inline, and save results to the trial folder.
Locally: Execute the script with Python. Intermediate and final results will be saved to disk.

Output
After processing, you’ll get:

A trial/ folder containing the final preprocessed grid for each input image (same filenames)
A trial/processing_steps/ folder containing 7 intermediate images per input (e.g., 01_00_original.png, 01_01_resized.png, ..., 01_06_final_transformed.png)
The final output is a 288×288 pixel image (288 = 9 × 32), perfectly aligned and normalized, with only the Sudoku grid visible — ideal for the next stage (digit recognition).

Supported Image Formats:
.jpg / .jpeg
.png
.bmp
.tiff / .tif

License:
MIT License
