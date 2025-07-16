# 🌀 Face Morphing with Dlib & OpenCV

This project demonstrates how to morph one face into another using facial landmark detection and image warping techniques. The final result is a smooth transition from one image (Jennie) to another (Rihanna).

---

## 🖼️ Example

<p align="center">
  <img src="output/morphing_example.gif" alt="Morphing Example" width="500"/>
</p>

---

## 📁 Project Structure

````bash
face-morphing/
├── faceMorphing.py            # Core morphing logic (landmarks, triangulation, warping)
├── main.py                    # Main pipeline: load → morph → save
├── jennie.jpg                 # Source image
├── rihanna.jpg                # Destination image
├── shape_predictor_68_face_landmarks.dat  # Required Dlib model (not included here)
├── output/                    # Generated intermediate frames and final result
└── README.md

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install numpy opencv-python dlib matplotlib scipy
````

### 2. Download the Dlib landmark model

Download `shape_predictor_68_face_landmarks.dat` from:  
https://github.com/davisking/dlib-models

Place it in the root directory of the project.

### 3. Run the script

```bash
python main.py
```

This will:

- Load `jennie.jpg` and `rihanna.jpg`
- Detect facial landmarks
- Compute Delaunay triangulation
- Apply triangle warping and blending
- Save intermediate frames and an animation to the `output/` folder

---

## 📈 Results

The pipeline outputs images showing gradual morphing with smooth transitions for different α values:

    α = 0.0 → source face

    α = 0.5 → blended face

    α = 1.0 → destination face

Artifacts may appear in challenging cases (e.g. poor lighting, occlusion), mostly due to landmark misalignment or interpolation errors.

## 🧠 How It Works

1. **Landmark Detection**  
   Dlib detects 68 key facial landmarks on both faces.

2. **Face Alignment**  
   Align faces based on eye position for consistent transformations.

3. **Delaunay Triangulation**  
   Divide the face into triangles based on landmarks.

4. **Warping and Interpolation**  
   Affine transformations are applied per triangle and pixel values are blended.

5. **Result Generation**  
   Intermediate images are saved for various `α` values (0 → 1) and can be used to generate an animation.

---

## ⚠️ Known Limitations

- Landmark detection may fail with rotated or low-quality images.

- Delaunay triangulation is sensitive to landmark precision.

- Bilinear interpolation can create distortions near high-contrast features (eyes, mouth).

## 🔧 Customization

- Replace `jennie.jpg` and `rihanna.jpg` with your own front-facing images.
- Adjust the morphing steps or animation frame rate in `main.py`.
- Export to GIF using tools like `imageio` or OpenCV `VideoWriter`.

---

## 🔧 Possible Improvements

- Use deep learning models (e.g., MediaPipe, FaceMesh) for more accurate landmarks.

- Improve blending with spline interpolation or Poisson blending.

- Add 3D morphing support.

- Deploy as a web or mobile app for real-time morphing.

## 🙋 Author

**Imane El Meskiri**  
_Image Processing Project, ENSSAT (2024/2025)_

---

## 📜 License

MIT License
