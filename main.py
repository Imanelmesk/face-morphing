import cv2
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from faceMorphing import (
    get_landmarks,
    warp_image,
    draw_delaunay,
    weighted_average_points,
    align_face
)



# Path to the Dlib predictor
dlib_predictor_path = "Projet/shape_predictor_68_face_landmarks.dat"

# Load source and destination images
src_img = cv2.imread("Projet/rihanna.png")
dst_img = cv2.imread("Projet/jennie.png")

# Resize images for consistency
src_img = cv2.resize(src_img, (800, 800))
dst_img = cv2.resize(dst_img, (800, 800))

# Detect landmarks for alignment
src_initial_landmarks = get_landmarks(src_img, predictor_path=dlib_predictor_path)
dst_initial_landmarks = get_landmarks(dst_img, predictor_path=dlib_predictor_path)

# Align the images based on the initial landmarks
src_img_aligned = align_face(src_img, src_initial_landmarks)
dst_img_aligned = align_face(dst_img, dst_initial_landmarks)

# Detect landmarks again on the aligned images
src_points = get_landmarks(src_img_aligned, predictor_path=dlib_predictor_path)
dst_points = get_landmarks(dst_img_aligned, predictor_path=dlib_predictor_path)



# Create copies of the images for visualizing landmarks
src_img_with_landmarks = src_img_aligned.copy()
dst_img_with_landmarks = dst_img_aligned.copy()

src_with_triangles = src_img.copy()
draw_delaunay(src_with_triangles, src_points)

# Show the image
cv2.imshow("Source Image with Delaunay Triangulation", src_with_triangles)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Draw landmarks on the copies
for point in src_points:
    cv2.circle(src_img_with_landmarks, tuple(point), 3, (0, 0, 255), -1)

for point in dst_points:
    cv2.circle(dst_img_with_landmarks, tuple(point), 3, (0, 255, 0), -1)

# Show images with landmarks
cv2.imshow("Source Landmarks", src_img_with_landmarks)
cv2.imshow("Destination Landmarks", dst_img_with_landmarks)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Check if landmarks are detected
if len(src_points) == 0 or len(dst_points) == 0:
    print("Erreur : Impossible de détecter les points de repère.")
    exit()

# Morphing process
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
alpha_values = np.linspace(1, 0, 9)

for i, (alpha, ax) in enumerate(zip(alpha_values, axes.flat)):
    inter_points = weighted_average_points(src_points, dst_points, alpha)
    src_warped, _ = warp_image(src_img_aligned, src_points, inter_points, src_img_aligned.shape)
    dst_warped, _ = warp_image(dst_img_aligned, dst_points, inter_points, dst_img_aligned.shape)
    blended_img = cv2.addWeighted(src_warped, alpha, dst_warped, 1 - alpha, 0)
    blended_img = cv2.cvtColor(blended_img, cv2.COLOR_BGR2RGB)
    ax.imshow(blended_img)
    ax.set_title(f"Alpha = {alpha:.3f}")
    ax.axis("off")

plt.suptitle("Face Morphing", size=20)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()


