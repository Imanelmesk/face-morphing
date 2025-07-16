import cv2
import dlib
import numpy as np
from scipy.spatial import Delaunay

def get_landmarks(img, add_boundary_point=True, predictor_path="shape_predictor_68_face_landmarks.dat"):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    try:
        dets = detector(img, 1)
        points = np.zeros([68, 2])
        for k, d in enumerate(dets):
            shape = predictor(img, d)
            for i in range(68):
                points[i, 0] = shape.part(i).x
                points[i, 1] = shape.part(i).y
    except Exception as e:
        print("Failed running face points:", e)
        return []
    points = points.astype(np.int32)
    return points

def weighted_average_points(src_points, dst_points, alpha):
    intermediate_points = (1 - alpha) * np.array(src_points) + alpha * np.array(dst_points)
    print(f"Alpha: {alpha}, Intermediate Points: {intermediate_points[:5]}")  # Vérifiez les 5 premiers points
    return intermediate_points


def get_triangular_affine_matrices(vertices, src_points, dest_points):
    ones = [1, 1, 1]
    for tri_indices in vertices:
        src_tri = np.vstack((src_points[tri_indices, :].T, ones))
        dst_tri = np.vstack((dest_points[tri_indices, :].T, ones))
        mat = np.dot(src_tri, np.linalg.inv(dst_tri))[:2, :]
        yield mat

def process_warp(src_img, result_img, tri_affines, dest_points, delaunay):
    roi_coords = get_grid_coordinates(result_img.shape)
    roi_tri_indices = delaunay.find_simplex(roi_coords)

    for simplex_index in range(len(delaunay.simplices)):
        coords = roi_coords[roi_tri_indices == simplex_index]
        if len(coords) == 0:
            continue
        coords = coords.T
        x, y = coords
        out_coords = np.dot(tri_affines[simplex_index], np.vstack((x, y, np.ones(len(x)))))
        out_coords = out_coords[:2].T
        result_img[y, x] = bilinear_interpolate(src_img, out_coords)




def warp_image(src_img, src_points, dest_points, dest_shape):
    result_img = np.zeros(dest_shape, dtype=src_img.dtype)
    delaunay = Delaunay(dest_points)
    tri_affines = []

    for tri_indices in delaunay.simplices:
        src_tri = np.vstack([src_points[tri_indices].T, np.ones(3)])
        dest_tri = np.vstack([dest_points[tri_indices].T, np.ones(3)])
        affine = np.dot(src_tri, np.linalg.inv(dest_tri))[:2]
        tri_affines.append(affine)

    process_warp(src_img, result_img, tri_affines, dest_points, delaunay)
    return result_img, delaunay

def get_grid_coordinates(image_shape):
    height, width = image_shape[:2]
    y_coords, x_coords = np.meshgrid(range(height), range(width), indexing='ij')
    coords = np.stack([x_coords.ravel(), y_coords.ravel()], axis=-1)
    return coords


def bilinear_interpolate(image, coords):
    if coords.shape[1] != 2:
        raise ValueError(f"Expected coords to have shape (*, 2), got {coords.shape}")

    x, y = coords.T
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, image.shape[1] - 1)
    x1 = np.clip(x1, 0, image.shape[1] - 1)
    y0 = np.clip(y0, 0, image.shape[0] - 1)
    y1 = np.clip(y1, 0, image.shape[0] - 1)

    Ia = image[y0, x0]
    Ib = image[y1, x0]
    Ic = image[y0, x1]
    Id = image[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa[:, None] * Ia + wb[:, None] * Ib + wc[:, None] * Ic + wd[:, None] * Id

# Function to draw Delaunay triangulation on an image
def draw_delaunay(img, points, color=(0, 255, 0)):
    delaunay = Delaunay(points)
    for simplex in delaunay.simplices:
        pts = points[simplex].astype(int)
        cv2.line(img, tuple(pts[0]), tuple(pts[1]), color, 1)
        cv2.line(img, tuple(pts[1]), tuple(pts[2]), color, 1)
        cv2.line(img, tuple(pts[2]), tuple(pts[0]), color, 1)

# Function to align face based on eye landmarks
def align_face(image, landmarks):
    # Compute the center of the eyes
    left_eye = landmarks[36:42].mean(axis=0)
    right_eye = landmarks[42:48].mean(axis=0)

    # Calculate the angle between the eyes
    delta_x = right_eye[0] - left_eye[0]
    delta_y = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(delta_y, delta_x))

    # Compute the center between the eyes and cast to int
    center = tuple(map(int, ((left_eye + right_eye) / 2).tolist()))

    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotate the image
    aligned_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    return aligned_image