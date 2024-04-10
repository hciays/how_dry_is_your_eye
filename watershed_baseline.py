import cv2
import torch
import numpy as np


def watershed_segmentation(image):
    image = torch.stack([image] * 3, dim=0).squeeze()
    image_np = image.permute(1, 2, 0).numpy() * 255
    image_np = image_np.astype(np.uint8)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)

    markers = markers + 1

    markers[unknown == 255] = 0

    markers = cv2.watershed(image_np, markers)
    image_np[markers == -1] = [255, 0, 0]

    return image_np
