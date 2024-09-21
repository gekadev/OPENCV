import cv2
import numpy as np

                 
def getObjectCoordinates(x_center, y_center, box_w, box_h):
    x1 = int(x_center - box_w / 2)
    y1 = int(y_center - box_h / 2)
    x2 = int(x1 + box_w)
    y2 = int(y1 + box_h)
    return x1, y1, x2, y2            


def is_full_object_in_area(area, x1, y1, x2, y2):
    # Check if all four corners of the bounding box are inside the polygon
    top_left = cv2.pointPolygonTest(np.array(area, np.int32), (x1, y1), False)
    top_right = cv2.pointPolygonTest(np.array(area, np.int32), (x2, y1), False)
    bottom_left = cv2.pointPolygonTest(np.array(area, np.int32), (x1, y2), False)
    bottom_right = cv2.pointPolygonTest(np.array(area, np.int32), (x2, y2), False)
    return top_left >= 0 and top_right >= 0 and bottom_left >= 0 and bottom_right >= 0