
from cmath import rect
from math import inf
import os
import cv2
import numpy as np

def step_1_instructions(landmarks:np.ndarray,  fold_type:str='square') -> tuple[list, list]:
    if fold_type == 'star':
        # Pick point is landmark 3
        # Place point is midpoint between landmark 1, 6 and 8
        pick_point = [landmarks[2].x, landmarks[2].y]
        place_point_y = (landmarks[0].y + landmarks[5].y + landmarks[7].y) / 3
        place_point_x = (landmarks[0].x + landmarks[5].x + landmarks[7].x) / 3
        place_point = [place_point_x, place_point_y]
    elif fold_type == 'square':
        # Pick point is landmark 3
        # Place point is two thirds between landmark 3 and 6
        pick_point = [landmarks[2].x, landmarks[2].y]
        place_point_x = (landmarks[5].x - landmarks[2].x)*2/3 + landmarks[2].x
        place_point_y = (landmarks[5].y - landmarks[2].y)*2/3 + landmarks[2].y
        place_point = [place_point_x, place_point_y]
    else:
        raise ValueError(f"Unknown fold type: {fold_type}")

    return pick_point, place_point
    
def step_2_instructions(landmarks:np.ndarray, fold_type:str='square') -> tuple[list, list]:
    if fold_type == 'star':
        # Pick point is landmark 6
        # Place point is midpoint between landmark 6 and landmarks_origional 1
        pick_point = [landmarks[5].x, landmarks[5].y]
        place_point_x = (landmarks[5].x - landmarks[0].x)*0.5 + landmarks[0].x
        place_point_y = (landmarks[5].y - landmarks[0].y)*0.5 + landmarks[0].y
        place_point = [place_point_x, place_point_y]
    elif fold_type == 'square':
        # Pick point is landmark 1
        # Place point is two thirds from landmark_origional 1 to 8
        pick_point_x = (landmarks[7].x - landmarks[0].x)*0.1 + landmarks[0].x
        pick_point_y = (landmarks[7].y - landmarks[0].y)*0.1 + landmarks[0].y
        pick_point = [pick_point_x, pick_point_y]
        place_point_x = (landmarks[7].x - landmarks[0].x)*2/3 + landmarks[0].x
        place_point_y = (landmarks[7].y - landmarks[0].y)*2/3 + landmarks[0].y
        place_point = [place_point_x, place_point_y]
    else:
        raise ValueError(f"Unknown fold type: {fold_type}")

    return pick_point, place_point


def step_3_instructions(landmarks:np.ndarray, fold_type:str='square') -> tuple[list, list]:
    if fold_type == 'star':
        # Pick point is landmark 1
        # Place point is the midpoint between landmark 1 and landmarks_origional 6
        pick_point = [landmarks[0].x, landmarks[0].y]
        place_point_x = (landmarks[5].x - landmarks[0].x)*0.5 + landmarks[0].x
        place_point_y = (landmarks[5].y - landmarks[0].y)*0.5 + landmarks[0].y
        place_point = [place_point_x, place_point_y]
    elif fold_type == 'square':
        # Pick point is landmark 6
        # Place point is landmarks_origional 4
        pick_point = [landmarks[5].x, landmarks[5].y] 
        place_point_x = landmarks[3].x
        place_point_y = landmarks[3].y
        place_point = [place_point_x, place_point_y]
    else:
        raise ValueError(f"Unknown fold type: {fold_type}")
    
    return pick_point, place_point


def step_4_instructions(landmarks:np.ndarray, fold_type:str='square') -> tuple[list, list]:
    if fold_type == 'star':
        # Pick point is landmark 8
        # Place point is midpoint between landmark 8 and 3
        pick_point = [landmarks[7].x, landmarks[7].y]
        place_point_x = (landmarks[7].x - landmarks[2].x)*0.5 + landmarks[2].x
        place_point_y = (landmarks[7].y - landmarks[2].y)*0.5 + landmarks[2].y
        place_point = [place_point_x, place_point_y]
    elif fold_type == 'square':
        # Pick point is landmark 8
        # Place point is point two thirds from landmarks_origional 8 to 1
        pick_point_x = (landmarks[0].x - landmarks[7].x)*0.1 + landmarks[7].x
        pick_point_y = (landmarks[0].y - landmarks[7].y)*0.1 + landmarks[7].y
        pick_point = [pick_point_x, pick_point_y]
        place_point_x = (landmarks[0].x - landmarks[7].x)*2/3 + landmarks[7].x
        place_point_y = (landmarks[0].y - landmarks[7].y)*2/3 + landmarks[7].y
        place_point = [place_point_x, place_point_y]

    else:
        raise ValueError(f"Unknown fold type: {fold_type}")
    
    return pick_point, place_point


def step_5_instructions(landmarks:np.ndarray, fold_type:str='square') -> tuple[list, list] | None:
    if fold_type == 'star':
        print("Star fold only has 4 steps")
        return None
    elif fold_type == 'square':
        # Pick point is between landmarks_origional 1 and 8
        # Place point is between landmarks_origional 4 and 5
        pick_point_x = (landmarks[0].x - landmarks[7].x)*0.5 + landmarks[7].x
        pick_point_y = (landmarks[0].y - landmarks[7].y)*0.5 + landmarks[7].y
        pick_point = [pick_point_x, pick_point_y]
        place_point_x = (landmarks[3].x - landmarks[4].x)*0.5 + landmarks[4].x
        place_point_y = (landmarks[3].y - landmarks[4].y)*0.5 + landmarks[4].y
        place_point = [place_point_x, place_point_y]
    else:
        raise ValueError(f"Unknown fold type: {fold_type}")
    
    return pick_point, place_point

    
    
if __name__ == "__main__":
    print("prank")