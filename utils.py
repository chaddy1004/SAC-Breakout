import numpy as np
import cv2
from torchvision.transforms import ToTensor

def frame_to_tensor(raw_rgb_frame):
    gray_frame = cv2.cvtColor(raw_rgb_frame, cv2.COLOR_RGB2GRAY)
    gray_frame_resized = cv2.resize(gray_frame, dsize=(84,84), interpolation=cv2.INTER_AREA)
    gray_frame_resized = gray_frame_resized[..., np.newaxis]
    gray_frame_resized_normalized = gray_frame_resized/np.max(gray_frame_resized)
    gray_frame_resized_normalized = gray_frame_resized_normalized.astype(np.float32)
    # changing image to fit pytorch's convention
    gray_frame_resized_normalized_to_tensor = ToTensor()(gray_frame_resized_normalized)
    # adding batch dimension
    gray_frame_resized_normalized_to_tensor = gray_frame_resized_normalized_to_tensor.unsqueeze(0)
    # from not on, this processed frame will be referred to as state
    state = gray_frame_resized_normalized_to_tensor
    return state

# def