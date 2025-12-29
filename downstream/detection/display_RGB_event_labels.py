
import cv2, os
import numpy as np
import random


RGB_folder = "/home/bhzhang/Documents/datasets/DSEC_detection_data/bdd100k_correctPixel/speed_direction/100K_top40MSE/images/train"
event_folder = ""
label_folder = "/home/bhzhang/Documents/datasets/DSEC_detection_data/bdd100k_correctPixel/speed_direction/100K_top40MSE/labels/train"
output_folder = ""
if output_folder:
    os.makedirs(output_folder, exist_ok=True)


event_prompt = ["convert_to_event_frame_with_fast_speed_going_ahead",
                "convert_to_event_frame_with_slow_speed_going_ahead",
                "convert_to_event_frame_with_fast_speed_going_backward",
                "convert_to_event_frame_with_slow_speed_going_backward"]
class_names = ["pedestrian", "rider", "car", "bus", "truck", "bicycle", "motorcycle", "train"]
class_colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]

def vis_yolo_data(annotations, im):
    # Draw bounding boxes on the image
    height, width, _ = im.shape
    for annotation in annotations:
        parts = annotation.strip().split()
        class_index = int(parts[0])
        center_x, center_y, bbox_width, bbox_height = map(float, parts[1:])

        # Convert normalized coordinates to actual pixel values
        x1 = int((center_x - bbox_width / 2) * width)
        y1 = int((center_y - bbox_height / 2) * height)
        x2 = int((center_x + bbox_width / 2) * width)
        y2 = int((center_y + bbox_height / 2) * height)

        # Draw rectangle and label
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(im, class_names[class_index], (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return im


for filename in os.listdir(RGB_folder):
    RGB_file = os.path.join(RGB_folder, filename)
    label_file = os.path.join(label_folder, filename.split(".")[0] + ".txt")
    event_files = []
    for prompt in event_prompt:
        event_files.append(os.path.join(event_folder, "{}---{}.{}".format(filename.split(".")[0], prompt, filename.split(".")[-1])))
    OK = True
    for event_file in event_files:
        if not os.path.exists(event_file):
            OK = False
    if not OK:
        continue

    RGB_img = cv2.imread(RGB_file)
    # image = cv2.cvtColor(RGB_img, cv2.COLOR_BGR2RGB)

    if label_folder:
        with open(label_file, 'r') as file:
            annotations = file.readlines()

    images = [vis_yolo_data(annotations, RGB_img)] if label_folder else [RGB_img]
    for event_file in event_files:
        event_img = cv2.imread(event_file)
        event_image = cv2.cvtColor(event_img, cv2.COLOR_BGR2RGB)
        if label_folder:
            images.append(vis_yolo_data(annotations, event_image))
        else:
            images.append(event_image)

    event_image_1 = np.concatenate((images[1], images[2]), 0)
    event_image_2 = np.concatenate((images[3], images[4]), 0)
    event_image = np.concatenate((event_image_1, event_image_2), 1)
    RGB_width, RGB_height = RGB_img.shape[1], RGB_img.shape[0]
    event_image = cv2.resize(event_image, (RGB_width, RGB_width))
    # cv2.resize(concated_image, (RGB_width, RGB_height))
    if output_folder:
        concated_image = np.concatenate((RGB_img, event_image), 0)
        cv2.imwrite(os.path.join(output_folder, filename), concated_image)
    else:
        cv2.imshow("event", cv2.resize(event_image, (RGB_height, RGB_height)))
        cv2.imshow("RGB", cv2.resize(RGB_img, (RGB_height//2, RGB_height//2)))
        cv2.waitKey(0)


    # height, width, _ = image.shape
    # #
    # # Draw bounding boxes on the image
    # for annotation in annotations:
    #     parts = annotation.strip().split()
    #     class_index = int(parts[0])
    #     center_x, center_y, bbox_width, bbox_height = map(float, parts[1:])
    #
    #     # Convert normalized coordinates to actual pixel values
    #     x1 = int((center_x - bbox_width / 2) * width)
    #     y1 = int((center_y - bbox_height / 2) * height)
    #     x2 = int((center_x + bbox_width / 2) * width)
    #     y2 = int((center_y + bbox_height / 2) * height)
    #
    #     # Draw rectangle and label
    #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     cv2.putText(image, class_names[class_index], (x1, y1 - 10),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



