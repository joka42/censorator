import os
import argparse

from nudenet import NudeDetector

import cv2
import numpy as np


def pixelize(image, blocks=7):
    # divide the input image into NxN blocks
    pixelized_image = image.copy()
    (h, w) = image.shape[:2]

    if h < w:
        vertical_steps = blocks
        horizontal_steps = int(w/h*blocks)
    elif w < h:
        vertical_steps = int(h/w*blocks)
        horizontal_steps = blocks
    else:
        horizontal_steps = blocks
        vertical_steps = blocks

    xSteps = np.linspace(0, w, horizontal_steps + 1, dtype="int")
    ySteps = np.linspace(0, h, vertical_steps + 1, dtype="int")
    # loop over the blocks in both the x and y direction
    for i in range(1, len(ySteps)):
        for j in range(1, len(xSteps)):
            # compute the starting and ending (x, y)-coordinates
            # for the current block
            startX = xSteps[j - 1]
            startY = ySteps[i - 1]
            endX = xSteps[j]
            endY = ySteps[i]
            # extract the ROI using NumPy array slicing, compute the
            # mean of the ROI, and then draw a rectangle with the
            # mean RGB values over the ROI in the original image
            roi = pixelized_image[startY:endY, startX:endX]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(pixelized_image, (startX, startY), (endX, endY), (B, G, R), -1)
    # return the pixelated image
    return pixelized_image


def stamp(image, box):
    stamp = cv2.imread("resources/stamp.png", -1)
    # rectangle to square conversion with average sizes
    x_dim = abs(box[0]-box[2])
    y_dim = abs(box[1]-box[3])
    medium_size = int((x_dim + y_dim) / 2)

    # calculate the offset including medium error due to rectangle to square conversion
    offset_x = min(box[0], box[2]) + int((x_dim-medium_size)/2)
    offset_y = min(box[1], box[3]) + int((y_dim-medium_size)/2)

    s_img = cv2.resize(stamp, (medium_size, medium_size),
                       interpolation=cv2.INTER_AREA)

    alpha_s = s_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        try:
            image[offset_y:offset_y+medium_size, offset_x:offset_x+medium_size, c] = (alpha_s * s_img[:, :, c] +
                                                                                      alpha_l * image[offset_y:offset_y+medium_size, offset_x:offset_x+medium_size, c])
        except Exception:
            print("Stamping failed:")
            print(Exception)


def replace_in_image(into_image, from_image, box, shape="rectangle"):
    """
    box: [x_min, y_min, x_max, y_max]

    """
    if shape == "circle":
        # calculate average radius
        x_dim = abs(box[0]-box[2])
        y_dim = abs(box[1]-box[3])
        r = int(0.9*(x_dim + y_dim) / 4)
        center = (x_dim/2 + min([box[0], box[2]]), y_dim/2 + min([box[1], box[3]]))
        for y_row in range(2*r):
            y = r-y_row
            # x = 2r*sin(acos(y/(2r))) = 2r * sqrt(1-(y/(2r)^2))
            # https://socratic.org/questions/how-do-you-simplify-sin-arccos-x-1
            width = r*np.sqrt(abs(1-(y/r)*(y/r)))
            x_min = int(center[0] - width)
            x_max = int(center[0] + width)
            y_min = int(center[1] - y)
            y_max = y_min + 1
            into_image[y_min:y_max, x_min:x_max] = from_image[y_min:y_max, x_min:x_max]
    elif shape == "rectangle":
        into_image[box[1]:box[3], box[0]:box[2]
                   ] = from_image[box[1]:box[3], box[0]:box[2]]
    else:
        print("ERROR: unknown shape")


def censor(image, boxes, parts_to_blur=[], with_stamp=False):
    censored_image = image.copy()
    pixelized_image = pixelize(image, 50)

    if parts_to_blur:
        boxes = [i for i in boxes if i["label"] in parts_to_blur]
    else:
        boxes = [i for i in boxes]

    # put pussy at the end so that the stamp is not distorted
    genitalia = [i for i in boxes if i["label"] == "EXPOSED_GENITALIA_F"]
    boxes = [i for i in boxes if i["label"] != "EXPOSED_GENITALIA_F"]
    boxes += genitalia

    for item in boxes:
        box = item["box"]
        replace_in_image(censored_image, pixelized_image, box,
                         "circle" if item["label"] == "EXPOSED_GENITALIA_F" else "rectangle")
        if with_stamp:
            if item["label"] == "EXPOSED_GENITALIA_F":
                stamp(censored_image, box)

    return censored_image


def images_in(path):
    filenames = []
    if os.path.isdir(path):
        # print("Filenames: ")
        filenames = []
        for root, dirs, files in os.walk(path):
            for f in files:
                filenames.append(os.path.abspath(os.path.join(path, f)))
    elif os.path.isfile(path):
        filenames.append(path)
    else:
        print("File or directory not found.")

    images = [f for f in filenames if f.lower().endswith(
        ".jpg") or f.lower().endswith(".png") or f.lower().endswith(".jpeg")]

    print(images)
    return images


def main(args):
    images = images_in(args.input)
    out_dir = args.output if args.output else "./"
    if not out_dir.endswith("/"):
        out_dir += "/"
    detector = NudeDetector()
    exposed_parts = ["EXPOSED_ANUS", "EXPOSED_BUTTOCKS", "EXPOSED_BREAST_F", "EXPOSED_GENITALIA_F"]
    covered_parts = ["COVERED_BUTTOCKS", "COVERED_BREAST_F", "COVERED_GENITALIA_F"]

    if args.strict:
        to_blur = exposed_parts + covered_parts
    elif args.casual:
        to_blur = ["EXPOSED_BREAST_F", "EXPOSED_GENITALIA_F"]
    else:
        to_blur = exposed_parts

    processed_images = len(images)

    for index, f in enumerate(images):
        path, filename = os.path.split(f)
        print(f"[ {str(int(index/len(images)*100)).rjust(3)}% ]  Processing file ({str(index + 1)}/{str(len(images))}) {filename}")
        name, extension = os.path.splitext(filename)
        detection_result = detector.detect(f)
        image = cv2.imread(f)
        if image is None:
            print(f'Processing failed. Image "{filename}" may be corrupted...')
            processed_images -= 1
            continue
        censored_image = censor(image, boxes=detection_result, parts_to_blur=to_blur, with_stamp=args.stamped)
        censored_file_name = filename
        out_path = out_dir + censored_file_name
        cv2.imwrite(out_path, censored_image)
    print(f"[ 100% ]  Processed {processed_images} of {len(images)}. Failed: {len(images) - processed_images}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output', required=False)
    parser.add_argument('-s', '--strict', action="store_true", default=False)
    parser.add_argument('-c', '--casual', action="store_true", default=False)
    parser.add_argument('--stamped', action="store_true", default=False)
    args = parser.parse_args()
    print(args)
    main(args)
