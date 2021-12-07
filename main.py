import argparse
import math
import os
import subprocess
import tempfile
import webp

import cv2
from filterpy.kalman import KalmanFilter
import numpy as np
from nudenet import NudeDetector
from PIL import Image


import frameextractor


PROGRESS_BAR_WIDTH = 20


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

    # we don't want the stamp to exceed the image frame, so we have to check if it does
    # there is probably a small error because of not adjusting the other values. It should generally
    # be in the single pixel range, so I don't care
    if offset_x < 0:
        offset_x = 0
    if offset_y < 0:
        offset_y = 0
    if image.shape[0] < offset_y + medium_size:
        medium_size = image.shape[0] - offset_y
    if image.shape[1] < offset_x + medium_size:
        medium_size = image.shape[1] - offset_x

    # resize the stamp to the medium size of the area
    s_img = cv2.resize(stamp, (medium_size, medium_size), interpolation=cv2.INTER_AREA)

    # get a matrix with alpha values between 0 and 1 from the orinal image with alpha-channel 3
    alpha_s = s_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        try:
            image[offset_y:offset_y+medium_size, offset_x:offset_x+medium_size, c] = (alpha_l * image[offset_y:offset_y+medium_size,
                                                                                                      offset_x:offset_x+medium_size, c] + alpha_s * s_img[:, :, c])
        except Exception as e:
            print(f"Stamping failed: {e}")
            print(f"offset: {offset_y}, {offset_x}")  # this way it si the same as numpy shape: (row, colums)
            print(f"medium_size: {medium_size}")
            print(f"c: {c}")
            print(
                f"shapes: {image.shape}, {alpha_s.shape}, {alpha_l.shape}, {image[offset_y:offset_y+medium_size, offset_x:offset_x+medium_size, c].shape}, {s_img[:, :, c].shape}")


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
        ".jpg") or f.lower().endswith(".png") or f.lower().endswith(".jpeg") or f.lower().endswith(".gif") or f.lower().endswith(".webp")]

    # print(images)
    return images


def calculate_centeroid(box):
    return [int((box[2] - box[0])/2.0), int(box[3] - box[1]/2.0)]


def filter_results(detection_results, results_of_interest):
    """
    box: [x_min, y_min, x_max, y_max]
    """
    filtered = []
    centeroids = []
    for result in detection_results:
        single_list = [obj for obj in result if obj.get("label") in results_of_interest]
        centeroids.append([{"label": obj.get("label"), "center": calculate_centeroid(obj.get("box"))}
                          for obj in single_list])
        filtered.append(single_list)
    # https://pypi.org/project/filterpy/
    # kf = KalmanFilter(dim_x=2, dim_z=1)
    print(filtered)
    print(centeroids)
    # split the objects by similarity

    # kalman filter for tracking

    return filtered


def main(args):
    images = images_in(args.input)

    out_dir = args.output if args.output else "./"
    if not out_dir.endswith("/"):
        out_dir += "/"

    if args.skip_existing:
        existing_images = [os.path.splitext(os.path.split(image)[1])[0] for image in images_in(out_dir)]
        original_image_count = len(images)
        images = [image for image in images if os.path.splitext(os.path.split(image)[1])[0] not in existing_images]

        print(f"Skipping {original_image_count-len(images)} images that are already processed.")

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
        name, extension = os.path.splitext(filename)

        # Animated images
        if extension.lower() == ".gif" or extension.lower() == ".webp":
            progress_bar = "[" + "░" * PROGRESS_BAR_WIDTH + "]"
            base_string = f"[ {str(int(index/len(images)*100)).rjust(3)}% ]  Processing file ({str(index + 1)}/{str(len(images))}) {filename}: " + "\t"
            print(base_string + progress_bar, end="\r", flush=True)

            tempdir = tempfile.mkdtemp()
            frame_duration = 40  # ms, defaults to 40 ms, 25 Hz
            if extension.lower() == ".gif":
                # ffmpeg seems to have fewer/no artifacts than the frameextractor.py that I found online
                # it had some black artifacts in a couple of frames that I could not fix
                os.system(f"ffmpeg -loglevel quiet -i {f} -vsync 0 {os.path.join(tempdir, name)}%d.png")
                frame_duration = frameextractor.analyseImage(f)['duration']
            elif extension.lower() == ".webp":
                frames = webp.load_images(f)
                with open(f, 'rb') as webp_file:
                    webp_data = webp.WebPData.from_buffer(webp_file.read())
                    dec = webp.WebPAnimDecoder.new(webp_data)
                    for arr, timestamp_ms in dec.frames():
                        frame_duration = timestamp_ms
                        break
                for index, frame in enumerate(frames):
                    frame.save(f'{os.path.join(tempdir, name)}_{index}.png', 'PNG')
            else:
                print("Wrong image format.")
                continue

            frame_files = images_in(tempdir)
            censored_frames = []
            detection_results = []

            # detect
            for index, frame_file in enumerate(frame_files):
                detection_result = detector.detect(frame_file)
                detection_results.append(detection_result)
                progress = int(len(detection_results)/len(frame_files)*PROGRESS_BAR_WIDTH)
                progress_bar = "[" + "█" * progress + "░" * \
                    (PROGRESS_BAR_WIDTH-progress) + "]" + f" (Frame {len(detection_results)}/{len(frame_files)})"
                print(base_string + progress_bar, end="\r", flush=True)

            detection_results = filter_results(detection_results, to_blur)

            for frame_file, detection_result in zip(frame_files, detection_results):
                image = cv2.imread(frame_file, flags=cv2.IMREAD_UNCHANGED)
                censored_frame = censor(image, boxes=detection_result, parts_to_blur=to_blur, with_stamp=args.stamped)

                # Convert image to pil image
                pil_frame = Image.fromarray(cv2.cvtColor(censored_frame, cv2.COLOR_BGR2RGB))
                censored_frames.append(pil_frame)

            censored_frames[0].save(os.path.join(out_dir, f'{name}.webp'), append_images=censored_frames[1:],
                                    save_all=True, optimize=False, duration=frame_duration, loop=0)
            print("")  # go to next line
            continue
        # Image is not animated
        else:
            print(f"[ {str(int(index/len(images)*100)).rjust(3)}% ]  Processing file ({str(index + 1)}/{str(len(images))}) {filename}")

            detection_result = detector.detect(f)
            image = cv2.imread(f)
            if image is None:
                print(f'Processing failed. Image "{filename}" may be corrupted...')
                processed_images -= 1
                continue
            censored_image = censor(image, boxes=detection_result, parts_to_blur=to_blur, with_stamp=args.stamped)
            censored_file_name = filename
            out_path = os.path.join(out_dir, censored_file_name)
            cv2.imwrite(out_path, censored_image)
    print(f"[ 100% ]  Processed {processed_images} of {len(images)}. Failed: {len(images) - processed_images}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output', required=False)
    parser.add_argument('-s', '--strict', action="store_true", default=False)
    parser.add_argument('-c', '--casual', action="store_true", default=False)
    parser.add_argument('--stamped', action="store_true", default=False)
    parser.add_argument('--skip_existing', action="store_true", default=False)
    args = parser.parse_args()
    print(args)
    main(args)
