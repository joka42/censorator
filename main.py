import argparse
import logging
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
INFINITY = 9223372036854775807

# bodyParts: "enum" so they can be stored in the np.ndarray
BodyPart = {"EXPOSED_ANUS": 0,
            "EXPOSED_BUTTOCKS": 1,
            "EXPOSED_BREAST_F": 2,
            "EXPOSED_GENITALIA_F": 3,
            "COVERED_BUTTOCKS": 4,
            "COVERED_BREAST_F": 5,
            "COVERED_GENITALIA_F": 6}


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
    for color in range(0, 3):
        image[offset_y:offset_y+medium_size, offset_x:offset_x+medium_size, color] = \
            (alpha_l * image[offset_y:offset_y+medium_size, offset_x:offset_x +
             medium_size, color] + alpha_s * s_img[:, :, color])


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
        for _, _, files in os.walk(path):
            for file in files:
                filenames.append(os.path.abspath(os.path.join(path, file)))
    elif os.path.isfile(path):
        filenames.append(path)
    else:
        print("File or directory not found.")

    images = [f for f in filenames if f.lower().endswith(
        ".jpg") or f.lower().endswith(".png") or f.lower().endswith(".jpeg") or f.lower().endswith(".gif") or f.lower().endswith(".webp")]

    # print(images)
    return images


def calculate_centeroid(box):
    # box: [x_min, y_min, x_max, y_max]
    # return [x_center, y_center]
    return [int((box[2] + box[0])/2.0), int((box[3] + box[1])/2.0)]


def check_similarity(x_1, x_2):
    MAX_DISTANCE = 50  # pixel
    dist_x = abs(int(x_1[0] - x_2[0]))
    dist_y = abs(int(x_1[1] - x_2[1]))
    logging.debug("similarity check: class diff: %s", abs(x_1[5] - x_2[5]))
    return abs(x_1[5] - x_2[5]) < 0.5 and dist_x < MAX_DISTANCE and dist_y < MAX_DISTANCE, dist_x + dist_y


def update(x_pred, x_meas):
    return x_meas * 0.7 + x_pred * 0.3


def update_box(centeroid, box):
    box_center = calculate_centeroid(box)
    diff_x = int(centeroid[0] - box_center[0])
    diff_y = int(centeroid[1] - box_center[1])
    box[0] += diff_x
    box[2] += diff_x
    box[1] += diff_y
    box[3] += diff_y
    return box


def centeroid_from(detection_result):
    center = calculate_centeroid(detection_result.get("box"))
    classification = BodyPart.get(detection_result.get("label"))
    # all classifications must, by design, be in the dict!
    assert classification is not None
    x = np.array([center[0], center[1], 0, 0, 1, classification])
    return x


def filter_results(detection_results, results_of_interest):
    # other ideas
    # https://pypi.org/project/filterpy/
    # kf = KalmanFilter(dim_x=2, dim_z=1)
    # split the objects by similarity
    # kalman filter for tracking
    """
    box: [x_min, y_min, x_max, y_max]
    """
    # ====== Parameters ======
    # a : acceleration
    a = 0.3  # we assume, that the center movement declines
    # d : visibility decline per frame
    d = 0.8
    # visbility_threshold : threshold when an object is removed
    visibility_threshold = 0.4
    # t : time in frame is set to 1, because it does not change and units don't matter
    t = 1
    # A : prediction matrix
    A = np.array([[1, 0, t, 0, 0, 0],
                  [0, 1, 0, t, 0, 0],
                  [0, 0, a, 0, 0, 0],
                  [0, 0, 0, a, 0, 0],
                  [0, 0, 0, 0, d, 0],
                  [0, 0, 0, 0, 0, 1]
                  ])
    #A = np.ndarray(shape=(6, 6), buffer=A, dtype=float)
    # ========================
    # Data extraction
    # TODO: calculate velocity and add it to the vector
    filtered = []

    for frame_results in detection_results:
        filtered_frame_results = [detection_result for detection_result in frame_results
                                  if detection_result.get("label") in results_of_interest]
        filtered.append(filtered_frame_results)

    for frame_result in filtered:
        for item in frame_result:
            item['centeroid'] = centeroid_from(item)

    logging.debug(str([len(x) for x in filtered]))

    # we start predicting from the first measurement and filter the following measurement
    # so the last frame can be skipped, because a new prediction is not necessary
    for frame, frame_result in enumerate(filtered[:-1]):
        logging.debug("Frame: %s", frame + 1)
        for result in frame_result:
            # calculate prediction
            x = result.get("centeroid")
            x_pred = A.dot(x)

            min_similarity = math.inf
            match = -1
            for index, next_frame_results in enumerate(filtered[frame + 1]):
                x_meas = next_frame_results.get("centeroid")
                is_similar, similarity = check_similarity(x_pred, x_meas)
                # check if possible match
                if not is_similar:
                    logging.debug("not similar")
                    continue
                # find closest match
                if similarity < min_similarity:
                    min_similarity = similarity
                    match = index

            # combine the measurement and the prediction
            if match >= 0:
                logging.debug("Comb: Match %s", match)
                x_meas = filtered[frame + 1][match].get("centeroid")
                x_comb = update(x_pred, x_meas)
                filtered[frame + 1][match]["centeroid"] = x_comb
                filtered[frame + 1][match]["box"] = update_box(x_comb, filtered[frame + 1][match].get("box"))
                continue

            # if the prediction is too old do not continue to track it
            if x_pred[4] < visibility_threshold:
                logging.debug("Skipping, low visibility")
                continue

            # if there is no match found, use prediction only
            logging.debug("add new")
            filtered[frame + 1].append(filtered[frame][match])
            filtered[frame + 1][match]["box"] = update_box(x_pred, filtered[frame + 1][match].get("box"))
            filtered[frame + 1][match]["centeroid"] = x_pred

    logging.debug(str([len(x) for x in filtered]))

    return filtered


def main(args):
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

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
        _, filename = os.path.split(f)
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
                for frame_index, frame in enumerate(frames):
                    frame.save(f'{os.path.join(tempdir, name)}_{frame_index}.png', 'PNG')
            else:
                print("Wrong image format.")
                continue

            frame_files = images_in(tempdir)
            censored_frames = []
            detection_results = []

            # detect
            for frame_file in frame_files:
                detection_result = detector.detect(frame_file)
                detection_results.append(detection_result)
                progress = int(len(detection_results)/len(frame_files)*PROGRESS_BAR_WIDTH)
                progress_bar = "[" + "█" * progress + "░" * \
                    (PROGRESS_BAR_WIDTH-progress) + "]" + f" (Frame {len(detection_results)}/{len(frame_files)})"
                print(base_string + progress_bar, end="\r", flush=True)
            print("")  # go to next line

            if args.filter:
                detection_results = filter_results(detection_results, to_blur)

            for frame_file, frame_result in zip(frame_files, detection_results):
                image = cv2.imread(frame_file, flags=cv2.IMREAD_UNCHANGED)
                censored_frame = censor(image, boxes=frame_result, parts_to_blur=to_blur, with_stamp=args.stamped)

                if args.debug:
                    for result in frame_result:
                        center = result.get("centeroid")
                        censored_frame = cv2.circle(censored_frame, (int(center[0]), int(center[1])), radius=5,
                                                    color=(0, 0, 255), thickness=-1)
                # Convert image to pil image
                pil_frame = Image.fromarray(cv2.cvtColor(censored_frame, cv2.COLOR_BGR2RGB))
                censored_frames.append(pil_frame)

            censored_frames[0].save(os.path.join(out_dir, f'{name}.webp'), append_images=censored_frames[1:],
                                    save_all=True, optimize=False, duration=frame_duration, loop=0)

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
    parser.add_argument('--filter', action="store_true", default=False)
    parser.add_argument('--debug', action="store_true", default=False)
    args = parser.parse_args()
    print(args)
    main(args)
