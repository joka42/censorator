#!/bin/python3

import argparse
import logging
import math
import os
import tempfile

import cv2
from nudenet import NudeDetector
import numpy as np
from PIL import Image
import webp

from util import *

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


def censor(image, boxes, parts_to_blur=[], with_stamp=False, black_bar=False):
    censored_image = image.copy()
    pixelized_image = pixelize(image, 25)

    if parts_to_blur:
        boxes = [i for i in boxes if i["label"] in parts_to_blur]
    else:
        boxes = [i for i in boxes]

    # split by category, so we can apply different methods of censoring
    genitalia = [i for i in boxes if i["label"] == "EXPOSED_GENITALIA_F"]
    boobs = [i for i in boxes if i["label"] == "EXPOSED_BREAST_F"]
    rest = [i for i in boxes if i not in genitalia and i not in boobs]

    logging.debug("Censoring %s genitalia, %s boobs and %s rest", len(genitalia), len(boobs), len(rest))

    # Order of censoring:
    # 1. plain pixelization
    # 2. black bars, if applicable
    # 3. stamped pussy
    # TODO if it is not stamped, then the black bars should go last
    for item in rest:
        box = item.get("box")
        box = resize_box(box, 0.9)
        replace_in_image(censored_image, pixelized_image, box, "rectangle")

    if black_bar and len(boobs) == 2:
        apply_black_bar(censored_image, boobs[0].get("box"), boobs[1].get("box"))
    else:
        for item in boobs:
            box = item.get("box")
            box = resize_box(box, 0.9)
            replace_in_image(censored_image, pixelized_image, box, "rectangle")

    for item in genitalia:
        box = item.get("box")
        replace_in_image(censored_image, pixelized_image, box, "circle")
        if with_stamp:
            stamp(censored_image, box)

    return censored_image


def calculate_centeroid(box):
    # box: [x_min, y_min, x_max, y_max]
    # return [x_center, y_center]
    return [int((box[2] + box[0])/2.0), int((box[3] + box[1])/2.0)]


def apply_black_bar(image, box_1, box_2, scaling=0.65):
    # Average box size offset based on box sizes
    offset = box_1[2] - box_1[0] + box_1[3] - box_1[1] + box_2[2] - box_2[0] + box_2[3] - box_2[1]
    offset /= 4
    # we measure from the center so only half
    offset /= 2
    # we don't want to cover all of the area
    offset *= scaling

    center_1 = np.array(calculate_centeroid(box_1))
    center_2 = np.array(calculate_centeroid(box_2))
    connection = center_2-center_1

    # assuming that the first box is on the left, we need to change the centers if that is not the case
    if connection[0] < 0:
        tmp = center_1
        center_1 = center_2
        center_2 = tmp
        connection = center_2 - center_1

    # calculate width
    dist = np.linalg.norm(connection)

    top_left = [center_1[0] - offset, center_1[1] - offset]
    top_right = [center_1[0] + dist + offset, center_1[1] - offset]
    bottom_right = [center_1[0] + dist + offset, center_1[1] + offset]
    bottom_left = [center_1[0] - offset, center_1[1] + offset]
    pts = np.array([top_left, top_right, bottom_right, bottom_left])

    # rotation matrix
    theta = np.arccos(connection[0] / dist)
    # the angle is not signed, so check rotation direction and set sign accordingly
    if center_2[1] < center_1[1]:
        theta = -theta
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    pts_shifted = [pt - center_1 for pt in pts]
    pts_shifted_rotated = [R.dot(pt) for pt in pts_shifted]
    # pts_shifted_rotated = [pt for pt in pts_shifted]
    pts_rotated = [pt + center_1 for pt in pts_shifted_rotated]
    pts_rotated = np.array(pts_rotated, np.int32)

    logging.debug("Black bar poly: %s", np.asarray(pts_rotated))
    cv2.fillPoly(image, [pts_rotated], color=(0, 0, 0))
    # add anti aliasing to smooth the edges
    cv2.polylines(image, [pts_rotated], True, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)


def check_similarity(x_1, x_2, max_distance_x, max_distance_y):
    dist_x = abs(int(x_1[0] - x_2[0]))
    dist_y = abs(int(x_1[1] - x_2[1]))
    logging.debug("similarity check: class diff: %s", abs(x_1[5] - x_2[5]))
    return abs(x_1[5] - x_2[5]) < 0.5 and dist_x < max_distance_x and dist_y < max_distance_y, dist_x/max_distance_x + dist_y/max_distance_y


def update(x_pred, x_meas):
    factor_meas = pow(x_meas[4], 2)
    factor_pred = pow(x_pred[4], 2)
    new = (x_meas * factor_meas + x_pred * factor_pred) / (factor_meas + factor_pred)
    new[2] = x_meas[2]
    new[3] = x_meas[3]
    new[4] = max(x_pred[4], x_meas[4])  # if measurement was found, the certainty is reset to 1
    return new


def update_box(centeroid, box):
    box_center = calculate_centeroid(box)
    diff_x = int(centeroid[0] - box_center[0])
    diff_y = int(centeroid[1] - box_center[1])
    new_box = [box[0] + diff_x, box[1] + diff_y, box[2] + diff_x, box[3] + diff_y]
    return new_box


def centeroid_from(detection_result):
    center = calculate_centeroid(detection_result.get("box"))
    classification = BodyPart.get(detection_result.get("label"))
    # all classifications must, by design, be in the dict!
    assert classification is not None
    x = np.array([center[0], center[1], 0, 0, 1, classification])
    return x


def resize_box(box, factor):
    # box: [x_min, y_min, x_max, y_max]
    width = box[2] - box[0]
    height = box[3] - box[1]
    diff_width = (1-factor) * width
    diff_height = (1-factor) * height
    new_box = [int(box[0] + diff_width/2), int(box[1] + diff_height/2),
               int(box[2] - diff_width/2), int(box[3] - diff_height/2)]
    return new_box


def processImage(path, out_path=None):
    '''
    Iterate the GIF, extracting each frame.
    '''
    frames = []
    mode = analyzeImage(path)['mode']
    print(mode)
    im = Image.open(path)

    i = 0
    p = im.getpalette()
    last_frame = im.convert('RGBA')

    try:
        while True:
            print(i)
            # print(f"saving {path} ({mode}) frame {i}, {im.size}")

            '''
            If the GIF uses local colour tables, each frame will have its own palette.
            If not, we need to apply the global palette to the new frame.
            '''
            if not im.getpalette():
                try:
                    im.putpalette(p)
                except ValueError:
                    # If only one image is found, only the one image is put into the frame list
                    new_frame = Image.new('RGBA', im.size)
                    new_frame.paste(im, (0, 0), im.convert('RGBA'))
                    frames.append(new_frame)
                    break

            '''
            Is this file a "partial"-mode GIF where frames update a region of a different size to the entire image?
            If so, we need to construct the new frame by pasting it on top of the preceding frames.
            '''
            if mode == 'partial':
                new_frame = Image.new('RGBA', im.size, color=None)
                new_frame.paste(last_frame)
                new_frame.paste(im, (0, 0), im.convert('RGBA'))
            elif mode == "full":
                new_frame = im

            if out_path:
                new_frame.save(os.path.join(out_path, os.path.basename(path).split(".")[:-1][0]) + f'_{i}.png', 'PNG')

            i += 1
            last_frame = new_frame
            frames.append(new_frame)
            im.seek(i)
    except EOFError:
        pass

    return frames


def get_total_frames(images):
    total_frames = 0
    for image in images:
        im = Image.open(image)
        i = 0
        try:
            while True:
                i += 1
                im.seek(i)
        except EOFError:
            total_frames += i

    return total_frames


def find_match(result, result_pool):
    match = -1
    min_similarity = math.inf
    box = result.get("box")
    scaling_factor = 1  # a little more than half
    max_distance_x = int(scaling_factor*(box[2] - box[0]))
    max_distance_y = int(scaling_factor*(box[3] - box[1]))
    centeroid = result.get("centeroid")
    for index, possible_match in enumerate(result_pool):
        possible_match_centeroid = possible_match.get("centeroid")
        is_similar, similarity = check_similarity(centeroid, possible_match_centeroid, max_distance_x, max_distance_y)
        # check if possible match
        if not is_similar:
            logging.debug("not similar")
            continue
        # find closest match
        if similarity < min_similarity:
            min_similarity = similarity
            match = index
    return match


def filter_results(detection_results, results_of_interest, with_velocity=False):
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
    a = 1  #
    # d : prediction confidence decline per frame
    d = 0.9
    # visbility_threshold : threshold when an object is removed
    visibility_threshold = 0.1
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
    # A = np.ndarray(shape=(6, 6), buffer=A, dtype=float)
    # ========================
    # Data extraction
    # TODO: calculate velocity and add it to the vector
    filtered = []
    for frame_results in detection_results:
        filtered_frame_results = [detection_result for detection_result in frame_results
                                  if detection_result.get("label") in results_of_interest]
        filtered.append(filtered_frame_results)

    # for frame, frame_result in enumerate(filtered):
    #     for index, result in enumerate(frame_result):
    #         filtered[frame][index]['centeroid'] = centeroid_from(result)
    for frame_result in filtered:
        for result in frame_result:
            result['centeroid'] = centeroid_from(result)

    # calculate velocity between frames
    if with_velocity:
        for current_frame, next_frame in zip(filtered[:-1], filtered[1:]):
            for result in current_frame:
                match = find_match(result, next_frame)
                if match < 0:
                    continue
                next_center = next_frame[match].get("centeroid")
                current = result.get("centeroid")
                current[2] = next_center[0] - current[0]  # x velocity
                current[3] = next_center[1] - current[1]  # y velocity

    logging.debug(str([len(x) for x in filtered]))

    # we start predicting from the first measurement and filter the following measurement
    # so the last frame can be skipped, because a new prediction is not necessary
    for frame, _ in enumerate(filtered[:-1]):
        logging.debug("Frame: %s", frame + 1)

        for result in filtered[frame]:
            # calculate prediction
            if logging.root.level == logging.DEBUG:
                for index, print_result in enumerate(filtered[frame]):
                    logging.debug("Before: Current - Index %s: %s", index, print_result)
                for index, print_result in enumerate(filtered[frame + 1]):
                    logging.debug("Before: Next - Index %s: %s", index, print_result)
            # logging.debug("Before: Result Index %s: %s", result_index, frame_result)
            x = result.get("centeroid")
            x_pred = A.dot(x)

            match = find_match(result, filtered[frame + 1])
            logging.debug("Found Match: %s", match)

            # combine the measurement and the prediction
            if match >= 0:
                logging.debug("Comb: Match %s", match)
                logging.debug("Match: %s", filtered[frame + 1][match])
                x_meas = filtered[frame + 1][match].get("centeroid")
                x_comb = update(x_pred, x_meas)
                filtered[frame + 1][match]["centeroid"] = x_comb
                filtered[frame + 1][match]["box"] = update_box(x_comb, filtered[frame + 1][match].get("box"))
                if logging.root.level == logging.DEBUG:
                    for index, print_result in enumerate(filtered[frame]):
                        logging.debug("Current - Index %s: %s", index, print_result)
                    for index, print_result in enumerate(filtered[frame + 1]):
                        logging.debug("Next - Index %s: %s", index, print_result)
                continue

            # if the prediction is too old do not continue to track it
            if x_pred[4] < visibility_threshold:
                logging.debug("Skipping, low visibility")
                continue

            # if there is no match found, use prediction only
            logging.debug("Add prediction result")
            filtered[frame + 1].append(result)
            filtered[frame + 1][-1]["box"] = update_box(x_pred, filtered[frame + 1][-1].get("box"))
            filtered[frame + 1][-1]["centeroid"] = x_pred

    return filtered


def main(args):
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    images = images_in(args.input)

    total_frames = get_total_frames(images)
    logging.debug("Total frame count: %s", total_frames)

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

    for index, file in enumerate(images):
        _, filename = os.path.split(file)
        name, extension = os.path.splitext(filename)

        # Animated images
        if extension.lower() == ".gif" or extension.lower() == ".webp":
            progress_bar = "[" + "░" * PROGRESS_BAR_WIDTH + "]"
            base_string = f"[ {str(int(index/len(images)*100)).rjust(3)}% ]  Processing file ({str(index + 1)}/{str(len(images))}) {filename}: " + "\t"
            print(base_string + progress_bar, end="\r", flush=True)

            tempdir = tempfile.mkdtemp()
            frame_duration = 0
            if extension.lower() == ".gif":
                # ffmpeg seems to have fewer/no artifacts than the frameextractor.py that I found online
                # it had some black artifacts in a couple of frames that I could not fix
                ffmpeg_loglevel = "-loglevel quiet" if not args.debug else ""
                os.system(f"ffmpeg {ffmpeg_loglevel} -i {file} -vsync 0 {os.path.join(tempdir, name)}%d.png")
                analysis_result = analyzeImage(file)
                frame_duration = analysis_result.get('duration')
                logging.debug("Analysis: %s", analysis_result)
                logging.debug("Frame duration: %s", frame_duration)
            elif extension.lower() == ".webp":
                frames = webp.load_images(file)
                with open(file, 'rb') as webp_file:
                    webp_data = webp.WebPData.from_buffer(webp_file.read())
                    dec = webp.WebPAnimDecoder.new(webp_data)
                    for _, timestamp_ms in dec.frames():
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
                progress_bar = "[" + "█" * progress + "░" * (PROGRESS_BAR_WIDTH-progress) + "]"
                frame_counter = f" (Frame {len(detection_results)}/{len(frame_files)})"
                print(base_string + progress_bar + frame_counter, end="\r", flush=True)
            print("")  # go to next line

            if args.filter:
                for i in range(2):
                    if not i % 2:
                        detection_results = filter_results(detection_results, to_blur, with_velocity=True)
                        continue
                    detection_results.reverse()
                    detection_results = filter_results(detection_results, to_blur)
                    detection_results.reverse()

            boobs_in_frames = []
            for frame_result in detection_results:
                boobs_in_frames.append([item for item in frame_result if item.get("label") == "EXPOSED_BREAST_F"])
            black_bar = all([len(boobs_in_frame) == 2 or len(boobs_in_frame)
                            == 0 for boobs_in_frame in boobs_in_frames])
            logging.debug("Black bars active: %s", black_bar)
            for frame_file, frame_result in zip(frame_files, detection_results):
                image = cv2.imread(frame_file, flags=cv2.IMREAD_UNCHANGED)
                censored_frame = censor(image, boxes=frame_result, parts_to_blur=to_blur,
                                        with_stamp=args.stamped, black_bar=black_bar)

                if args.debug and args.filter:
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

            detection_result = detector.detect(file)
            image = cv2.imread(file)
            if image is None:
                print(f'Processing failed. Image "{filename}" may be corrupted...')
                processed_images -= 1
                continue
            censored_image = censor(image, boxes=detection_result, parts_to_blur=to_blur,
                                    with_stamp=args.stamped, black_bar=True)
            censored_file_name = filename
            out_path = os.path.join(out_dir, censored_file_name)
            cv2.imwrite(out_path, censored_image)
    failed_string = f"Failed: {len(images) - processed_images}." if len(images) - processed_images > 0 else ""
    print(f"[ 100% ]  Processed {processed_images} of {len(images)} files. {failed_string}")


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
