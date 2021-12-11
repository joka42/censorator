#!/bin/python3

import os
from PIL import Image


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


def analyzeImage(path):
    '''
    Pre-process pass over the image to determine the mode (full or additive).
    Necessary as assessing single frames isn't reliable. Need to know the mode
    before processing all frames.
    '''
    im = Image.open(path)
    results = {
        'size': im.size,
        'mode': 'full',
        'duration': im.info['duration'] if "duration" in im.info.keys() and im.info.get("duration") > 0 else 80,
    }
    try:
        while True:
            if im.tile:
                tile = im.tile[0]
                update_region = tile[1]
                update_region_dimensions = update_region[2:]
                if update_region_dimensions != im.size:
                    results['mode'] = 'partial'
                    break
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    return results
