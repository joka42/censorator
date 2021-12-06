#!/bin/bash

ffmpeg -i testgif/artifacts.gif -vsync 0 out/artifact%d.png
