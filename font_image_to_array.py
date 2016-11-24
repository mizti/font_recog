#! /usr/bin/env python
# coding: utf-8
# coding=utf-8
# -*- coding: utf-8 -*-
# vim: fileencoding=utf-8

import sys
import random
import numpy as np
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

NORMALIZE = True

fonts = [
    'font_files/Helvetica.ttf',
    'font_files/BodoniSvtyTwoITCTT-Book.ttf'
]
i = random.randint(0,len(fonts)-1)
fontFile = fonts[i]
font = ImageFont.truetype(fontFile, 60)

# TODO separate train / test characters
# TODO choose random character
text = 'X'

w, h = 64, 64
text_w, text_h = font.getsize(text)
text_x, text_y = (w - text_w) * random.random(), (h - text_h) * random.random()

im = Image.new('L', (w, h), 255)
draw = ImageDraw.Draw(im)
draw.text((text_x, text_y), text, fill=(0), font=font)

#im.show()
im.save('image.png')

image_array = np.asarray(im)

if NORMALIZE:
    image_array = image_array / np.max(image_array)

# TODO flatten for non cnn model
# TODO return chainer.datasets.tuple_dataset.TupleDataset
# TODO make cnn model with chainer
