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
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

NORMALIZE = True
FLATTEN = True

class FontImageDataset(chainer.dataset.DatasetMixin):
	def __init__(self, datanum=10):
		pairs = []
		for _ in range(datanum):
			image_array, label = self.generate_image()
			pairs.append([image_array, label])
		self._pairs = pairs
						

	def __len__(self):
		return len(self._pairs)

	def generate_image(self):
		fonts = [
		    'font_files/Helvetica.ttf',
		    'font_files/BodoniSvtyTwoITCTT-Book.ttf'
		]
		label = random.randint(0,len(fonts)-1)
		fontFile = fonts[label]
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
		
		#im.save('image' + str(random.randint(0, 100)) + '.png')
		
		image_array = np.asarray(im)
		
		if NORMALIZE:
		    image_array = image_array / np.max(image_array)
		
		if FLATTEN:
			image_array = image_array.flatten()
		
		# TODO return chainer.datasets.tuple_dataset.TupleDataset
		# TODO make cnn model with chainer

		# type cast
		image_array = image_array.astype('float32')
		label = np.int32(label)

		return image_array, label

	def get_example(self, i):
		image_array, label = self._pairs[i][0], self._pairs[i][1]
		return image_array, label

