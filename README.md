# Getting Started

This document briefly describes how to install and use the code.


## Environment

We tested this code in the following environment:
 - Linux
 - Python 3
 - Libjpeg

Similar environments (e.g. with OSX, Python 2) might work with small modification, but not tested.


## Description

This is a python wrapper that opens a JPEG image and extracts the DCT coefficients as a numpy array.


#### Install

 - Download Libjpeg (`curl https://www.ijg.org/files/jpegsrc.v9c.tar.gz | tar xvz`).
 - Go to Libjpeg home
 - `make clean`
 - `./configure --prefix=${LIBJPEG_INSTALL_PATH} --enable-shared`
 - `make`
 - `make install`
 - If needed, add `${LIBJPEG_INSTALL_PATH}/lib/` to `$LD_LIBRARY_PATH`.
 - Modify `setup.py` to use your Libjpeg path (`${LIBJPEG_INSTALL_PATH}`).
 - `./install.sh`


#### Usage

The python wrapper has three functions: `parse`, `load` and `save`.

The following call parses the JPEG raw data and returns the DCT coefficients in a numpy array.
```python
from jpeg import parse
parse([fname], normalize=True, quality=100, subsampling='keep', upsample=True, stack=True)
```
 - __fname__: path to a file from which the JPEG image is loaded.
 - __normalize__: if present and `True`, the DCT coefficients are normalized with quantification tables. If `False`, no normalization is performed. The default is `True`.
 - __quality__: if present and less than 100, transcodes the image to a different quality, on a scale from 1 (degraded image) to 100 (original image). The default is 100.
 - __subsampling__: if present, transcodes the image to a different subsampling. The options are:
   - `keep`: retain the original image setting. (_default_)
   - `4:4:4`, `4:2:2`, `4:2:0`, `4:4:0`, `4:1:1`: specific sampling values.
 - __upsample__: if present and `True`, the DCT coefficients for the chroma channels (Cb and Cr) are upsampled to have the dimensions of the luminance channel (Y). If `False`, no upsampling is performed. The default is `True`.
 - __stack__: if present and `True`, the numpy arrays for the luminance (Y) and chroma channels (Cb and Cr) are stacked along a new axis, returning a single numpy array. If `False`, a tuple with one numpy array for each channel is returned. The default is `True`.

For example,
```
parse('image.jpg')
```

The following call loads a JPEG image into a numpy array.
```python
from jpeg import load
load([fname], color_space='keep', scale=1.0, dct_method='islow')
```
 - __fname__: path to a file from which the JPEG image is loaded.
 - __color\_space__: if present, sets the output color space. This feature depends on the input color space, since not all possible color space transforms are currently implemented. The options are:
   - `keep`: retain the original image setting. (_default_)
   - `grayscale`: transform to a grayscale image.
   - `RGB`: convert to the RGB color space.
   - `YCbCr`: convert to the YCbCr color space.
   - `CMYK`: convert to the CMYK color space.
   - `YCCK`: convert to the YCCK color space.
   - `RGB565`: convert to the RGB565 color space.
 - __scale__: if present, scales the image size by a factor, ranging from 0.125 (smallest) to 2.0 (largest). The default is 1.0
 - __dct\_method__: if present, selects the algorithm used for the DCT step. Choices are:
   - `islow`: slow but accurate integer algorithm. (_default_)
   - `ifast`: faster, less accurate integer method.
   - `float`: floating-point method.

For example, 
```
load('image.jpg')
```

The following call saves a numpy array as a JPEG image.
```python
from jpeg import save
save([fname], [array], color_space='RGB', quality=75, dct_method='islow', subsampling='4:2:0', optimize=False, progressive=False)
```
 - __fname__: path to a file to which the JPEG image is saved.
 - __array__: a numpy array with shape `height` x `width` x 3, in case colored images; or `height` x `width` for grayscale images.
 - __color\_space__: if present, selects the input color space. The options are:
   - `grayscale`: a grayscale image.
   - `RGB`: an image in the RGB color space. (_default_)
   - `YCbCr`: an image in the YCbCr color space.
   - `CMYK`: an image in the CMYK color space.
   - `YCCK`: an image in the YCCK color space.
   - `RGB565`: an image in the RGB565 color space.
 - __quality__: if present, sets the image quality, on a scale from 1 (worst) to 100 (best). The default is 75.
 - __dct\_method__: if present, selects the algorithm used for the DCT step. Choices are:
   - `islow`: slow but accurate integer algorithm. (_default_)
   - `ifast`: faster, less accurate integer method.
   - `float`: floating-point method.
 - __subsampling__: if present, sets the subsampling for the encoder. The options are: `4:4:4`, `4:2:2`, `4:2:0`, `4:4:0`, and `4:1:1`. The default is `4:2:0`.
 - __optimize__: if present and `True`, indicates that the encoder should make an extra pass over the image in order to select optimal encoder settings. The default is `False`.
 - __progressive__: if present and `True`, indicates that this image should be stored as a progressive JPEG file. The default is `False`.

For example, 
```
imarr = numpy.random.randint(255, size=(100,100,3), dtype='uint8')
save('image.jpg', imarr)
```
