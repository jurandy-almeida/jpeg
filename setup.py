from distutils.core import setup, Extension
import numpy as np

jpeg_utils_module = Extension('jpeg',
		sources = ['jpeg_data_loader.c'],
		include_dirs=[np.get_include(), './libjpeg/include/'],
		extra_compile_args=['-O3'],
		extra_link_args=['-ljpeg', '-L./libjpeg/lib/']
)

setup ( name = 'jpeg',
	version = '0.1',
	description = 'Utils for handling JPEG images.',
	ext_modules = [ jpeg_utils_module ]
)
