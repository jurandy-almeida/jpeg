// JPEG python data loader.
// Part of this implementation is modified from the exampel at
// https://aessedai101.github.io/c++/jpeg/jpg/dct/libjpeg/2014/07/10/extracting-jpeg-dct-coefficients.html


#include <Python.h>
#include "numpy/arrayobject.h"

#include <stdio.h>
#include <jpeglib.h>

static const char *filename = NULL;

static PyObject *JPEGError;

int parse_image(PyArrayObject ** arr)
{
    FILE *fp;
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    fp = fopen(filename, "rb");
    if (! fp) {
        printf("Could not open file: %s\n", filename);
        return -1;
    }

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);

    jpeg_stdio_src(&cinfo, fp);

    (void) jpeg_read_header(&cinfo, FALSE);

    jvirt_barray_ptr * src_coeff_arrays = jpeg_read_coefficients(&cinfo);

    int depth  = cinfo.num_components;
    int height = cinfo.comp_info[0].height_in_blocks;
    int width  = cinfo.comp_info[0].width_in_blocks;

    int blocksize  = DCTSIZE2 * sizeof(JCOEF);

    int stride_0 = height * width * blocksize;
    int stride_1 = width * blocksize;
    int stride_2 = blocksize;

    // Initialize arrays. 
    if (! (*arr)) {
        npy_intp dims[4];
        dims[0] = depth;
        dims[1] = height;
        dims[2] = width;
        dims[3] = DCTSIZE2;
        *arr = (PyArrayObject*) PyArray_ZEROS(4, dims, NPY_INT16, 0);
    }

    for (JDIMENSION d=0; d < depth; d++)
        for (JDIMENSION h=0; h < height; h++) {
            JBLOCKARRAY ptr = ((&cinfo)->mem->access_virt_barray)((j_common_ptr) &cinfo, src_coeff_arrays[d], h, (JDIMENSION) 1, FALSE);
            for (JDIMENSION w=0; w < width; w++)
                memcpy((*arr)->data + d * stride_0 + h * stride_1 + w * stride_2, ptr[0][w], blocksize);
        }

    jpeg_destroy_decompress(&cinfo);
    fclose(fp);

    return 0;
}

int decode_image(PyArrayObject ** arr)
{
    FILE *fp;
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    fp = fopen(filename, "rb");
    if (! fp) {
        printf("Could not open file: %s\n", filename);
        return -1;
    }

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    cinfo.dct_method = JDCT_FLOAT;

    jpeg_stdio_src(&cinfo, fp);

    (void) jpeg_read_header(&cinfo, TRUE);
    (void) jpeg_start_decompress(&cinfo);

    int depth  = cinfo.output_components;
    int height = cinfo.output_height;
    int width  = cinfo.output_width;

    int linesize = width * depth;

    // Initialize arrays. 
    if (! (*arr)) {
        npy_intp dims[3];
        dims[0] = height;
        dims[1] = width;
        dims[2] = depth;
        *arr = (PyArrayObject*) PyArray_ZEROS(3, dims, NPY_UINT8, 0);
    }

    JSAMPARRAY ptr = (*cinfo.mem->alloc_sarray)((j_common_ptr) &cinfo, JPOOL_IMAGE, linesize, 1);
    for (int h=0; h < height; h++) {
        (void) jpeg_read_scanlines(&cinfo, ptr, 1);
        memcpy((*arr)->data + h * linesize, ptr[0], linesize);
    }

    (void) jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(fp);

    return 0;
}

int encode_image(PyArrayObject * arr)
{
    FILE *fp;
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    fp = fopen(filename, "wb");
    if (! fp) {
        printf("Could not open file: %s\n", filename);
        return -1;
    }

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    cinfo.dct_method = JDCT_FLOAT;

    jpeg_stdio_dest(&cinfo, fp);

    npy_intp * dims = PyArray_DIMS(arr);
    cinfo.image_height = dims[0];
    cinfo.image_width  = dims[1];
    cinfo.input_components = dims[2];
    cinfo.in_color_space = JCS_RGB;

    int linesize = cinfo.image_width * cinfo.input_components;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 100, TRUE);
    cinfo.comp_info[0].h_samp_factor = 1;
    cinfo.comp_info[0].v_samp_factor = 1;
    cinfo.comp_info[1].h_samp_factor = 1;
    cinfo.comp_info[1].v_samp_factor = 1;
    cinfo.comp_info[2].h_samp_factor = 1;
    cinfo.comp_info[2].v_samp_factor = 1; 

    jpeg_start_compress(&cinfo, TRUE);

    JSAMPARRAY ptr = (*cinfo.mem->alloc_sarray)((j_common_ptr) &cinfo, JPOOL_IMAGE, linesize, 1);
    for (int h=0; cinfo.next_scanline < cinfo.image_height; h++) {
        memcpy(ptr[0], arr->data + h * linesize, linesize);
        (void) jpeg_write_scanlines(&cinfo, ptr, 1);
    }

    (void) jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(fp);

    return 0;
}

static PyObject *load(PyObject *self, PyObject *args)
{
    int decode;

    if (!PyArg_ParseTuple(args, "si", &filename, &decode)) return NULL;

    PyArrayObject *arr = NULL;

    if(!decode && parse_image(&arr) < 0) {
        printf("Parsing image failed.\n");

        Py_XDECREF(arr);
        return Py_None;
    }

    if(decode && decode_image(&arr) < 0) {
        printf("Decoding image failed.\n");

        Py_XDECREF(arr);
        return Py_None;
    }

    return arr;
}

static PyObject *save(PyObject *self, PyObject *args)
{
    PyObject *arg1 = NULL; // image data.
    PyArrayObject *arr = NULL;

    if (!PyArg_ParseTuple(args, "sO", &filename, &arg1)) return NULL;

    arr = (PyArrayObject*) PyArray_FROM_OTF(arg1, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    if (!arr) return NULL;

    if (encode_image(arr) < 0) 
        printf("Encoding image failed.\n");

    Py_DECREF(arr);
    Py_RETURN_NONE;
}

static PyMethodDef JPEGMethods[] = {
    {"load",  load, METH_VARARGS, "Load a JPEG image."},
    {"save",  save, METH_VARARGS, "Save a JPEG image."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};  

    
static struct PyModuleDef jpegmodule = {
    PyModuleDef_HEAD_INIT,
    "jpeg",  /* name of module */
    NULL,    /* module documentation, may be NULL */
    -1,      /* size of per-interpreter state of the module,
              or -1 if the module keeps state in global variables. */
    JPEGMethods
};
    
    
PyMODINIT_FUNC PyInit_jpeg(void)
{   
    PyObject *m;

    m = PyModule_Create(&jpegmodule);
    if (m == NULL)
        return NULL;
    
    /* IMPORTANT: this must be called */
    import_array();
                    
    JPEGError = PyErr_NewException("jpeg.error", NULL, NULL);
    Py_INCREF(JPEGError);
    PyModule_AddObject(m, "error", JPEGError);
    return m;
}       
        
        
int main(int argc, char *argv[])
{   
    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }   
        
    /* Add a built-in module, before Py_Initialize */
    PyImport_AppendInittab("jpeg", PyInit_jpeg);
    
    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(program);

    /* Initialize the Python interpreter.  Required. */
    Py_Initialize();
        
    PyMem_RawFree(program);
    return 0;
}       

