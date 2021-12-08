// JPEG python data loader.
// Part of this implementation is modified from the example at
// https://aessedai101.github.io/c++/jpeg/jpg/dct/libjpeg/2014/07/10/extracting-jpeg-dct-coefficients.html


#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#include <stdio.h>
#include <stdarg.h>

#include <jpeglib.h>


/* ------------------------ Global variables ----------------------------- */

static const char *membuffer = NULL;
static size_t memlen = 0;

static PyObject *JPEGError;


/*------------------------- Macros and constants  -------------------------*/

/* Given a pointer, FREE deallocates the space used by it.
 */
#define FREE(pointer) { if (pointer != NULL) free(pointer); pointer = NULL; }


/* -------------------- Local function prototypes ------------------------ */

static char *stolower(char *str);
static char *stoupper(char *str);
static int fatal_error(char * fmt, ...);


static int is_grayscale(j_decompress_ptr cinfo);
static int is_444(j_decompress_ptr cinfo);
static int is_422(j_decompress_ptr cinfo);
static int is_420(j_decompress_ptr cinfo);
static int is_440(j_decompress_ptr cinfo);
static int is_411(j_decompress_ptr cinfo);
static void set_444(j_compress_ptr cinfo);
static void set_422(j_compress_ptr cinfo);
static void set_420(j_compress_ptr cinfo);
static void set_440(j_compress_ptr cinfo);
static void set_411(j_compress_ptr cinfo);
static char *get_subsampling(j_decompress_ptr cinfo);
static void set_subsampling(j_compress_ptr cinfo,
                            char *subsampling);


static void transcode(j_decompress_ptr srcinfo,
                      int quality,
                      char *subsampling,
                      uint8_t **outbuffer);
static int parse_image_from_cinfo(j_decompress_ptr cinfo,
                                  PyArrayObject * arr[],
                                  int normalize,
                                  int quality,
                                  char *subsampling,
                                  int upsample,
                                  int stack);
static int parse_image_from_mem(PyArrayObject * arr[],
                                int normalize,
                                int quality,
                                char *subsampling,
                                int upsample,
                                int stack);
static int parse_image_from_stdio(PyArrayObject * arr[],
                                  int normalize,
                                  int quality,
                                  char *subsampling,
                                  int upsample,
                                  int stack);
static int parse_image(PyArrayObject * arr[],
                       int normalize,
                       int quality,
                       char *subsampling,
                       int upsample,
                       int stack,
                       int memory);


static void set_decompress_dct_method(j_decompress_ptr cinfo,
                                      char *dct_method);
static void set_out_color_space(j_decompress_ptr cinfo,
                                char *color_space);
static void set_scale(j_decompress_ptr cinfo,
                      float scale);
static void decode_image_from_cinfo(j_decompress_ptr cinfo,
                                    PyArrayObject ** arr,
                                    char *color_space,
                                    float scale,
                                    char *dct_method);
static int decode_image_from_mem(PyArrayObject ** arr,
                                 char *color_space,
                                 float scale,
                                 char *dct_method);
static int decode_image_from_stdio(PyArrayObject ** arr,
                                   char *color_space,
                                   float scale,
                                   char *dct_method);
static int decode_image(PyArrayObject ** arr,
                        char *color_space,
                        float scale,
                        char *dct_method,
                        int memory);


static void set_compress_dct_method(j_compress_ptr cinfo,
                                    char *dct_method);
static void set_in_color_space(j_compress_ptr cinfo,
                               char *color_space);
static void encode_image_to_cinfo(j_compress_ptr cinfo,
                                  PyArrayObject * arr,
                                  char *color_space,
                                  int quality,
                                  char *dct_method,
                                  char *subsampling,
                                  int optimize,
                                  int progressive);
static int encode_image_to_mem(PyArrayObject * arr,
                               char *color_space,
                               int quality,
                               char *dct_method,
                               char *subsampling,
                               int optimize,
                               int progressive);
static int encode_image_to_stdio(PyArrayObject * arr,
                                 char *color_space,
                                 int quality,
                                 char *dct_method,
                                 char *subsampling,
                                 int optimize,
                                 int progressive);
static int encode_image(PyArrayObject * arr,
                        char *color_space,
                        int quality,
                        char *dct_method,
                        char *subsampling,
                        int optimize,
                        int progressive);


/*----------------------------- Routines ----------------------------------*/

static char *stoupper(char *str)
{
    char *s = str;
    for(; *s; s++)
        if(('a' <= *s) && (*s <= 'z'))
            *s = 'A' + (*s - 'a');
    return str;
}


static char *stolower(char *str)
{
    char *s = str;
    for(; *s; s++)
        if(('A' <= *s) && (*s <= 'Z'))
            *s = 'a' + (*s - 'A');
    return str;
}


static int fatal_error(char * fmt, ...)
{
    va_list args;

    va_start(args, fmt);
    fprintf(stderr, "ERROR: ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);

    return -1;
}


static int is_grayscale(j_decompress_ptr cinfo)
{
    return cinfo->num_components == 1;
}


static int is_444(j_decompress_ptr cinfo)
{
    return (cinfo->comp_info[0].h_samp_factor == 1) &&
           (cinfo->comp_info[0].v_samp_factor == 1) &&
           (cinfo->comp_info[1].h_samp_factor == 1) &&
           (cinfo->comp_info[1].v_samp_factor == 1) &&
           (cinfo->comp_info[2].h_samp_factor == 1) &&
           (cinfo->comp_info[2].v_samp_factor == 1);
}


static int is_422(j_decompress_ptr cinfo)
{
    return (cinfo->comp_info[0].h_samp_factor == 2) &&
           (cinfo->comp_info[0].v_samp_factor == 1) &&
           (cinfo->comp_info[1].h_samp_factor == 1) &&
           (cinfo->comp_info[1].v_samp_factor == 1) &&
           (cinfo->comp_info[2].h_samp_factor == 1) &&
           (cinfo->comp_info[2].v_samp_factor == 1);
}


static int is_420(j_decompress_ptr cinfo)
{
    return (cinfo->comp_info[0].h_samp_factor == 2) &&
           (cinfo->comp_info[0].v_samp_factor == 2) &&
           (cinfo->comp_info[1].h_samp_factor == 1) &&
           (cinfo->comp_info[1].v_samp_factor == 1) &&
           (cinfo->comp_info[2].h_samp_factor == 1) &&
           (cinfo->comp_info[2].v_samp_factor == 1);
}


static int is_440(j_decompress_ptr cinfo)
{
    return (cinfo->comp_info[0].h_samp_factor == 1) &&
           (cinfo->comp_info[0].v_samp_factor == 2) &&
           (cinfo->comp_info[1].h_samp_factor == 1) &&
           (cinfo->comp_info[1].v_samp_factor == 1) &&
           (cinfo->comp_info[2].h_samp_factor == 1) &&
           (cinfo->comp_info[2].v_samp_factor == 1);
}


static int is_411(j_decompress_ptr cinfo)
{
    return (cinfo->comp_info[0].h_samp_factor == 4) &&
           (cinfo->comp_info[0].v_samp_factor == 1) &&
           (cinfo->comp_info[1].h_samp_factor == 1) &&
           (cinfo->comp_info[1].v_samp_factor == 1) &&
           (cinfo->comp_info[2].h_samp_factor == 1) &&
           (cinfo->comp_info[2].v_samp_factor == 1);
}


static void set_444(j_compress_ptr cinfo)
{
    cinfo->comp_info[0].h_samp_factor = 1; // Y
    cinfo->comp_info[0].v_samp_factor = 1;
    cinfo->comp_info[1].h_samp_factor = 1; // Cb
    cinfo->comp_info[1].v_samp_factor = 1;
    cinfo->comp_info[2].h_samp_factor = 1; // Cr
    cinfo->comp_info[2].v_samp_factor = 1;
}


static void set_422(j_compress_ptr cinfo)
{
    cinfo->comp_info[0].h_samp_factor = 2; // Y
    cinfo->comp_info[0].v_samp_factor = 1;
    cinfo->comp_info[1].h_samp_factor = 1; // Cb
    cinfo->comp_info[1].v_samp_factor = 1;
    cinfo->comp_info[2].h_samp_factor = 1; // Cr
    cinfo->comp_info[2].v_samp_factor = 1;
}


static void set_420(j_compress_ptr cinfo)
{
    cinfo->comp_info[0].h_samp_factor = 2; // Y
    cinfo->comp_info[0].v_samp_factor = 2;
    cinfo->comp_info[1].h_samp_factor = 1; // Cb
    cinfo->comp_info[1].v_samp_factor = 1;
    cinfo->comp_info[2].h_samp_factor = 1; // Cr
    cinfo->comp_info[2].v_samp_factor = 1;
}


static void set_440(j_compress_ptr cinfo)
{
    cinfo->comp_info[0].h_samp_factor = 1; // Y
    cinfo->comp_info[0].v_samp_factor = 2;
    cinfo->comp_info[1].h_samp_factor = 1; // Cb
    cinfo->comp_info[1].v_samp_factor = 1;
    cinfo->comp_info[2].h_samp_factor = 1; // Cr
    cinfo->comp_info[2].v_samp_factor = 1;
}


static void set_411(j_compress_ptr cinfo)
{
    cinfo->comp_info[0].h_samp_factor = 4; // Y
    cinfo->comp_info[0].v_samp_factor = 1;
    cinfo->comp_info[1].h_samp_factor = 1; // Cb
    cinfo->comp_info[1].v_samp_factor = 1;
    cinfo->comp_info[2].h_samp_factor = 1; // Cr
    cinfo->comp_info[2].v_samp_factor = 1;
}


static char *get_subsampling(j_decompress_ptr cinfo)
{
    if (is_444(cinfo))
        return "4:4:4";
    else if (is_422(cinfo))
        return "4:2:2";
    else if (is_420(cinfo))
        return "4:2:0";
    else if (is_440(cinfo))
        return "4:4:0";
    else if (is_411(cinfo))
        return "4:1:1";
    else
        return "unknown";
}


static void set_subsampling(j_compress_ptr cinfo,
                            char *subsampling)
{
    if (!strcmp(subsampling, "4:4:4"))
        set_444(cinfo);
    else if (!strcmp(subsampling, "4:2:2"))
        set_422(cinfo);
    else if (!strcmp(subsampling, "4:2:0"))
        set_420(cinfo);
    else if (!strcmp(subsampling, "4:4:0"))
        set_440(cinfo);
    else if (!strcmp(subsampling, "4:1:1"))
        set_411(cinfo);
}


static void transcode(j_decompress_ptr srcinfo,
                      int quality,
                      char *subsampling,
                      uint8_t **outbuffer)
{
    // start decompress
    (void) jpeg_start_decompress(srcinfo);

    // create the compression structure
    struct jpeg_compress_struct dstinfo;
    struct jpeg_error_mgr jerr;

    dstinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&dstinfo);

    size_t outlen = 0;
    jpeg_mem_dest(&dstinfo, outbuffer, &outlen);

    dstinfo.image_width = srcinfo->image_width;
    dstinfo.image_height = srcinfo->image_height;
    dstinfo.input_components = srcinfo->output_components;
    dstinfo.in_color_space = srcinfo->out_color_space;

    jpeg_set_defaults(&dstinfo);

    jpeg_set_quality(&dstinfo, quality, TRUE);
    set_subsampling(&dstinfo, subsampling);

    jpeg_start_compress(&dstinfo, TRUE);

    // transcode
    uint8_t *line_buffer = (uint8_t*) malloc(srcinfo->output_width *
                                             srcinfo->output_components);
    while (srcinfo->output_scanline < srcinfo->output_height) {
        jpeg_read_scanlines(srcinfo, &line_buffer, 1);
        (void) jpeg_write_scanlines(&dstinfo, &line_buffer, 1);
    }

    (void) jpeg_finish_compress(&dstinfo);
    jpeg_destroy_compress(&dstinfo);
    FREE(line_buffer);

    // re-create decompress
    jpeg_destroy_decompress(srcinfo);
    jpeg_create_decompress(srcinfo);

    jpeg_mem_src(srcinfo, *outbuffer, outlen);
    (void) jpeg_read_header(srcinfo, TRUE);
}


static int parse_image_from_cinfo(j_decompress_ptr cinfo,
                                  PyArrayObject * arr[],
                                  int normalize,
                                  int quality,
                                  char *subsampling,
                                  int upsample,
                                  int stack)
{
    (void) jpeg_read_header(cinfo, FALSE);

    uint8_t *buffer = NULL;
    if (!is_grayscale(cinfo) && strcmp(subsampling, "keep") &&
        strcmp(get_subsampling(cinfo), subsampling))
        transcode(cinfo, quality, subsampling, &buffer);
    else if (quality < 100)
        transcode(cinfo, quality, get_subsampling(cinfo), &buffer);

    jvirt_barray_ptr * src_coeff_arrays = jpeg_read_coefficients(cinfo);

    int depth  = cinfo->num_components;
    int height = cinfo->comp_info[0].height_in_blocks;
    int width  = cinfo->comp_info[0].width_in_blocks;

    int blocksize  = DCTSIZE2 * sizeof(JCOEF);

    // Initialize arrays. 
    if (! (*arr)) {
        if (stack) {
            npy_intp dims[4];
            dims[0] = depth;
            dims[1] = height;
            dims[2] = width;
            dims[3] = DCTSIZE2;
            arr[0] = (PyArrayObject*) PyArray_ZEROS(4, dims, NPY_INT16, 0);
        } else {
            for (JDIMENSION channel=0; channel < depth; channel++) {
                npy_intp dims[3];
                dims[0] = upsample ? height : cinfo->comp_info[channel].height_in_blocks;
                dims[1] = upsample ? width  : cinfo->comp_info[channel].width_in_blocks;
                dims[2] = DCTSIZE2;
                arr[channel] = (PyArrayObject*) PyArray_ZEROS(3, dims, NPY_INT16, 0);
            }
        }
    }

    JCOEF *src = (JCOEF*) malloc(blocksize);
    for (JDIMENSION channel=0; channel < depth; channel++) {
        int height_in_blocks = cinfo->comp_info[channel].height_in_blocks;
        int width_in_blocks = cinfo->comp_info[channel].width_in_blocks;
        int quant_idx = cinfo->comp_info[channel].quant_tbl_no;

        int v_scale_factor = height / height_in_blocks;
        int h_scale_factor = width / width_in_blocks;
        short unscale = 1;

        for (JDIMENSION b_y=0; b_y < height_in_blocks; b_y++) {
            JBLOCKARRAY ptr = (cinfo->mem->access_virt_barray)((j_common_ptr) cinfo,
                                                                src_coeff_arrays[channel],
                                                                b_y,
                                                                (JDIMENSION) 1,
                                                                FALSE);
            for (JDIMENSION b_x=0; b_x < width_in_blocks; b_x++) {
                for (JDIMENSION coeff=0; coeff < DCTSIZE2; coeff++) {
                    if (normalize)
                        unscale = cinfo->quant_tbl_ptrs[quant_idx]->quantval[coeff];
                    src[coeff] = ptr[0][b_x][coeff] * unscale;
                }

                if (upsample) {
                    for (JDIMENSION i=0; i < v_scale_factor; i++) {
                        int y = v_scale_factor * b_y + i;
                        for (JDIMENSION j=0; j < h_scale_factor; j++) {
                            int x = h_scale_factor * b_x + j;
                            int16_t *dest = stack ?
                                            (int16_t*) PyArray_GETPTR4(arr[0], channel, y, x, 0) :
                                            (int16_t*) PyArray_GETPTR3(arr[channel], y, x, 0);
                            memcpy(dest, src, blocksize);
                        }
                    }
                } else {
                    int16_t *dest = stack ?
                                    (int16_t*) PyArray_GETPTR4(arr[0], channel, b_y, b_x, 0) :
                                    (int16_t*) PyArray_GETPTR3(arr[channel], b_y, b_x, 0);
                    memcpy(dest, src, blocksize);
                }
            }
        }
    }
    FREE(src);
    FREE(buffer);

    return stack ? 1 : depth;
}


static int parse_image_from_mem(PyArrayObject * arr[],
                                int normalize,
                                int quality,
                                char *subsampling,
                                int upsample,
                                int stack)
{
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);

    jpeg_mem_src(&cinfo, (uint8_t*) membuffer, memlen);

    int ret = parse_image_from_cinfo(&cinfo,
                                     arr,
                                     normalize,
                                     quality,
                                     subsampling,
                                     upsample,
                                     stack);

    jpeg_destroy_decompress(&cinfo);

    return ret;
}


static int parse_image_from_stdio(PyArrayObject * arr[],
                                  int normalize,
                                  int quality,
                                  char *subsampling,
                                  int upsample,
                                  int stack)
{
    FILE *fp;
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    fp = fopen(membuffer, "rb");
    if (! fp)
        return fatal_error("Could not open file %s.", membuffer);

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);

    jpeg_stdio_src(&cinfo, fp);

    int ret = parse_image_from_cinfo(&cinfo,
                                     arr,
                                     normalize,
                                     quality,
                                     subsampling,
                                     upsample,
                                     stack);

    jpeg_destroy_decompress(&cinfo);
    fclose(fp);

    return ret;
}


static int parse_image(PyArrayObject * arr[],
                       int normalize,
                       int quality,
                       char *subsampling,
                       int upsample,
                       int stack,
                       int memory)
{
    return memory ?
           parse_image_from_mem(arr,
	                            normalize,
	                            quality,
	                            subsampling,
	                            upsample,
	                            stack) :
           parse_image_from_stdio(arr,
                                  normalize,
                                  quality,
                                  subsampling,
                                  upsample,
                                  stack);
}


static void set_decompress_dct_method(j_decompress_ptr cinfo,
                                      char *dct_method)
{
    if (!strcmp(dct_method, "islow"))
        cinfo->dct_method = JDCT_ISLOW;
    else if (!strcmp(dct_method, "ifast"))
        cinfo->dct_method = JDCT_IFAST;
    else if (!strcmp(dct_method, "float"))
        cinfo->dct_method = JDCT_FLOAT;
    else
        cinfo->dct_method = JDCT_DEFAULT;
}


static void set_out_color_space(j_decompress_ptr cinfo,
                                char *color_space)
{
    if (!strcmp(color_space, "keep"))
        cinfo->out_color_space = cinfo->jpeg_color_space;
    else if (!strcmp(color_space, "grayscale") &&
             (cinfo->jpeg_color_space == JCS_RGB ||
              cinfo->jpeg_color_space == JCS_YCbCr))
        cinfo->out_color_space = JCS_GRAYSCALE;
    else if (!strcmp(color_space, "rgb") &&
             (cinfo->jpeg_color_space == JCS_GRAYSCALE ||
              cinfo->jpeg_color_space == JCS_YCbCr))
        cinfo->out_color_space = JCS_RGB;
    else if (!strcmp(color_space, "cmyk") &&
             cinfo->jpeg_color_space == JCS_YCCK)
        cinfo->out_color_space = JCS_CMYK;
}


static void set_scale(j_decompress_ptr cinfo,
                      float scale)
{
    if (scale >= 0.125 && scale <= 2.0) {
        cinfo->scale_num = (unsigned int)(8.0 * scale);
        cinfo->scale_denom = 8;
    }
}


static void decode_image_from_cinfo(j_decompress_ptr cinfo,
                                    PyArrayObject ** arr,
                                    char *color_space,
                                    float scale,
                                    char *dct_method)
{
    (void) jpeg_read_header(cinfo, TRUE);

    set_decompress_dct_method(cinfo, dct_method);
    set_out_color_space(cinfo, color_space);
    set_scale(cinfo, scale);

    (void) jpeg_start_decompress(cinfo);

    int depth  = cinfo->output_components;
    int height = cinfo->output_height;
    int width  = cinfo->output_width;

    int linesize = width * depth;

    // Initialize arrays. 
    if (! (*arr)) {
        int ndim = 2 + (depth > 1);
        npy_intp dims[ndim];
        dims[0] = height;
        dims[1] = width;
        if (ndim > 2) dims[2] = depth;
        *arr = (PyArrayObject*) PyArray_ZEROS(ndim, dims, NPY_UINT8, 0);
    }

    JSAMPARRAY ptr = (*cinfo->mem->alloc_sarray)((j_common_ptr) cinfo, JPOOL_IMAGE, linesize, 1);
    for (int h=0; h < height; h++) {
        (void) jpeg_read_scanlines(cinfo, ptr, 1);
        memcpy(PyArray_DATA(*arr) + h * linesize, ptr[0], linesize);
    }

    (void) jpeg_finish_decompress(cinfo);
}


static int decode_image_from_mem(PyArrayObject ** arr,
                                 char *color_space,
                                 float scale,
                                 char *dct_method)
{
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);

    jpeg_mem_src(&cinfo, (uint8_t*) membuffer, memlen);

    decode_image_from_cinfo(&cinfo,
                            arr,
                            color_space,
                            scale,
                            dct_method);

    jpeg_destroy_decompress(&cinfo);

    return 0;
}


static int decode_image_from_stdio(PyArrayObject ** arr,
                                   char *color_space,
                                   float scale,
                                   char *dct_method)
{
    FILE *fp;
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    fp = fopen(membuffer, "rb");
    if (! fp)
        return fatal_error("Could not open file %s.", membuffer);

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);

    jpeg_stdio_src(&cinfo, fp);

    decode_image_from_cinfo(&cinfo,
                            arr,
                            color_space,
                            scale,
                            dct_method);

    jpeg_destroy_decompress(&cinfo);
    fclose(fp);

    return 0;
}


static int decode_image(PyArrayObject ** arr,
                        char *color_space,
                        float scale,
                        char *dct_method,
                        int memory)
{
    return memory ?
           decode_image_from_mem(arr,
	                             color_space,
	                             scale,
	                             dct_method) :
           decode_image_from_stdio(arr,
                                   color_space,
                                   scale,
                                   dct_method);
}


static void set_compress_dct_method(j_compress_ptr cinfo,
                                    char *dct_method)
{
    if (!strcmp(dct_method, "islow"))
        cinfo->dct_method = JDCT_ISLOW;
    else if (!strcmp(dct_method, "ifast"))
        cinfo->dct_method = JDCT_IFAST;
    else if (!strcmp(dct_method, "float"))
        cinfo->dct_method = JDCT_FLOAT;
    else
        cinfo->dct_method = JDCT_DEFAULT;
}


static void set_in_color_space(j_compress_ptr cinfo,
                               char *color_space)
{
    if (!strcmp(color_space, "grayscale"))
        cinfo->in_color_space = JCS_GRAYSCALE;
    else if (!strcmp(color_space, "rgb"))
        cinfo->in_color_space = JCS_RGB;
    else if (!strcmp(color_space, "ycbcr"))
        cinfo->in_color_space = JCS_YCbCr;
    else if (!strcmp(color_space, "cmyk"))
        cinfo->in_color_space = JCS_CMYK;
    else if (!strcmp(color_space, "ycck"))
        cinfo->in_color_space = JCS_YCCK;
    else if (!strcmp(color_space, "rgb565"))
        cinfo->in_color_space = JCS_RGB565;
    else
        cinfo->in_color_space = JCS_UNKNOWN;
}


static void encode_image_to_cinfo(j_compress_ptr cinfo,
                                  PyArrayObject * arr,
                                  char *color_space,
                                  int quality,
                                  char *dct_method,
                                  char *subsampling,
                                  int optimize,
                                  int progressive)
{
    int ndim = PyArray_NDIM(arr);
    npy_intp * dims = PyArray_DIMS(arr);
    cinfo->image_height = dims[0];
    cinfo->image_width  = dims[1];
    cinfo->input_components = (ndim > 2) ? dims[2] : 1;
    set_in_color_space(cinfo, (ndim > 2) ? color_space : "grayscale");

    int linesize = cinfo->image_width * cinfo->input_components;

    jpeg_set_defaults(cinfo);

    set_compress_dct_method(cinfo, dct_method);
    jpeg_set_quality(cinfo, quality, TRUE);
    set_subsampling(cinfo, subsampling);
    cinfo->optimize_coding = (optimize != 0);
    if (progressive) jpeg_simple_progression(cinfo);

    jpeg_start_compress(cinfo, TRUE);

    JSAMPARRAY ptr = (*cinfo->mem->alloc_sarray)((j_common_ptr) cinfo, JPOOL_IMAGE, linesize, 1);
    for (int h=0; cinfo->next_scanline < cinfo->image_height; h++) {
        memcpy(ptr[0], PyArray_DATA(arr) + h * linesize, linesize);
        (void) jpeg_write_scanlines(cinfo, ptr, 1);
    }

    (void) jpeg_finish_compress(cinfo);
}


static int encode_image_to_mem(PyArrayObject * arr,
                               char *color_space,
                               int quality,
                               char *dct_method,
                               char *subsampling,
                               int optimize,
                               int progressive)
{
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    jpeg_mem_dest(&cinfo, (uint8_t**) &membuffer, &memlen);

    encode_image_to_cinfo(&cinfo,
                          arr,
                          color_space,
                          quality,
                          dct_method,
                          subsampling,
                          optimize,
                          progressive);

    jpeg_destroy_compress(&cinfo);

    return memlen;
}


static int encode_image_to_stdio(PyArrayObject * arr,
                                 char *color_space,
                                 int quality,
                                 char *dct_method,
                                 char *subsampling,
                                 int optimize,
                                 int progressive)
{
    FILE *fp;
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    fp = fopen(membuffer, "wb");
    if (! fp)
        return fatal_error("Could not open file %s.", membuffer);

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    jpeg_stdio_dest(&cinfo, fp);

    encode_image_to_cinfo(&cinfo,
                          arr,
                          color_space,
                          quality,
                          dct_method,
                          subsampling,
                          optimize,
                          progressive);

    jpeg_destroy_compress(&cinfo);
    fclose(fp);

    return 0;
}


static int encode_image(PyArrayObject * arr,
                        char *color_space,
                        int quality,
                        char *dct_method,
                        char *subsampling,
                        int optimize,
                        int progressive)
{
    return membuffer == NULL ?
           encode_image_to_mem(arr,
                               color_space,
                               quality,
                               dct_method,
                               subsampling,
                               optimize,
                               progressive) :
           encode_image_to_stdio(arr,
                                 color_space,
                                 quality,
                                 dct_method,
                                 subsampling,
                                 optimize,
                                 progressive);
}


static PyObject *parse(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyArrayObject *arr[3] = {NULL, NULL, NULL};
    int normalize = TRUE;
    int quality = 100;
    char *subsampling = "keep";
    int upsample = TRUE;
    int stack = TRUE;
    int memory = FALSE;
    int len;

    static char *kwlist[] = {"input",
                             "normalize",
                             "quality",
                             "subsampling",
                             "upsample",
                             "stack",
                             "memory",
                             NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s#|pisppp", kwlist,
                                     &membuffer, &memlen,
                                     &normalize,
                                     &quality,
                                     &subsampling,
                                     &upsample,
                                     &stack,
                                     &memory))
        return NULL;

    if((len = parse_image(arr,
                          normalize,
                          quality,
                          stolower(subsampling),
                          upsample,
                          stack,
                          memory)) < 0) {
        printf("Parsing image failed.\n");

        for (int i=0; i < len; i++)
            Py_XDECREF(arr[i]);
        Py_RETURN_NONE;
    }

    if (len == 1)
        return (PyObject*) arr[0];

    PyObject *ret = PyTuple_New(len);
    for (int i=0; i < len; i++)
        PyTuple_SetItem(ret, i, (PyObject*) arr[i]);
    return ret;
}


static PyObject *load(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyArrayObject *arr = NULL;
    char *color_space = "keep";
    float scale = 1.0;
    char *dct_method = "islow";
    int memory = FALSE;

    static char *kwlist[] = {"input",
                             "color_space",
                             "scale",
                             "dct_method",
                             "memory",
                             NULL};
   if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s#|sfsp", kwlist,
                                    &membuffer, &memlen,
                                    &color_space,
                                    &scale,
                                    &dct_method,
                                    &memory))
        return NULL;

    if(decode_image(&arr,
                    stolower(color_space),
                    scale,
                    stolower(dct_method),
                    memory) < 0) {
        printf("Decoding image failed.\n");

        Py_XDECREF(arr);
        Py_RETURN_NONE;
    }

    return (PyObject*) arr;
}


static PyObject *save(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *arg1 = NULL; // image data.
    PyArrayObject *arr = NULL;
    char *color_space = "rgb";
    int quality = 75;
    char *dct_method = "islow";
    char *subsampling = "4:2:0";
    int optimize = FALSE;
    int progressive = FALSE;
    int len;

    static char *kwlist[] = {"arr",
                             "fname",
                             "color_space",
                             "quality",
                             "dct_method",
                             "subsampling",
                             "optimize",
                             "progressive",
                             NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|z#sisspp", kwlist,
                                     &arg1,
                                     &membuffer, &memlen,
                                     &color_space,
                                     &quality,
                                     &dct_method,
                                     &subsampling,
                                     &optimize,
                                     &progressive))
        return NULL;

    arr = (PyArrayObject*) PyArray_FROM_OTF(arg1,
                                            NPY_UINT8,
                                            NPY_ARRAY_IN_ARRAY);
    if (!arr) return NULL;

    if ((len = encode_image(arr,
                            stolower(color_space),
                            quality,
                            stolower(dct_method),
                            stolower(subsampling),
                            optimize,
                            progressive)) < 0)
        printf("Encoding image failed.\n");

	PyObject *ret = len ? Py_BuildValue("y#", membuffer, memlen) : Py_None;
	if (len) FREE(membuffer);

    Py_DECREF(arr);
    return ret;
}


static PyMethodDef JPEGMethods[] = {
    {"parse", (PyCFunction) parse, METH_VARARGS | METH_KEYWORDS, "Parse a JPEG image."},
    {"load" , (PyCFunction)  load, METH_VARARGS | METH_KEYWORDS, "Load a JPEG image." },
    {"save" , (PyCFunction)  save, METH_VARARGS | METH_KEYWORDS, "Save a JPEG image." },
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

