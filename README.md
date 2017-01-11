# Photops
Photops is an image processing tool capable of applying filters or performing edit operations on images. It is implemented using CUDA
and therefore uses highly efficient parallel programming techniques to perform the operations using the GPU. The tool processes an HD image in less 
than 0.5ms on a decent GPU. This tool can be employed in bulk image processing or video processing applications to scale systems easily.  
## Usage
1. Build the program 
It is a command line tool and the source code can be compiled using `nvcc` (Nvidia C/C++ Compiler) and `g++`. 
```
  cd Photops
  make
```
The program can be executed using the command line operations.

2. Specify the input file
```
  ./photops path_to_input_file [option] [argument(s)]... 
```
3. Command line options

Option | Short name | Arguments | Function
-------|-----------|-----------|---------
 --output |-o|output_filename|For specifying output filename  
 --blur   |-b|            |For blurring the image.
 --mirror |-m|orientation |For mirror image formation.
 --sqBlur |-q|            |For squaring and blurring the added strips.
 --square |-s|            |For adding coloured strips to image to make it square.
 --filter |-f|name        |For applying filters.
 --amount |-a|value       |For setting blur or square blur amount
 --color  |-c|name        |For specifying color used in the operation like squaring.
----------|--|-----------|------

## Outputs 
 
 1. Input image ![Peacock](/images/peacock.jpg)
 2. Square operations
   * Squaring with coloured strips ![Squared](/images/square_out.jpg)
   * Square Blur ![SquareBlur](/images/sqblur_out.jpg)
 3. Filter
   * Vignette Filter ![Vignette](/images/vignette_out.jpg)
   * Greyscale Filter ![Greyscale](/images/greyscale_out.jpg)
 4. Mirror ![Mirror](/images/mirror_out.jpg)
 5. Blur ![Blur](/images/blur_output.jpg)
