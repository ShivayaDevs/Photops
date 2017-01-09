/* Usage:   ./photops input_file [option] [argument]
   Options: 
          --blur                    For blurring the image.
          --mirror orientation      For mirror image formation.
          --sqBlur                  For squaring and blurring the added strips.
          --square                  For adding coloured strips to image to make it square.
          --filter name             For applying filters.
          --amount value            For setting blur or square blur amount
          --color name              For specifying color used in the operation like squaring.
*/

#include <iostream>
#include <cuda_runtime.h>
#include <boost/program_options.hpp>
#include "include/load_save.h"
#include "include/square_ops.h"
#include "include/mirror_ops.h"
#include "include/blur_ops.h"
#include "include/filter_ops.h"

using namespace std;
using namespace boost::program_options;

uchar4* square_yash(uchar4* const d_in, size_t &numRows, size_t &numCols, uchar4 color);


size_t numRows, numCols;

uchar4* load_image_in_GPU(string filename)
{ // Load the image into main memory
  uchar4 *h_image, *d_in;
  loadImageRGBA(filename, &h_image, &numRows, &numCols);
  // Allocate memory to the GPU
  cudaMalloc((void **) &d_in, numRows * numCols * sizeof(uchar4));
  cudaMemcpy(d_in, h_image, numRows * numCols * sizeof(uchar4), cudaMemcpyHostToDevice);
  // No need to keep this image in RAM now.
  free(h_image);
  return d_in;
}

int main(int argc, char **argv){

  // Using boost library to parse commandline arguments.
  // Adding the possible options.
  options_description desc("Allowed Options");
  desc.add_options()
    ("help,h",  "Display help screen")
    ("output,o", value<string>()->default_value("images/output.jpg"), "Specify output file")
    ("blur,b",  "Blur the image")
    ("mirror,m", value<char>(), "Mirror the image")
    ("square,s", "Square the image by attaching strips")
    ("sqBlur,sb","Square Blur the image")
    ("filter,f", value<string>())
    ("amount,a", value<int>()->default_value(20), "Specifies amount of blur")
    ("color,c", value<string>()->default_value("white"), "Specifies the color to be used");

  // Generating a variable map for the options.
  variables_map vm;
  store(parse_command_line(argc, argv, desc), vm);
  notify(vm);

  if(vm.count("help")){
    cout<<desc<<"\n";
    exit(0);
  }

  if(argc < 2){
    cerr<<"Please specify the input file's name!\n";
    exit(1);
  }
  string input_file = string(argv[1]);
  string output_file = vm["output"].as<string>();
  
  uchar4 *d_in = load_image_in_GPU(input_file);

  // Performing the required operation
  if(vm.count("blur")){
    int amount = vm["amount"].as<int>();
    // Call the blur function here
    // @param h_image numRows numCols amount
  }
  else if(vm.count("mirror")){
    bool isVertical = ((vm["mirror"].as<char>() == 'v') ? true:false);
    // Call the mirror function here
    // @param h_image numRows numCols isVertical
  }
  else if(vm.count("sqBlur")){
    int amount = vm["amount"].as<int>();
    // Call the square blur function here
    // @param h_image numRows numCols amount 
  }
  else if(vm.count("square")){
    string color = vm["color"].as<string>();
      
    uchar4* h_out = square_yash(d_in, numRows, numCols, make_uchar4(255,255,255,255));
    saveImageRGBA(h_out, output_file, numRows, numCols);  



    // size_t n_numRows, n_numCols;
    // uchar4* h_out = square(d_in, numRows, numCols, n_numRows, n_numCols, make_uchar4(255,255,255,255));
    // saveImageRGBA(h_out, output_file, n_numRows,n_numCols);
    // Call the square_image function here
    // @param h_image numRows numCols strip_color  
  }
  else if(vm.count("filter")){
    string filter_name = vm["filter"].as<string>();
    uchar4* h_out = apply_filter(d_in, numRows, numCols, filter_name);
    saveImageRGBA(h_out, output_file, numRows,numCols);
    cout<<filter_name<<" filter applied. Output File: "<<output_file<<"\n";
  }
  cudaFree(d_in);
}
