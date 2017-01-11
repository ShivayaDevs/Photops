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

int hex_to_int(string hexa)
{
  int i;
  stringstream s(hexa);
  s>>std::hex>>i;
  return i;
}

uchar4 hex_to_uchar4_color(string& color)
{
  int r = hex_to_int(color.substr(0, 2));
  int g = hex_to_int(color.substr(2, 2));
  int b = hex_to_int(color.substr(4, 2));
  return make_uchar4(r, g, b, 255);
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
    ("sqBlur,q","Square Blur the image")
    ("filter,f", value<string>())
    ("amount,a", value<int>()->default_value(21), "Specifies amount of blur")
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
  uchar4 *h_out = NULL;

  // Performing the required operation
  if(vm.count("blur")){
    int amount = vm["amount"].as<int>();
    if(amount % 2 == 0)
      amount++;
    h_out = blur_ops(d_in, numRows, numCols, amount);                      
  }
  else if(vm.count("mirror")){
    bool isVertical = ((vm["mirror"].as<char>() == 'v') ? true:false);
    h_out = mirror_ops(d_in, numRows, numCols, isVertical);
  }
  else if(vm.count("sqBlur")){
    int amount = vm["amount"].as<int>();
    if(amount % 2 == 0)
      amount++;
    h_out = square_blur(d_in, numRows, numCols, amount); 
  }
  else if(vm.count("square")){
    string color_hex = vm["color"].as<string>();
    uchar4 color = hex_to_uchar4_color(color_hex);
    h_out = square_image(d_in, numRows, numCols, color); 
  }
  else if(vm.count("filter")){
    string filter_name = vm["filter"].as<string>();
    h_out = apply_filter(d_in, numRows, numCols, filter_name);
  }

  cudaFree(d_in);
  if(h_out != NULL)
    saveImageRGBA(h_out, output_file, numRows, numCols); 
  
}
