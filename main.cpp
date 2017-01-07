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

int main(int argc, char **argv){

  // Using boost library to parse commandline arguments.
  // Adding the possible options.
  options_description desc("Allowed Options");
  desc.add_options()
    ("help,h",  "Display help screen")
    ("output,o", value<string>()->default_value("output.jpg"), "Specify output file")
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
  
  // Load the image into memory
  uchar4 *h_image;
  size_t numRows, numCols;
  loadImageRGBA(input_file, &h_image, &numRows, &numCols);

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
    // Call the square_image function here
    // @param h_image numRows numCols strip_color  
  }
  else if(vm.count("filter")){
    string filter = vm["filter"].as<string>();
    // Call the apply_filter function here
    // @param h_image numRows numCols filter_name
  }

}
