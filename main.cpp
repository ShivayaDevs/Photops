#include <iostream>
#include <cuda_runtime.h>
#include "include/load_save.h"
#include "include/square_ops.h"
#include "include/mirror_ops.h"
#include "include/blur_ops.h"
#include "include/filter_ops.h"

using namespace std;

int main(int argc, char **argv){

  string input_file;
  string output_file = "image_out.jpg";

  if(argc < 2){
    cerr<<"Usage: ./photops input_file_path [operation] [arguments] ...";
  }

  input_file = string(argv[1]);


  // Command line argument parsing will be added later. 
  // Let's start with the business logic first.
  
}
