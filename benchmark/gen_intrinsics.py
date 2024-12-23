# File to generate all small kernels given the params in params.h

import os
import sys
import re
import json

#parse the params.h file to get the parameters
def extract_params(params_file):
    params = {}
    with open(params_file) as f:
        for line in f:
            if line.startswith("#define"):
                line = line.split()
                params[line[1]] = line[2]
    print(params)
    return params

# initialize the vector ISA for the target platform from the vector_isa.json file
def init_vector_isa():
    with open("vector_isa.json") as f:
        vector_isa = json.load(f)
    return vector_isa

# Datastructure to store the kernel names and types, read from kernels.json file
def init_kernels():
    with open("kernels.json") as f:
        kernels = json.load(f)
    return kernels

datatype = "FLOAT"

# Generate the small kernels for the target platform

#Macro preamble to overrule other small kernels (ifdef undef)
def gen_preamble(kernel_name):
    preamble = f"#ifdef {kernel_name}\n #undef {kernel_name}\n #endif\n"
    return preamble

# Declaration macros


#main
if __name__ == "__main__":
    extract_params(sys.argv[1])
    print(init_vector_isa())
    print(init_kernels())
