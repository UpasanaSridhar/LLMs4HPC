import sys
import json

kernel_name = sys.argv[1]
isa_spec_file = sys.argv[2]

def flatten_json(json_data):
    """
    Flattens a hierarchical JSON object into a single-level dictionary.
    
    :param json_data: JSON object (dictionary)
    :return: Flattened dictionary
    """
    def recursive_flatten(d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(recursive_flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    return recursive_flatten(json_data)


def read_and_flatten_json(json_file):
    """
    Reads a JSON file, flattens its content, and returns it as a dictionary.

    :param json_file: Path to the JSON file.
    :return: Flattened JSON dictionary.
    """
    with open(json_file, 'r') as file:
        json_data = json.load(file)
    
    isa = flatten_json(json_data)
        # set vector.width
    isa['vector.width'] = int(isa['vector.bits'])//int(isa['scalar.bits'])
    print(isa['vector.width'])
    return isa




#generate register definitions  
def define_registers(isa, M, N, unroll_factor):
    c_registers = ""
    N_registers = N//int(isa['vector.width'])

    assert(N_registers*M <= int(isa['vector.regs']));

    for i in range(M):
        for j in range(N//int(isa['vector.width'])):
            c_registers += f"{isa['vector.dtype']} c_{i}_{j};\n"
    
    #Assuming that the remaining registers are divided between A and B
    regs_remaining = int(isa['vector.regs']) - M*N_registers
    b_registers = ""
    for i in range(N_registers):
        b_registers += f"{isa['vector.dtype']} b_{i};\n"
    
    regs_remaining -= N_registers
    a_registers = ""
    a_registers += f"{isa['vector.dtype']} a_i;\n"
    return c_registers + b_registers + a_registers


#generate register initialization
def init_C_registers(isa, M, N, type="load"):
    c_init = ""
    N_registers = N//int(isa['vector.width'])
    for i in range(M):
        for j in range(N_registers):
            if type == "load":
                c_init += f"c_{i}_{j} = {isa['vector.load']}(C + {i}*{N} + {j*int(isa['vector.width'])});\n"
            elif type == "zero":
                c_init += f"c_{i}_{j} = {isa['vector.zero']}();\n"
            else:
                raise ValueError(f"Unknown type {type}")    
            
    return c_init

#generate register stores
def store_C_registers(isa, M, N):
    c_store = ""
    N_registers =  N//int(isa['vector.width'])
    for i in range(M):
        for j in range(N_registers):
            c_store += f"{isa['vector.store']}(C + {i}*{N} + {j*int(isa['vector.width'])}, c_{i}_{j});\n"
    return c_store

#generate outerpdt
def outerpdt(isa, M, N, unroll_factor,  compute_inst_key, num_a_regs, a_ptr_str, stride="1"):
    outerpdt = ""
    for i in range(M):
        #broadcast a_i
        outerpdt += f"a_i = {isa['vector.bcast']}(*({a_ptr_str} + {i}*{stride}));\n"
        for j in range( N//int(isa['vector.width'])):
            outerpdt += f"c_{i}_{j} = {isa[compute_inst_key]}(a_i, b_{j}, c_{i}_{j});\n"
    return outerpdt


#generate load_B_registers
def load_B_registers(isa, N, b_ptr_str):
    N_registers = N//int(isa['vector.width'])
    b_load = ""
    for i in range(N_registers):
        b_load += f"b_{i} = {isa['vector.load']}({b_ptr_str} + {i*int(isa['vector.width'])});\n"
    return b_load

#generate advance_ptrs
def advance_ptrs(a_ptr_str, b_ptr_str, M, N):
    return f"{a_ptr_str} += {M};\n{b_ptr_str} += {N};\n"

#generate unroll_snippet
def unroll_snippet(code):
    snippet = ""
    for i in range(unroll_factor):
        snippet += code
    return snippet


#generate gemm_kernel
def kernel_generator(isa, M, N, K, unroll_factor=4):

    outerpdt_code = f"""
            {load_B_registers(isa, N, "b_ptr")}
            {outerpdt(isa, M, N, unroll_factor, 'vector.TernaryOps.fmadd', 1, "a_ptr")}
            {advance_ptrs("a_ptr", "b_ptr", M, N)}
    """


    kernel = f"""
    // {kernel_name} kernel
    // M: {M}, N: {N}, unroll_factor: {unroll_factor}
    // K must be a multiple of {unroll_factor}
    // computes C += A(mxk) * B(kxn)
    // A is column major, B is row major, C is row major
    void {kernel_name}( const {isa['scalar.dtype']} *A, const {isa['scalar.dtype']} *B, {isa['scalar.dtype']} *C) {{
        //define register variables
        {define_registers(isa, M, N, unroll_factor)}

        //load values into C registers
        {init_C_registers(isa, M, N, type="zero")}

        //init vector pointers
        const {isa['scalar.dtype']} *a_ptr = A;
        const {isa['scalar.dtype']} *b_ptr = B;

        //outer product loop over K, unrolled by {unroll_factor}
        for(int k=0; k<{K}; k+={unroll_factor}) {{
            //load values into B registers
            {unroll_snippet(outerpdt_code)}
        }}

        //store C registers back to memory
        {store_C_registers(isa, M, N)}
    }}
    """
    return kernel

if __name__ == "__main__":
    # try:
    isa = read_and_flatten_json(isa_spec_file)
    print("Flattened ISA spec:")
    print(json.dumps(isa, indent=4))
    # except Exception as e:
    #     print(f"Error reading or flattening ISA spec: {e}")
    #     sys.exit(1)

    #accept M, N, K, unroll_factor as command line arguments

    M = int(sys.argv[4])
    N = int(sys.argv[5])
    K = int(sys.argv[6])
    unroll_factor = 1
    kernel_code = kernel_generator(isa, M, N, K, unroll_factor)
    print(kernel_code)
    # append the generated kernel code to a file
    kernel_file = sys.argv[3]
    with open(kernel_file, 'a') as file:
        file.write(kernel_code)
        print(f"Kernel code written to {kernel_file}")
