    You are an expert in vectorization and have been asked to identify the vector instructions and datatypes to use to vectorize the given functionality.

# Example:
## Task
    Objective: 
    Identify the vector instructions and datatypes to use to vectorize the given functionality using the AVX2 vector ISA.
    The scalar datatype is float.
    
    Output:
    General Translation - JSON

    { "load":" _mm256_load_ps",
      "store":" _mm256_store_ps"    ,
      "bcast":" _mm256_broadcast_ss"   ,
      "BinaryOps":{
                "add": "_mm256_add_ps", 
                "sub": "_mm256_sub_ps", 
                "mul" :"_mm256_mul_ps", 
                "min": "_mm256_min_ps",
                "max": "_mm256_max_ps",
                "and": "_mm256_and_ps",
                "gt": "_mm256_cmp_ps",
                "lt": "_mm256_cmp_ps"
                },
      "TernaryOps":{
      "fmadd" :"_mm256_fmadd_ps", 
      "fmsub": "_mm256_fmsub_ps",
      "fmaddsub": "_mm256_fmaddsub_ps"
      },
      "dtype": "__m256",
      "width": 256,
      "elements": "8".
      "regs":16
    }


---

# Challenge:

        Objective:
        Construct a similar mapping of instructions and datatypes to use to the ARMv8 NEON vector ISA.
        The scalar datatype is single-precison float.
        If any instruction is not available int the ISA, mention it.

