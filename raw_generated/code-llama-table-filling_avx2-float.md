{ "vector":{
      "load":" _mm256_load_ps",
      "store":" _mm256_store_ps"    ,
      "bcast":" _mm256_set1_ps"   ,
      "zero":"_mm256_setzero_ps",
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
      "bits": "256",
      "regs":"16"
    },
    "scalar":{
        "dtype":"float",
        "bits":"32"
    }
}

