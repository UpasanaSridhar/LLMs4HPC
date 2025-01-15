{ "load":" _mm256_loadu_pd",
      "store":" _mm256_storeu_pd"    ,
      "bcast":" _mm256_broadcast_sd"   ,
      "BinaryOps":{
                "add": "_mm256_add_pd", 
                "sub": "_mm256_sub_pd", 
                "mul" :"_mm256_mul_pd", 
                "min": "_mm256_min_pd",
                "max": "_mm256_max_pd",
                "and": "_mm256_and_pd",
                "gt": "_mm256_cmp_pd",
                "lt": "_mm256_cmp_pd"
                },
      "TernaryOps":{
      "fmadd" :"_mm256_fmadd_pd", 
      "fmsub": "_mm256_fmsub_pd",
      "fmaddsub": "_mm256_fmaddsub_pd"
      },
      "dtype": "__m256d",
      "width": "256",
      "elements": "4",
      "regs":"16"
}

