
{
  "vector": {
    "load": "_mm256_load_pd",
    "store": "_mm256_store_pd",
    "bcast": "_mm256_set1_pd",
    "zero": "_mm256_setzero_pd",
    "BinaryOps": {
      "add": "_mm256_add_pd",
      "sub": "_mm256_sub_pd",
      "mul": "_mm256_mul_pd",
      "min": "_mm256_min_pd",
      "max": "_mm256_max_pd",
      "and": "_mm256_and_pd",
      "gt": "_mm256_cmp_pd",
      "lt": "_mm256_cmp_pd"
    },
    "TernaryOps": {
      "fmadd": "_mm256_fmadd_pd",
      "fmsub": "_mm256_fmsub_pd",
      "fmaddsub": "_mm256_fmaddsub_pd"
    },
    "dtype": "__m256d",
    "bits": "256",
    "regs": "8"
  },
  "scalar": {
    "dtype": "double",
    "bits": "64"
  }
}

