{ "vector":{
      "load":" _mm512_load_ps",
      "store":" _mm512_store_ps"    ,
      "bcast":" _mm512_set1_ps"   ,
      "zero":"_mm512_setzero_ps",
      "BinaryOps":{
                "add": "_mm512_add_ps", 
                "sub": "_mm512_sub_ps", 
                "mul" :"_mm512_mul_ps", 
                "min": "_mm512_min_ps",
                "max": "_mm512_max_ps",
                "and": "_mm512_and_ps",
                "gt": "_mm512_cmp_ps",
                "lt": "_mm512_cmp_ps"
                },
      "TernaryOps":{
      "fmadd" :"_mm512_fmadd_ps", 
      "fmsub": "_mm512_fmsub_ps",
      "fmaddsub": "_mm512_fmaddsub_ps"
      },
      "dtype": "__m512",
      "bits": "512",
      "regs":"32"
    },
    "scalar":{
        "dtype":"float",
        "bits":"32"
    }
    }
