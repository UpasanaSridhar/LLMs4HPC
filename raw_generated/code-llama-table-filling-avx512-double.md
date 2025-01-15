
{ "vector":{
      "load":" _mm512_load_pd",
      "store":" _mm512_store_pd"    ,
      "bcast":" _mm512_set1_pd"   ,
      "zero":"_mm512_setzero_pd",
      "BinaryOps":{
                "add": "_mm512_add_pd", 
                "sub": "_mm512_sub_pd", 
                "mul" :"_mm512_mul_pd", 
                "min": "_mm512_min_pd",
                "max": "_mm512_max_pd",
                "and": "_mm512_and_pd",
                "gt": "_mm512_cmp_pd",
                "lt": "_mm512_cmp_pd"
                },
      "TernaryOps":{
      "fmadd" :"_mm512_fmadd_pd", 
      "fmsub": "_mm512_fmsub_pd",
      "fmaddsub": "_mm512_fmaddsub_pd"
      },
      "dtype": "__m512d",
      "bits": "512",
      "width": "8",
      "regs":"32"
    },
    "scalar":{
        "dtype":"double",
        "bits":"64"
    }
    }

