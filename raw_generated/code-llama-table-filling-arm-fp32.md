  { "vector":{
    "load":" vld1q_f32",
      "store":" vst1q_f32"    ,
      "bcast":" vdupq_n_f32"   ,
      "BinaryOps":{
                "add": "vaddq_f32", 
                "sub": "vsubq_f32", 
                "mul" :"vmulq_f32", 
                "min": "vminq_f32",
                "max": "vmaxq_f32",
                "and": "vandq_u32",
                "gt": "vcmpq_f32",
                "lt": "vcmpq_f32"
                },
      "TernaryOps":{
      "fmadd" :"vfmaq_f32", 
      "fmsub": "vfmsq_f32",
      "fmaddsub": "vfmssq_f32"
      },
      "dtype": "float32x4_t",
      "bits": 128,
      "width": "4",
      "regs":16
    }
    "scalar":{
        "dtype":"float32_t",
        "bits":"32"
    }
    }
---
{
"vector": {
  "load": "vld1q_f32",
  "store": "vst1q_f32",
  "bcast": "vdupq_n_f32",
  "BinaryOps": {
    "add": "vaddq_f32",
    "sub": "vsubq_f32",
    "mul": "vmulq_f32",
    "min": "vminq_f32",
    "max": "vmaxq_f32",
    "and": "vandq_u32",
    "gt": "vcmpq_f32",
    "lt": "vcmpq_f32"
  },
  "TernaryOps": {
    "fmadd": "vfmaq_f32",
    "fmsub": "vfmsq_f32",
    "fmaddsub": "vfmssq_f32"
  },
  "dtype": "float32x4_t",
  "bits": 128,
  "width": "4",
  "regs": 16
},
"scalar": {
  "dtype": "float32_t",
  "bits": "32"
}
}

