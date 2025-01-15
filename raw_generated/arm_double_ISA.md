    { "loads":" vld1q_f32",
      "stores":" vst1q_f32" ,
      "broadcast":" vdupq_n_f32",
      "Binary Vector Operations":{
                "add": "vaddq_f32", 
                "sub": "vsubq_f32", 
                "mul" :"vmulq_f32", 
                "min": "vminq_f32",
                "max": "vmaxq_f32",
                "and": "vandq_f32",
                "<": "vcmpq_f32",
                ">": "vcmpq_f32"
      },
      "Ternary Vector Operations":{
      "fused multipy-add" :"vmlaq_f32", 
      "fused multiply sub": "vmlsq_f32",
      "fused multiply-add/sub": "vmlasq_f32"
      },
      "Vector Data Type": "float32x4_t",
      "Register width": 128,
      "Elements per vector register": "4".
      "Num registers": 16
    }

