from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


benchmark ={
    "tasks":{
     "loop reordering": {
        "context": """
Objective: Re-order the loops to have different output elements be updated in the innermost loops.
Idea: loop iterators that are present in the LHS of the assignment are iterating over different output elements.

## Example

Original Loops
```c
for(int i=0; i< M; i++)
{
    for (int j = 0; j< N; j++)
    {
        for(int p = 0; p < K; p++)
        {
            C[i*N+j] += A[p*M+i] * B[p*N+j];
        }
    }
}
```

Output(Reordered)
```c
for(int p = 0; p < K; p++)
{
    for(int i=0; i< M; i++)
    {
        for (int j = 0; j< N; j++)
        {

            C[i*N+j] += A[p*M+i] * B[p*N+j];
        }
    }
}
```
Analysis: The loop body is "C[i*N+j] += A[p*M+i] * B[p*N+j];" . The LHS of the assignment in contains the loop iterators i, and j. So the loop order must be changed to have i and j as the innermost loops. This leaves k as the outermost loop.

""",
        "prompts": {
   0: """ Reorder the following loops to have different output elements computed in the inner loops
```c
for(int i=0; i< M; i++)
{
    for (int j = 0; j< N; j++)
    {
        for(int k = 0; k < P; k++)
        {
            C[i*N+j] += A[k*M+i] * B[k*N+j];
        }
    }
}
```
""",
1:"""
Reorder the following loops to expose parallelism in the inner loops
```c
for(int i=0; i< M; i++)
{
    for (int j = 0; j< N; j++)
    {
        for(int k = 0; k < P; k++)
        {
            C[i*N+j] += A[i*P+k] * B[k*N+j];
        }
    }
}
```
""",


2:"""
Reorder the following loops to expose parallelism in the inner loops
```c
for(int k = 0; k < P; k++)
{
    for (int j = 0; j< N; j++)
    {
        for(int i=0; i< M; i++)
        {
            C[i*N+j] += A[i*P+k] * B[k*N+j];
        }
    }
}
```
""",
3:"""
Reorder the following loops to expose parallelism in the inner loops
```c
for(int i=0; i< M; i++)
{
    for (int j = 0; j< N; j++)
    {
        for(int k = 0; k < P; k++)
        {
            C[i*N+j] += (A[i*P+k] < B[k*N+j])? A[i*P+k] : B[k*N+j];
        }
    }
}
```
""",
4:"""
Reorder the following loops to expose parallelism in the inner loops
```c
for(int k = 0; k < P; k+=8)
{
    for (int j = 0; j< N; j++)
    {
        for(int i=0; i< M; i++)
        {
            for(int kk = 0; kk < 8; kk++)
            C[i*N+j] += (A[i*P+(k+kk)] < B[(k+kk)*N+j])? A[i*P+(k+kk)] : B[(k+kk)*N+j];
        }
    }
}
```
""",
5:"""
Reorder the following loops to expose parallelism in the inner loops
```c
for(int a = 0; a < M; a++)
{
    for (int b = 0; b < N; b++)
    {
        for(int c=0; c< P; c++)
        {
            C[a*N+b] += A[c*M + a] * B[c*N+b];
        }
    }
}
```
"""

}


},
"data layout permutation": {
    "context": """
    """,
    "prompts": {
}
},
"vectorization": {
    "context": """
    # Example

    You are an expert in vectorization and have been asked to vectorize the following code snippet using SIMD instructions.
    ## Context
    Target Architecture: Intel Haswell
    Data Type: float
    Vector ISA: AVX2

    Objective: Vectorize the following code snippet using SIMD instructions. 

    Assumptions: Vectorize the innermost loop of loopnests.
                 The loop extent is a multiple of the vector width. 
                 All data structures are aligned to cachelines.

    We use 3 examples of vectorization to inform our reasoning

    ### Example 1: Each iteration uses sequential, distinct elements of the inputs.
    Original Code: 
    ```c
    for(int i=0; i< N; i++)
    {
        A[i] = B[i] + C[i];
    }
    ```

    Vectorized Code:
    ```c
    for(int i=0; i< N; i+=8)
    {
        //Load 8 elements from B and C
        __m256 b = _mm256_load_ps(&B[i]);
        __m256 c = _mm256_load_ps(&C[i]);
        //Add the elements
        __m256 a = _mm256_add_ps(b, c);
        //Store the result
        _mm256_store_ps(&A[i], a);
    }
    ```

    Analysis: The loop body is "A[i] = B[i] + C[i];".
              Each iteration of the vectorized loop should do the work of 8 (vector length) iterations of the original loop.
              The loop can be vectorized by loading 8 elements from B and C, adding them and storing the result in A. 
              The loop is then incremented by the vector length.
    
    
    ### Example 2: In a loopnest, each inner iteration uses the same value of one input, and distinct elements of the other and stores in an output.
    Original Code:
    ```c
    for(int i=0; i< M; i++)
    {
        for(int j=0; j< N; j++)
        {
            C[i*N + j] = A[i] * B[j];
        }
    }
    ```

    Vectorized Code:
    ```c
    for(int i=0; i< M; i++)
    {
        //Broadcast A[i]
        __m256 a = _mm256_broadcast_ss(&A[i]);
        for(int j=0; j< N; j+=8)
        {
            //Load 8 elements from B
            __m256 b = _mm256_loadu_ps(&B[j]);
            //Multiply with A[i]
            __m256 c = _mm256_mul_ps(a, b);
            //Store the result
            _mm256_storeu_ps(&C[i*N + j], c);
        }
    }

    Analysis: The loop body is " C[i*N + j] = A[i] * B[j];".
              The same value of A is used across all iterations of the inner loop.
              So the elements of A can be broadcasted, outside the inner loop.
              The inner loop can be vectorized by loading 8 elements from B, multiplying with the broadcasted A, then storing in C.
              The loop is then incremented by vector length.
    
    ### Example 3: In a loop, each iteration uses the same value of A, and distinct elements of B and accumulates into an output C
    Original Code:
    ```c
    for(int j=0; j< N; j++)
    {
        c[j] += a[0] * B[j];
    }
    ```

    Vectorized Code:
    ```c
    //Broadcast a[0]
    __m256 a = _mm256_broadcast_ss(&a[0]);
    for(int j=0; j< N; j+=8)
    {
        __m256 b = _mm256_loadu_ps(&B[j]);
        __m256 c = _mm256_loadu_ps(&c[j]);
        c = _mm256_fmadd_ps(a, b, c);
        _mm256_storeu_ps(&c[j], c);
    }
    ```
    Analysis: The loop body is "c[j] += a[0] * B[j];".
              The same value of a is used across all iterations of the loop, so we broadcast a.
              The product of a and b is added to C. So we must load B and C. 
              Then, we can use the fused multiply-add instruction to perform the computation in the loop body.
              The loop is then incremented by the vector length.

    """,
    "prompts": {
    0: """ 
    Vectorize the following code snippet using SIMD instructions.
     Original Code: 
    ```c
    for(int i=0; i< N; i++)
    {
        A[i] = B[i] + C[i];
    }
    ```
 
    """,
    1: """
    Vectorize the following code snippet using SIMD instructions.
    Original Code: 
    ```c
    for(int i=0; i< M; i++)
    {
        for(int j=0; j< N; j++)
        {
            C[i*N + j] = std::max(A[i] ,B[j]);
        }
    }
    ```
    """,
    2: """
    Vectorize the following code snippet using SIMD instructions.
    Original Code: 
    ```c
    for(int i=0; i< N; i++)
    {
        A[i] = B[i] || C[i];
    }
    ```
    """,
    3: """
    Vectorize the following code snippet using SIMD instructions.
    Original Code: 
    ```c
        for(int j=0; j< N; j++)
        {
            c[j] -= a[0] * B[j];
        }
    ```
    """,
    4: """
    Vectorize the following code snippet using SIMD instructions.
    Original Code: 
    ```c
        for(int i=0; i< N; i++)
        {
            c[i] = 0;
            c[i] = (A[i] > 0)? A[i] : b[0]*A[i];
        }
    ```
    """,
    5: """
    Vectorize the following code snippet using SIMD instructions.
    Original Code: 
    ```c
        for(int i=0; i< N; i++)
        {
            c[j] += a[0] * B[j];
        }
    ```
    """

}
},


"vectorISA retrieval": {
    "context": """
    Objective: Identify the vector instructions/intrinsics to use to vectorize the given code.
    Idea: Assume all data structures are alinged to cachelines,
          All instructions are data type dependent, and there is a fixed number of elements in the vector register.
          We need to know instructions for 
          1) vector-loads and stores, 
          2) broadcasts, and 
          3) vector arithmetic operations - based on the computation in the loop body.
          The specific instructions depend on the target architecture.
    Assume the the loop extent is a multiple of the vector width.

    ## Example 1:
    Original Code: 
    ```c
    for(int i=0; i< N; i++)
    {
        A[i] = B[i] + C[i];
    }
    ```
    Target Architecture: Intel Haswell
    Data Type: float
    Vector ISA: AVX2

    Output:
    Vector ISA Instrinsics needed:
    register width: 256 bits
    elements per vector register: 8
    vector data type: __m256
    Loads: _mm256_load_ps
    Stores: _mm256_store_ps
    Arithmetic: _mm256_add_ps

    Assumption: 
    The loop extent is a multiple of the vector width.
    Data structures are aligned to cachelines.

    Analysis: The loop body is "A[i] = B[i] + C[i];". 
              Each iteration uses sequential, distinct elements of the inputs, B and C.
              If C was a scalar, we could use a broadcast instruction to load it.

    ## Example 2:
    Original Code: 
    ```c
    for(int i=0; i< M; i++)
    {
        float * a = A[i];
        float * c_row = C + i*N;
        for(int j=0; j< N; j++)
        {
            c_row[j] += (&a) * B[j];
        }
    }
    ```

    Target Architecture: Intel Haswell
    Data Type: float
    Vector ISA: AVX2

    Output:
    Vector ISA Instrinsics needed:
    register width: 256 bits
    elements per vector register: 8
    vector data type: __m256
    Loads: _mm256_load_ps, _mm256_broadcast_ss
    Stores: _mm256_store_ps
    Arithmetic: _mm256_fmadd_ps

    Assumption: 
    The loop extent is a multiple of the vector width.
    Data structures are aligned to cachelines.

    Analysis: 
    We consider the inner loop for vectorization.
    The loop body is "c_row[j] += a * B[j];". 
    The computation uses, a, B[j] and c_row[j] as inputs. The output is stored in c_row[j].
    Each iteration uses the same value of a, and distinct elements of B and C 
    The value of a is shared across all iterations of the inner loop. 
    So we can use a *broadcast* instruction to load it.
    B and C are loaded from memory using the *Packed load* instruction.
    We can use a *fused multiply-add* instruction to perform the computation in the loop body.
    The computed value is stored back to memory using the *Packed store* instruction.
    """,
    "prompts": {
    0: """ 
    Objective: Identify the vector instructions/intrinsics to use to vectorize the given code.
     Original Code: 
    ```c
    for(int i=0; i< N; i++)
    {
        A[i] = B[i] + C[i];
    }
    ```
    Target Architecture: AMD zen 4
    Data Type: float
    Vector ISA: AVX512

    List the vector intrinsics needed to vectorize the code.
    """,
    1: """
    Objective: Identify the vector instructions/intrinsics to use to vectorize the given code.
    Original Code: 
    ```c
    for(int i=0; i< M; i++)
    {
        for(int j=0; j< N; j++)
        {
            C[i*N + j] = A[i] * B[j];
        }
    }
    ```
    Target Architecture: AMD zen 4
    Data Type: float
    Vector ISA: AVX2
    """,
    2: """
    Objective: Identify the vector instructions/intrinsics to use to vectorize the given code.
    Original Code: 
    ```c
    for(int i=0; i< N; i++)
    {
        A[i] = B[i] + C[i];
    }
    ```
    Target Architecture: AMD zen 4
    Data Type: double
    Vector ISA: AVX2
    """,
    3: """
    Objective: Identify the vector instructions/intrinsics to use to vectorize the given code.
    Original Code: 
    ```c
        for(int j=0; j< N; j++)
        {
            c[j] += a[0] * B[j];
        }
    ```
    Target Architecture: AMD zen 4
    Data Type: float
    Vector ISA: AVX2
   
    """,
    4: """
    Objective: Identify the vector instructions/intrinsics to use to vectorize the given code.
    Original Code: 
    ```c
   
    for(int j=0; j< N; j++)
    {
        C[j] = a[0] * B[j];
    }

    ```
    Target Architecture: AMD zen 4
    Data Type: float
    Vector ISA: AVX2

    List the vector intrinsics , and vector datatype needed to vectorize the code.
    """

}
},


"unroll": {
    "context": """
    """,
    "prompts": {
}
},
"Loop invariant analysis": {
    "context": """
    """,
    "prompts": {
}
},
"in-regsiter updates": {
    "context": """
    """,
    "prompts": {
}
},
"register tiling": {
    "context": """
    """,
    "prompts": {
}
},
"kernel template": {
    "context": """
    Objective: 
    Write Macro that performs a store of a tile of 6x16 single-precision elements 
    from 12 vector registers to aligned memory. Use the AVX2 vector ISA.
  
    ## Example:
    ```c
    #define FLOAT_STORE_TILE_C(O, W_ob, C_ob)                  \
    {                                                      \
        _mm256_store_ps(O + (0 * C_ob), c00);               \
        _mm256_store_ps(O + (0 * C_ob) + FLOAT_SIMD, c01);  \
        _mm256_store_ps(O + (1 * C_ob), c10);               \
        _mm256_store_ps(O + (1 * C_ob + FLOAT_SIMD), c11);  \
        _mm256_store_ps(O + (2 * C_ob), c20);               \
        _mm256_store_ps(O + (2 * C_ob + FLOAT_SIMD), c21);  \
        _mm256_store_ps(O + (3 * C_ob), c30);               \
        _mm256_store_ps(O + (3 * C_ob + FLOAT_SIMD), c31);  \
        _mm256_store_ps(O + (4 * C_ob), c40);               \
        _mm256_store_ps(O + (4 * C_ob + FLOAT_SIMD), c41);  \
        _mm256_store_ps(O + (5 * C_ob), c50);              \
        _mm256_store_ps(O + (5 * C_ob + FLOAT_SIMD), c51); \
    }
    ```

    Analysis:
    - The 12 registers can be viewed as a 6x2 matrix. 
    - AVX2 has 256-bit wide registers and can store 8 single precision floating point elements.
        So, FLOAT_SIMD is 8.
    - Each colmun contains 8 elements.
    - The pointer O points to the start of the output tile, and is algined to a 64-byte boundary.
        So, we can use the aligned store instruction *_mm256_store_ps* to store 8 elements from the vector register to memory.
    
    Assumptions:
    - W_ob is 6 and C_ob is 16
    - Output should be stored in  row-major order. 
    - The 16 columns in the output tile are split across 2 vector registers.

    Register variable naming convention:
    - cij: The jth register of the ith row in the tile.
    """,
    "prompts": {
        0: """
        Objective: 
        Write Macro that performs a store of a tile of 6x8 double-precision elements. Use the AVX2 vector ISA. 
        """
    }
},
"vectorISA translation": {  
    "context": """
    You are an expert in vectorization and have been asked to identify the vector instructions and datatypes to use to vectorize the given functionality.

    ## Example:
    Objective: Identify the vector instructions and datatypes to use to vectorize the given functionality using the AVX2 vector ISA.
                The scalar datatype is float.
    
    Output:
    General Translation
    { "loads":" _mm256_load_ps",
      "stores":" _mm256_store_ps"    ,
      "broadcast":" _mm256_broadcast_ss"   ,
      "Binary Vector Operations":{
                "add": "_mm256_add_ps", 
                "sub": "_mm256_sub_ps", 
                "mul" :"_mm256_mul_ps", 
                "min": "_mm256_min_ps",
                "max": "_mm256_max_ps",
                "and": "_mm256_and_ps",
                "<": "_mm256_cmp_ps",
                ">": "_mm256_cmp_ps",
                },
      "Ternary Vector Operations":{
      "fused multipy-add" :"_mm256_fmadd_ps", 
      "fused multiply sub": "_mm256_fmsub_ps",
      "fused multiply-add/sub": "_mm256_fmaddsub_ps"
      },
      "Vector Data Type": "__m256",
      "Register width": 256,
      "Elements per vector register": "8"
    }
    """,
    "prompts": {
        0: """
        Objective:
        Construct a similar mapping of instructions and datatypes to use to the AVX2 vector ISA.
        The scalar datatype is double.
        If any instruction is not available int the ISA, mention it.

        Output:
        """,
        1: """
        Objective:
        Construct a similar mapping of instructions and datatypes to use to the ARMv8 NEON vector ISA.
        The scalar datatype is float.
        If any instruction is not available in the ISA, mention it.
        """,
        2: """
        Objective:
        Construct a similar mapping of instructions and datatypes to use to the ARMv8 NEON vector ISA.
        The scalar datatype is double.
        If any instruction is not available in the ISA, mention it.
        """,
        3:"""
        Objective:
        Construct a similar mapping of instructions and datatypes to use to the AVX2 vector ISA.
        The scalar datatype is half.
        If any instruction is not available in the ISA, mention it.
        """
}
},

}
}





def combine_with_context(context: str, prompt: str) -> str:
    """
    Combine a given context with a user-provided prompt to create an in-context learning prompt.

    Args:
        context (str): The existing context or examples to provide in the prompt.
        prompt (str): The new prompt to append to the context.

    Returns:
        str: The combined in-context learning prompt.
    """
    # Combine the context and prompt with a clear delimiter and the word 'challenge'
    in_context_prompt = f"{context}\n\n---\n\n## Challenge:\n{prompt}"
    return in_context_prompt

def run_code_llama(prompt: str):
    """
    Run the combined prompt using Code LLaMA 7B model.

    Args:
        prompt (str): The combined in-context learning prompt.

    Returns:
        str: The model's response.
    """
    # Load the Code LLaMA model and tokenizer
    model_name = "codellama/CodeLlama-7b-Instruct-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    # Ensure model uses GPU if available
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate a response with fast attention settings
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.9,
        use_cache=True,
        do_sample=True
    )

    # Decode and return the response
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    """Main function to demonstrate combining context and prompt and running Code LLaMA."""
    # Iterate over the benchmark prompts
    task = benchmark["tasks"]["vectorISA translation"]
    print(f"\nBenchmark Task: vectorISA translation\n")
    print(task["context"])
    for i in range(5):
        print(f"\nPrompt {i}:\n")
        print(task["prompts"][i])
        #combine with context
        combined_prompt = combine_with_context(task["context"], task["prompts"][i])
        print("\nCombined Prompt:\n")
        print(combined_prompt)

        #wait for user to press enter
        input("Press Enter to continue...")

        # #Run the prompt with Code LLaMA
        # print("\nCode LLaMA Response:\n")
        # response = run_code_llama(benchmark["prompts"][i])
        # print(response)
        # # Run the combined prompt with Code LLaMA
        # print("\nCode LLaMA Response:\n")
        # response = run_code_llama(combined_prompt)
        # print(response)




if __name__ == "__main__":
    main()
