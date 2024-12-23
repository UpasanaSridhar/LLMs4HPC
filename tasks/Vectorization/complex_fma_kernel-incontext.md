  """
    You are an expert in vectorization and have been asked to vectorize the following code snippet using SIMD instructions.
    ## Context
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
            __m256 b = _mm256_load_ps(&B[j]);
            //Multiply with A[i]
            __m256 c = _mm256_mul_ps(a, b);
            //Store the result
            _mm256_store_ps(&C[i*N + j], c);
        }
    }

    Analysis: The loop body is " C[i*N + j] = A[i] * B[j];".
              The same value of A is used across all iterations of the inner loop.
              So the elements of A can be broadcasted, *outside* the inner loop.
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
        __m256 b = _mm256_load_ps(&B[j]);
        __m256 c = _mm256_load_ps(&c[j]);
        c = _mm256_fmadd_ps(a, b, c);
        _mm256_store_ps(&c[j], c);
    }
    ```
    Analysis: The loop body is "c[j] += a[0] * B[j];".
              The same value of a is used across all iterations of the loop, so we broadcast a outside the inner loop.
              The product of a and b is added to C. So we must load B and C. 
              Then, we can use the fused multiply-add instruction to perform the computation in the loop body.
              The loop is then incremented by the vector length.

    

---

## Challenge:

    Vectorize the following code snippet using SIMD instructions.
    Original Code: 
    ```c
        for(int j=0; j< N; j++)
        {
            c[j] -= a[0] * B[j];
        }
    ```
"""