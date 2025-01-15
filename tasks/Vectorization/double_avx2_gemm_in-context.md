  """
You are an expert in vectorization and have been asked to vectorize the following code snippet using SIMD instructions.


# Objective:
 Vectorize the following code snippet using SIMD instructions. 



# Context
    Vector ISA: AVX2
    Scalar dataype: double

# Vectorization principles
1. Vectorize the innermost loop of loopnests.
2. Vector Width is the number of bits in a vector register divided by the size of 1 scalar
3. The loop extent is a multiple of the vector width. 
4. All data structures are aligned to cachelines.

## Examples of Vectorization
We use 3 examples of vectorization to inform our reasoning
### Example 1: Each iteration uses sequential, distinct elements of the inputs.
#### Original Code: 
    ```c
    for(int i=0; i< N; i++)
    {
        A[i] = B[i] + C[i];
    }
    ```

#### Vectorized Code:
    ```c
    for(int i=0; i< N; i+=4)
    {
        //Load 8 elements from B and C
        __m256d b = _mm256_load_pd(&B[i]);
        __m256d c = _mm256_load_pd(&C[i]);
        //Add the elements
        __m256d a = _mm256_add_pd(b, c);
        //Store the result
        _mm256_store_pd(&A[i], a);
    }
    ```
#### Analysis: 
The loop body is "A[i] = B[i] + C[i];".
Each iteration of the vectorized loop should do the work of 8 (vector length) iterations of the original loop.
The loop can be vectorized by loading 8 elements from B and C, adding them and storing the result in A. 
The loop is then incremented by the vector length.
    
    
### Example 2: In a loopnest, each inner iteration uses the same value of one input, and distinct elements of the other and stores in an output.
#### Original Code:
    ```c
    for(int i=0; i< M; i++)
    {
        for(int j=0; j< N; j++)
        {
            C[i*N + j] = A[i] * B[j];
        }
    }
    ```

#### Vectorized Code:
    ```c
    for(int i=0; i< M; i++)
    {
        //Broadcast A[i]
        __m256d a = _mm256_broadcast_sd(&A[i]);
        for(int j=0; j< N; j+=4)
        {
            //Load 8 elements from B
            __m256d b = _mm256_load_pd(&B[j]);
            //Multiply with A[i]
            __m256d c = _mm256_mul_pd(a, b);
            //Store the result
            _mm256_store_pd(&C[i*N + j], c);
        }
    }
    ```

#### Analysis: 
The loop body is " C[i*N + j] = A[i] * B[j];".
The same value of A is used across all iterations of the inner loop.
So the elements of A can be broadcasted, *outside* the inner loop.
The inner loop can be vectorized by loading 8 elements from B, multiplying with the broadcasted A, then storing in C.
The loop is then incremented by vector length.
    
### Example 3: In a loop, each iteration uses the same value of A, and distinct elements of B and accumulates into an output C
#### Original Code:
    ```c
    for(int j=0; j< N; j++)
    {
        c[j] = a[0] * B[j] - c[j];
    }
    ```

#### Vectorized Code:
    ```c
    //Broadcast a[0]
    __m256d a = _mm256_broadcast_ss(&a[0]);
    for(int j=0; j< N; j+=4)
    {
        __m256d b = _mm256_load_pd(&B[j]);
        __m256d c = _mm256_load_pd(&c[j]);
        c = _mm256_fmsub_pd(a, b, c);
        _mm256_store_pd(&c[j], c);
    }
    ```
#### Analysis: 
The loop body is "c[j] += a[0] * B[j];".
The same value of a is used across all iterations of the loop, so we broadcast a outside the inner loop.
The product of a and b is added to C. So we must load B and C. 
Then, we can use the fused multiply-add instruction to perform the computation in the loop body.
The loop is then incremented by the vector length.

    

---

# Challenge:
## Ojective
 Vectorize the following code snippet using SIMD instructions. Assume variables are defined
## Context
    Vector ISA: AVX2
    Scalar dataype: double
## Original Code: 
    ```c
    for(int k = 0; k< 256; k++)
    {
        for(int i=0; i< 6; i++)
        {
            for(int j=0; j< 8; j++)
            {   
                C[i*N + j] += A[k*6 + i] * B[k*8 +j];
            }
        }
    }
    ```
## Vectorized Code:

"""