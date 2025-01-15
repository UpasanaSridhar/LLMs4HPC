from transformers import AutoTokenizer
import sys

def count_tokens(model_name: str, prompt: str) -> int:
    """
    Count the number of tokens in a prompt for a given model.

    Args:
        model_name (str): The Hugging Face model name.
        prompt (str): The input text to tokenize.

    Returns:
        int: Number of tokens in the prompt.
    """
    # Load the tokenizer for the specified model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize the prompt
    tokens = tokenizer.tokenize(prompt)

    # Return the number of tokens
    return len(tokens)

def main():
    """Main function to demonstrate token counting."""
    # List of models to test
    model_names = [
        "gpt2",
        "bert-base-uncased",
        "facebook/bart-large",
        "google/t5-small-lm-adapt",
        "codellama/CodeLlama-7b-Instruct-hf"
    ]

    # Example prompt
    prompt = """
You are an expert C and high-performance programmer. This is  example of a kernel that uses AVX2 vector instructions to compute double-precision matrix multiplication of a matrix A, 6x256 and a matrix B, 256x8 to procude C, a 6x8 matrix. B and C are row-major and A is column major.

```c
//vectorized version of the kernel
//unroll both independent loops (spill regis)
// Hoist loads and stores of C (register resident)
//Only use max 16 registers (reuse for A)
//POinter arithmetic
void dgemm_avx2(const double* A, const double* B, double* C) {
__m256d c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
__m256d b0, b1;
__m256d a0,  a1,  a2,  a3,  a4,  a5;

//Loads of C
c00= _mm256_load_pd(C + 0* 8 + 0);
c01 = _mm256_load_pd(C + 0* 8 + 4);
c10 = _mm256_load_pd(C + 1 * 8 + 0);
c11 = _mm256_load_pd(C + 1 * 8 + 4);
c20 = _mm256_load_pd(C + 2 * 8 + 0);
c21 = _mm256_load_pd(C + 2 * 8 + 4);
c30 = _mm256_load_pd(C + 3 * 8 + 0);
c31 = _mm256_load_pd(C + 3 * 8 + 4);
c40 = _mm256_load_pd(C + 4 * 8 + 0);
c41 = _mm256_load_pd(C + 4 * 8 + 4);
c50 = _mm256_load_pd(C + 5 * 8 + 0);
c51 = _mm256_load_pd(C + 5 * 8 + 4);

const double *a_ptr = A;
const double *b_ptr = B;

for (int k = 0; k < 256; ++k) {
    // Loads of B can be hoisted
    b0 = _mm256_load_pd(b_ptr + 0);
    b1 = _mm256_load_pd(b_ptr + 4);
    //Unrolled
    // for (int i = 0; i < 6; ++i) {
        //Unrolled  and vectorized
        // for (int j = 0; j < 8; j+=4) {
                a0 = _mm256_broadcast_sd(a_ptr + 0);
                c00 = _mm256_fmadd_pd(a0, b0, c00);
                c01 = _mm256_fmadd_pd(a0, b1, c01);
                a1 = _mm256_broadcast_sd(a_ptr + 1);
                c10 = _mm256_fmadd_pd(a1, b0, c10);
                c11 = _mm256_fmadd_pd(a1, b1, c11);

                a0 = _mm256_broadcast_sd(a_ptr + 2);
                c20 = _mm256_fmadd_pd(a0, b0, c20);
                c21 = _mm256_fmadd_pd(a0, b1, c21);
                a1 = _mm256_broadcast_sd(a_ptr + 3);
                c30 = _mm256_fmadd_pd(a1, b0, c30);
                c31 = _mm256_fmadd_pd(a1, b1, c31);
                
                a0 = _mm256_broadcast_sd(a_ptr + 4);
                c40 = _mm256_fmadd_pd(a0, b0, c40);
                c41 = _mm256_fmadd_pd(a0, b1, c41);
                a1 = _mm256_broadcast_sd(a_ptr + 5);
                c50 = _mm256_fmadd_pd(a1, b0, c50);
                c51 = _mm256_fmadd_pd(a1, b1, c51);
            // }
        // }
        b_ptr += 8; 
        a_ptr += 6;
    }

    //Stores of C
    _mm256_store_pd(C + 0* 8 + 0, c00);
    _mm256_store_pd(C + 0* 8 + 4, c01);
    _mm256_store_pd(C + 1 * 8 + 0, c10);
    _mm256_store_pd(C + 1 * 8 + 4, c11);
    _mm256_store_pd(C + 2 * 8 + 0, c20);
    _mm256_store_pd(C + 2 * 8 + 4, c21);
    _mm256_store_pd(C + 3 * 8 + 0, c30);
    _mm256_store_pd(C + 3 * 8 + 4, c31);
    _mm256_store_pd(C + 4 * 8 + 0, c40);
    _mm256_store_pd(C + 4 * 8 + 4, c41);
    _mm256_store_pd(C + 5 * 8 + 0, c50);
    _mm256_store_pd(C + 5 * 8 + 4, c51);

}
```
Use this example to write a single precision, matrix multiplication kernel. A is a 6x256 matrix, B is 256x16 and C is 6x16. Use AVX2 single precision instructions.
"""


    prompt = """
This is a set of loops that updates an MxN output matrix.
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
Objective: Re-order the loops to have different output elements be updated in the innermost loops.
Idea: loop iterators that are present in the LHS of the assignment are iterating over different output elements.
Analysis: The loop body is "C[i*N+j] += A[p*M+i] * B[p*N+j];" . The LHS of the assignment in contains the loop iterators i, and j. So the loop order must be changed to have i and j as the innermost loops. This leaves k as the outermost loop.

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

Challenge: Given this example, reorder the following loops to have different output elements computed in the inner loops

```c
for(int i=0; i< M; i++)
{
    for (int j = 0; j< N; j++)
    {
        for(int k = 0; k < P; k++)
        {
            C[i*N+j] += A[p*M+i] * B[p*N+j];
        }
    }
}
```
"""
    prompt_file = sys.argv[1]
    # Open the prompt file and read the prompt
    with open(prompt_file, "r") as file:
        prompt = file.read()

    print("Token counts for different models:")
    # for model_name in model_names:
    # with model_name as model_names[-1]:
    model_name = model_names[-1]
    try:
        num_tokens = count_tokens(model_name, prompt)
        print(f"{model_name}: {num_tokens} tokens")
    except Exception as e:
        print(f"Error with model {model_name}: {e}")





if __name__ == "__main__":
    main()
