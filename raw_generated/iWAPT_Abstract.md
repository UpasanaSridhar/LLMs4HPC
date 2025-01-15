Help me draft an extended abstract for IWAPT.
These are the sections I have with the outline for each section.

The problem

High-performance libraries use the abstraction of a kernel to isolate performance critical, and hardware-specific code to a small section of the implementation. However kernels are written by performance experts and the cost of writing new kernels for every dataype, workload and ISA is huge.

The insight

LLMs that can automatically generate code pose an  avenue for rapidly supporting new usecases of libraries. Other work has explored the use of LLMs but usually perform a cursory exploration of capabilities and then resort to fine-tuning.  We want to evaluate the strengths and weaknesses of LLMs and use our expertise in  systematically writing kernels to simplify the problem of writing kernels to the point where we can effectively use off-the-shelf models to automatically write kernels.

*Figure 1: Illustration of a step-by-step transformation of a reference piece of code to an  architecture specific one. Each step builds on the last, but vectorization and making sure that the kernel does not spill onto the stack are the biggest deltas in performance.*

 The training data used to train code-llms is not representative of HPC code. However, from our experimentation we see that LLMs are good at complex information retrieval and at style transfer like tasks. They are not as good at mathematical reasoning.

*Figure needed? I can just write in the prose that HPC code is usually written in C, C++ and Assembly but that these make up less than  20% of training data. With HPC code making up even less*

Approach

We use a running example of double-precision matrix multiply on an x86 AVX2 ARCHITECTURE.

We present 5 approaches to prompting Code-LLama to generate the kernel. These are 

A simple prompt that states the end-goals but does not prescribe anything about the approach.

A prompt with the steps to write the kernel prescribed in great detail.

A style-transfer-like prompt where a high-performance for a similar kernel (float on the same ISA) is given and the LLM is asked to translate to doubles.

A prompt  that performs a single step (vectorization) of the process of writing the kernel, this is similar to the above, where several  examples of vectorization are given as style guides.

A prompt that simply the LLM to fill out a table with the vector instructions of interest on the target ISA

These 5 approaches are ordered from most abstract to least abstract. However all tasks are useful to solve. For instance, the pattern of the sgemm kernel can be reused across the dgemm kernel, int kernels as well as kernels for other operations such as minimum spanning tree, k nearest neighbor estimations and so on. So if an LLM is able reliably style transfer, it can immediately be used to generate kernels in all these domains. Even the table filling prompt can be used in conjunction with abstract representation of vectorized kernels to automatically generate high-performance kernels

*Table comparing the performance of the different prompts on the different steps identified in the systematic way of writing the kernel*

Results

 1 and 2 perform atrociously.  1 is expected, it is just the control to establish that the type of information we want the LLM to retrieve is not often retrieved by it in training. With 2, some of the steps are followed, for instance, the reordering of the loops happens correctly, the expected vector instructions are used, but the loop bounds are wrong, the n loop is missing and some transformation such as the unrolling haven't happened at all. This seems to indicate that following multiple steps is not in the LLM's wheelhouse. The style transfer prompt (3) is very close to being correct, with unrolling, number of variables all being translated well. However, the loop iterator is incremented by the wrong amount (same as the example) and the pointer to the second input matrix is not incremented. This may be a case that could be improved with fine-tuning, as the produced output is literally 2 characters away from the correct answer. The specific task of vectorization generalize well as long as the target is the same ISA as the examples, and is better if the target is the same data type. The table filling task generalizes the best, but it is also the simplest task and requires integration with an abstract vector code-generator in order to be usable as an automatic kernel generator.

Conclusion

LLMs can be used to automatically generate HPC code with the tasks align with their abilities. They are more capable as is than a first glance would tell us.

This work  described ways in which LLMs maybe prompted and evaluated. Even if they are meant to be finetuned to the specific task, the prompting techniques described here may reduce the number of data points needed.

LLMs are good at style-transfer, retrieval. Not good at mathematical reasoning and following multiple steps.