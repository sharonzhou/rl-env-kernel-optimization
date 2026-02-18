Grade the final state of the sandbox.

Kernel-level grading
1. Calls Magpie to evaluate kernel-level performance
2. Scores based on: if the kernel compiled, is correct (passes unit tests), and the performance improvement of the individual kernel (AgentKernelArena, today)

Model-level grading
1. Calls Magpie to evaluate end-to-end model performance on relevant configurations
2. Scores based on: if the kernel compiled, is correct (passes unit tests), the performance improvement of the individual kernel, and the end-to-end performance of the model (and possibly: difficulty/importance of the configuration)
