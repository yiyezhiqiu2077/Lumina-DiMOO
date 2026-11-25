This project builds on [Lumina-DiMOO](https://github.com/Alpha-VLLM/Lumina-DiMOO)
by Alpha-VLLM; its original license and attribution remain in LICENSE and source files.

Region-DiMO/CBC kernels and LiveMGT method adaptations derive from
[EditMGT](https://github.com/Shi-qingyu/EditMGT) by Shi-qingyu and collaborators,
reference commit b0f7ce3f05a8c5c36c61ae4f07fd49870b8df55d.
The upstream license is preserved verbatim at licenses/EditMGT-LICENSE.
Adaptations include Lumina's VQ grid, unified visual vocabulary, Text/Image CFG,
compatible Type RoPE, distributed full-parameter training and checkpoint handling.

PyTorch, Transformers, Diffusers and optional bitsandbytes are separately
licensed dependencies. Model weights and datasets have their own upstream
licenses; they are not included in this code distribution.
