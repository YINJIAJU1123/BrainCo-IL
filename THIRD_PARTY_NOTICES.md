# Third-party notices

## Diffusion Policy

`src/openpi/models/diffusion.py` adapts the conditional temporal U-Net architecture
and epsilon prediction / sampling algorithm from
[real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy),
reference commit `5ba07ac6661db573af695b419a7947ecb704690f`.
The source files consulted are `diffusion_policy/model/diffusion/conditional_unet1d.py`,
`conv1d_components.py`, `positional_embedding.py`,
`diffusion_policy/policy/diffusion_unet_image_policy.py`, and
`diffusion_policy/config/train_diffusion_unet_image_workspace.yaml`.
The BrainCo implementation uses JAX/Flax NNX and is not compatible with upstream
PyTorch checkpoint files. The following license applies to the adapted portions.

```text
MIT License

Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
