
# What is this?
A fork of VLLM 0.2 that allows for fast decoding from a mixture of two models that get their own separate KV caches. Check out the example below.


# Instructions
Clone this repo and install from source (WARNING: this will install over your existing vllm install, if any):
```
git clone git@github.com:hunterlang/vllm-mixture.git
cd vllm-mixture
git checkout mixture
pip install -e .
```
# Example

```
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
] * 20
sampling_params = SamplingParams(temperature=0., top_p=0.95)

llm = LLM(model="mistralai/Mixtral-8x7B-Instruct-v0.1", mixin_model="mistralai/Mistral-7B-v0.1", mixture_coef=0.1, tensor_parallel_size=2)
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

# TODO

- allow for custom user-specified functions that tell the server how to combine the logits of the two models. This will allow for applications like Contrastive Decoding and Co-LLM.
- update to newer vllm
