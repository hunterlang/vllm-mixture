from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
] * 20000
sampling_params = SamplingParams(temperature=0., top_p=0.95)

#llm = LLM(model="mistralai/Mixtral-8x7B-Instruct-v0.1", mixin_model="mistralai/Mistral-7B-v0.1", mixture_coef=0.1, tensor_parallel_size=2)
#llm = LLM(model="facebook/opt-2.7b", mixin_model="facebook/opt-125m", mixture_coef=0.1)#, tensor_parallel_size=2)
llm = LLM(model="facebook/opt-2.7b")#, tensor_parallel_size=2)
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
