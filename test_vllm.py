from vllm import LLM, SamplingParams

prompts = [
    "Generate a rationale for the following question. End your response with 'So the answer is <answer>.' Let's think step by step.\n\nQuestion: Caleb bought 10 cartons of ice cream and 4 cartons of frozen yoghurt. Each carton of ice cream cost $4 and each carton of frozen yoghurt cost $1. How much more did Caleb spend on ice cream than on frozen yoghurt?\n\nAnswer: 36\nRationale:",
] * 10
sampling_params = SamplingParams(temperature=0., top_p=0.8, max_tokens=4096)

base_model = "/localdata/hjl/Mistral-7B-v0.1~128bz~2e-5lr"
asst_model = "/localdata/hjl/Mistral-7B-v0.1~128bz~2e-5lr"

llm = LLM(model=base_model, mixin_model=asst_model, mixture_coef=0.1, tensor_parallel_size=1)
#llm = LLM(model=base_model, asst_model=asst_model, threshold=0.1, tensor_parallel_size=2)
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
