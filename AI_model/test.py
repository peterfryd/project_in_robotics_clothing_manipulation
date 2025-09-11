# pip install transformers datasets evaluate accelera
# pip install typeguard torchvision pillow cython opencv-python
# pip install ipywidgets
# !pip install -U "huggingface_hub[cli]"
# !pip install torch
# !pip install -U transformers datasets evaluate accelerate timm


from huggingface_hub import notebook_login

notebook_login()







# !hf auth login



from transformers import AutoModelForCausalLM, AutoTokenizer

# model_id = "meta-llama/Llama-2-7b-hf"
# model_id = "LLM360/K2-Think"
model_id = "janhq/Jan-v1-2509"

model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)



model_inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="pt").to(model.device)



# generated_ids = model.generate(**model_inputs, max_length=30)
generated_ids = model.generate(**model_inputs)
ans = tokenizer.batch_decode(generated_ids)[0]
print(ans)
'<s> The secret to baking a good cake is 100% in the preparation. There are so many recipes out there,'



# from transformers import pipeline, infer_device

# device = infer_device()

# pipeline = pipeline("text-generation", model=model_id, device=device)


# print("HelloS")
# pipeline("The secret to baking a good cake is ", max_length=50)
# #[{'generated_text': 'The secret to baking a good cake is 100% in the batter. The secret to a great cake is the icing.\nThis is why we’ve created the best buttercream frosting reci'}]