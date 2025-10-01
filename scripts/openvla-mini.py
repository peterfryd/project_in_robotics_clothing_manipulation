# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import time


# Grab image input & format prompt
# image: Image.Image = get_from_camera(...)
print("Loading image...")
start = time.time()
image_path = "test_small.jpg"
image: Image.Image = Image.open(image_path)
prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"
end = time.time()
print("Loading image time:", end-start)


# Load Processor & VLA
print("Initialize model...")
start = time.time()
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    # attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.float32,
    #low_cpu_mem_usage=True,
    trust_remote_code=True
)
end = time.time()
print("Initialize model time:", end-start)
file = open("time_test.csv", "a")
file.write(f"{end-start} , , ")
file.close()


# Quantize model
print("Quantize model...")
start = time.time()
quantized_vla = torch.quantization.quantize_dynamic(
    vla, {torch.nn.Linear}, dtype=torch.qint8
)
quantized_vla.to("cpu")
end = time.time()
print("Quantize model time:", end-start)


# Predict action
print("Predict action time:", end-start)
file = open("time_test.csv", "a")
file.write(f" , {end-start} , ")
file.close()

# Predict Action (7-DoF; un-normalize for BridgeData V2)
print("Create input...")
start = time.time()
inputs = processor(prompt, image).to("cpu")
end = time.time()
print("Create input time:", end-start)

print("Predict action...")
start = time.time()
action = quantized_vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
print(action)
end = time.time()

print("Predict action time:", end-start)
file = open("time_test.csv", "a")
file.write(f", , {end-start} \n")
file.close()


    # Execute...
    # robot.act(action, ...)