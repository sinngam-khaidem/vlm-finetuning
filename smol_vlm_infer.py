import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load images
image = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
instruction = """
You are an expert at detecting sarcasm in online content. Analyze the provided input, which may be text only, image only, or both text and image together.
Is the given input(s) sarcastic?
"""

text = "I just love getting stuck in traffic"


# Initialize processor and model
processor = AutoProcessor.from_pretrained("sinngam-khaidem/Llama-3.2-11B-Vision-Instruct-MMSD-merged-2")
model = AutoModelForVision2Seq.from_pretrained(
    "sinngam-khaidem/Llama-3.2-11B-Vision-Instruct-MMSD-merged-2",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
).to(DEVICE)

# Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": instruction},
            {"type": "text", "text": f"Text: {text}"},
            {"type": "image"},   
        ]
    },
]
# Prepare inputs
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt")
inputs = inputs.to(DEVICE)
# Generate outputs
generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)
print(generated_texts[0])
"""
User:<image>Can you describe the two images?
Assistant: I can describe the first one, but I can't describe the second one.
"""
