from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os

hf_token = os.environ['HUGGINGFACE_TOKEN']

print("#### Creating model and tokenizers.")
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)
print("#### Model, Tokenizers created.")

print("#### Getting peft model.")
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)

print("#### Preparing dataset.")
df = pd.read_csv("/root/ft/selected_random_samples.csv")
def create_img_paths(id):
    img_directory = "/root/ft/selected_images"
    path = f"{img_directory}/{id}.jpg"
    return path
df["image"] = df["id"].apply(lambda x: create_img_paths(x))

print("#### Converting dataset to conversation format.")
instruction = """
You are an expert at detecting sarcasm in online content. Analyze the provided input, which may be text only, image only, or both text and image together.
Is the given input(s) sarcastic?
"""

def convert_to_conversation(sample):
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : instruction},
            {"type": "text", "text" : f"Text: {sample['text']}"},
            {"type" : "image", "image" : Image.open(sample["image"])} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : f"{sample['IsSarcasm']}"}
          ]
        },
    ]
    return { "messages" : conversation }

converted_dataset = [
    convert_to_conversation(sample) for i, sample in tqdm(df.iterrows(), total=len(df))
]
print("#### Dataset preparation completed.")

print("#### Inference Test")
def inference(image, text):
    FastVisionModel.for_inference(model) # Enable for inference!
    instruction = """
    You are an expert at detecting sarcasm in online content. Analyze the provided input, which may be text only, image only, or both text and image together.
    Is the given input(s) sarcastic?
    """

    messages = [
        {"role": "user",
        "content": [
            {"type": "text", "text": instruction},
            {"type": "text", "text": f"Text: {text}"},
            {"type": "image"}
        ]}
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens = False,
        return_tensors = "pt",
    ).to("cuda")
    from transformers import TextStreamer
    # text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True, temperature = 1.5, min_p = 0.1)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
inference(Image.open(df.iloc[23]["image"]), df.iloc[23]["text"])

print("""#### Training the model""")
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

FastVisionModel.for_training(model) # Enable for training!

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    train_dataset = converted_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        # num_train_epochs = 1, # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",     # For Weights and Biases

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 4,
        max_seq_length = 2048,
    ),
)

#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

print("""### Saving and pushing to hub""") 
model.save_pretrained("outputs")
model.save_pretrained_merged("outputs_merged", tokenizer, save_method = "merged_16bit")
model.push_to_hub("sinngam-khaidem/Qwen2-VL-7B-Instruct-bnb-4bit-MMSD-lora", token = hf_token) # Online saving
tokenizer.push_to_hub("sinngam-khaidem/Qwen2-VL-7B-Instruct-bnb-4bit-MMSD-lora", token = hf_token) # Online saving
model.push_to_hub_merged("sinngam-khaidem/Qwen2-VL-7B-Instruct-bnb-4bit-MMSD-merged", tokenizer, save_method = "merged_16bit",token = hf_token)



