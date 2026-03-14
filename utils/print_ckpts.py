import torch


state_dict = torch.load("/root/autodl-tmp/tangjt/PyMCIT/checkpoints/Task1_llava_lora_ours/non_lora_trainables.bin", map_location="cpu")
#state_dict = torch.load("/root/autodl-tmp/tangjt/PyMCIT/checkpoints/Task1_llava_lora_ours/non_lora_trainables.bin",map_location="cpu")

print("Keys in adapter_model.bin:")
for key in state_dict.keys():
    print(key)

