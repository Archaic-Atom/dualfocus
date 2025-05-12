from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch

model = AutoModel.from_pretrained("/data1/sunmingyu/xiaoxia_laboratory/my_study/models/siglip")
processor = AutoProcessor.from_pretrained("/data1/sunmingyu/xiaoxia_laboratory/my_study/models/siglip")

image = Image.open("./000000039769.jpg")

texts = ["a photo of 2 cats", "a photo of 2 dogs"]
inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits_per_image = outputs.logits_per_image
probs = torch.sigmoid(logits_per_image) # these are the probabilities
print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")
