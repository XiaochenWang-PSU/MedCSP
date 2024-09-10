# MedCSP
Here lies the materials of the paper "Unity in Diversity: Bridging Multimodal Medical Sources via Cross-source Pre-training". 

## Checkpoints
The pre-trained CLIP model is available at https://huggingface.co/xcwangpsu/MedCSP_clip. Here is a demo of how to utilize the CLIP for encoding: 

```python
from open_clip import create_model_from_pretrained, get_tokenizer
import torch
from urllib.request import urlopen
from PIL import Image

# import model, processor and tokenizer
model, processor = create_model_from_pretrained('hf-hub:xcwangpsu/MedCSP_clip')
tokenizer = get_tokenizer('hf-hub:xcwangpsu/MedCSP_clip')



# encode image:

# import raw radiological image:
image = Image.open(urlopen("https://huggingface.co/xcwangpsu/MedCSP_clip/resolve/main/image_sample.jpg"))

# preprocess the image, the final tensor should have 4 dimensions (B, C, H, W)
processed_image = processor(image)
processed_image = torch.unsqueeze(processed_image, 0)
print("Input size:", processed_image.shape)

# encode to a single embedding
image_embedding = model.encode_image(processed_image)
print("Individual image embedding size:",image_embedding.shape)

# sequential encoding
seq_image_embedding = model.visual.trunk.forward_features(processed_image)
print("Sequential image embedding size:",seq_image_embedding.shape)


# encode text:

text = "Chest X-ray reveals increased lung opacity, indicating potential fluid buildup or infection."
tokens = tokenizer(text)

# encode to a single embedding
text_embedding = model.encode_text(tokens)
print("Individual text embedding size:",text_embedding.shape)

# sequential encoding
seq_text_embedding = model.text.transformer(tokens, output_hidden_states=True).hidden_states[-1]
print("Sequential text embedding size:", seq_text_embedding.shape)

```
