from transformers import AutoModel

processor = AutoModel.from_pretrained("facebook/wav2vec2-xls-r-1b")

print(dir(processor))
