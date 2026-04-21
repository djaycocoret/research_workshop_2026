from transformers import AutoFeatureExtractor

processor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-1b")

print(dir(model))
