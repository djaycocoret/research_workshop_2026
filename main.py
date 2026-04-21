from transformers import AutoModelForPreTraining, AutoProcessor

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-xls-r-1b")
model = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-xls-r-1b")

print(dir(model))
