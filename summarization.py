
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

def summarize(text, max_len=50):
    input_text = "summarize: " + text
    inputs = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=max_len, min_length=5, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

example = "The delivery was late, but the product quality was excellent. I would recommend buying again."
print("Short:", summarize(example, max_len=20))
print("Detailed:", summarize(example, max_len=60))
