from transformers import T5ForConditionalGeneration, T5Tokenizer

def inference(finetuned_path: str, user_query: str):

    finetuned_model = T5ForConditionalGeneration.from_pretrained(finetuned_path)
    tokenizer = T5Tokenizer.from_pretrained(finetuned_path)

    inputs = "Please answer to this question: " + user_query
    inputs = tokenizer(inputs, return_tensors="pt")

    outputs = finetuned_model.generate(**inputs)
    answer = tokenizer.decode(outputs[0])

    return answer
