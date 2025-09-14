#This is part of the source code

#  Load fine-tuned model for inference
fine_tuned_model = AutoModelForCausalLM.from_pretrained(
    "./fine_tuned_model",
    device_map="auto",
    torch_dtype=dtype
)
pipe = pipeline("text-generation", model=fine_tuned_model, tokenizer=tokenizer)

# Test fine-tuned model
prompt = "Explain the significance of the Turing Test in artificial intelligence."
response = pipe(f"<s>[INST] {prompt} [/INST]", max_new_tokens=100)
print(response[0]["generated_text"])