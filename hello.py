# from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# # Load the pre-trained model and tokenizer
# model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
# tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

# # Define the prompt
# prompt = "Convert the following text to a question:\n\n"

# # Input text to convert to a question
# input_text = "The quick brown fox jumps over the lazy dog."

# # Concatenate the prompt and input text
# prompt += input_text

# # Encode the prompt
# input_ids = tokenizer.encode(prompt, return_tensors="pt")

# # Generate the question
# output = model.generate(input_ids, max_length=100, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)

# # Decode the output
# generated_question = tokenizer.decode(output[0], skip_special_tokens=True)

# print(f"Input text: {input_text}")
# print(f"Generated question: {generated_question}")

