import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Open-source model
MODEL_NAME = "microsoft/phi-2"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return tokenizer, model


tokenizer, model = load_model()
# Load tokenizer and model
#tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,device_map="auto")


st.title("Phi-2 Chat")

# Prompt
#prompt = "how are you?"
#prompt = input("input your question: ").strip()
prompt = st.text_input("Input your question:")

if prompt:
    # Count input tokens
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_token_count = input_ids.shape[1]

    # Generate response
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7
        )

    # Extract generated tokens only
    generated_ids = output_ids[0][input_token_count:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True,clean_up_tokenization_spaces=False)

    # Count output tokens
    output_token_count = len(generated_ids)

    # Total tokens
    total_token_count = input_token_count + output_token_count

    # Display results
    #print("Prompt:", prompt)
    #print("Response:", response.strip())
    st.write("Response:", response.strip())
    #print(f"Input Tokens:  {input_token_count}")
    st.write(f"Input Tokens:  {input_token_count}")
    #print(f"Output Tokens: {output_token_count}")
    st.write(f"Output Tokens: {output_token_count}")
    #print(f"Total Tokens:  {total_token_count}")
    st.write(f"Total Tokens:  {total_token_count}")