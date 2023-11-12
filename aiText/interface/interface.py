import streamlit as st

from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
my_model_name = "models/bert_classifier"
pretrained_model_name = "bert-base-multilingual-cased"

from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained(my_model_name)


tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

device = torch.device("cpu")


def preprocess(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    return inputs

def predict(test_string = "This is a test"):
    model.eval()    
    with torch.no_grad():
        inputs = preprocess(test_string)
        input_ids = inputs['input_ids'].squeeze(1).to(device)
        attention_mask = inputs['attention_mask'].squeeze(1).to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        return logits


def main():
    st.title("AI Text Detector")

    text = st.text_area("Enter text here", height=200)

    if st.button("Predict"):
        logits = predict(text)
        st.write(logits)

        if logits[0][0] > logits[0][1]:
            st.header("Human generated")

        else:
            st.header("AI generated")


if __name__ == "__main__":
    main()