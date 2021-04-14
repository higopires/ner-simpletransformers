#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 15:13:55 2021

@author: higo
"""

import streamlit as st
from simpletransformers.ner import NERModel
import torch
from simpletransformers.streamlit.streamlit_utils import simple_transformers_model

@st.cache(hash_funcs={NERModel: simple_transformers_model})
def get_prediction(model, input_text):
    predictions, _ = model.predict([input_text])

    return predictions

def main():
    st.title("NER - Simple Transformers")
    
    cuda_available = torch.cuda.is_available()
    custom_labels=['NN','NNP','IN','DT','JJ','NNS','VBD','VBN','VBZ','CD','VB',
                   'CC','TO','RB','VBG','VBP','PRP','POS','PRP$','MD','WDT','JJS',
                   'JJR','WP','NNPS','RP','WRB','RBR','EX','RBS','PDT','WP$','UH','FW']
    model = NERModel("roberta", "best_model", use_cuda=cuda_available, labels=custom_labels)
    
    st.sidebar.subheader("Parameters")
    model.args.max_seq_length = st.sidebar.slider("Max Seq Length", min_value=1, max_value=92,
                                                  value=model.args.max_seq_length)
    model.args.use_multiprocessing = False
    
    st.subheader("Digite o texto: ")
    input = st.text_area("")
    
    if st.button("Analisar"):
        prediction = get_prediction(model, input)[0]
        st.write(prediction)

if __name__ == '__main__':
    main()