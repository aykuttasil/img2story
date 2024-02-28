import os
from dotenv import load_dotenv, find_dotenv
import requests
from transformers import pipeline
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st


load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


def img2text(url):
    img2text = pipeline(
        "image-to-text", "Salesforce/blip-image-captioning-base")

    text = img2text(url)[0]['generated_text']
    print(text)
    return text


def generate_story(scenario):
    template = """
    You are a story teller;
    You can generate a short story based on a simple narrative, the story should be no more than 20 words;
    
    CONTEXT: {scenario}
    STORY:
    """

    template1 = """
    Sen dünyaca ünlü bir hikaye anlatıcısısın;
    Basit bir anlatıya dayanan kısa bir hikaye oluşturabilirsiniz, hikaye en fazla 20 kelime olmalıdır;
    Hikayeyi Türkçe olarak anlatmalısın.
    
    CONTEXT: Aizonai, Türkiye'de bulunan antik kent
    STORY:
    """

    prompt = PromptTemplate(template=template, input_variables=["scenario"])

    llm = Ollama(model="mistral")

    story_llm = LLMChain(llm=llm, prompt=prompt, verbose=True)

    story = story_llm.predict(scenario=scenario)
    print(story)
    return story


def text_to_speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payloads = {"inputs": message}

    response = requests.post(API_URL, headers=headers, json=payloads)
    with open('audio.mp3', 'wb') as file:
        file.write(response.content)


def main():
    st.title("LangChain")
    st.subheader("A language chain for language generation and translation")

    st.write("Upload an image to generate a story")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.read()
        with open(uploaded_file.name, "wb") as f:
            f.write(bytes_data)
        st.image(uploaded_file, caption='Uploaded Image.',
                 use_column_width=True)
        scenario = img2text(uploaded_file.name)
        story = generate_story(scenario)
        text_to_speech(story)

        with st.expander("sceneario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)

        st.audio('audio.mp3', format='audio/mp3')


if __name__ == "__main__":
    main()
