import streamlit as st
from diffusers import DiffusionPipeline
from PIL import Image
import torch

# Load the diffusion pipeline model
@st.cache_resource
def load_pipeline():
    pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe.load_lora_weights("Melonie/text_to_image_finetuned")
    return pipe

pipe = load_pipeline()

# Streamlit app
st.title("Text-to-Image Generation App")

# User input for prompt
user_prompt = st.text_input("Enter your image prompt", value="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k")

# Button to generate the image
if st.button("Generate Image"):
    if user_prompt:
        with st.spinner("Generating image..."):
            # Generate the image
            image = pipe(user_prompt).images[0]
            
            # Display the generated image
            st.image(image, caption="Generated Image", use_column_width=True)
    else:
        st.error("Please enter a valid prompt.")
