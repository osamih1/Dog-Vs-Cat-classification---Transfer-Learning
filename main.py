import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

def dog_vs_cat_prediction(image):
    img = Image.open(image)
    img = img.resize((224, 224))
    img_array = np.array(img).reshape(1, 224, 224, 3)
    img_array = img_array / 255
    y_pred = trained_model.predict(img_array)
    label = np.argmax(y_pred)

    if label == 0:
        return "This image represents a cat!"
    else:
        return "This image respresents a dog!"

def main():
    st.title("Dog Vs Cat classification - Transfer Learning")
    image = st.file_uploader("Upload your dog or cat image here!")
    if image:
        result = dog_vs_cat_prediction(image)
        st.success(result)




if __name__ == "__main__":
    trained_model = tf.keras.models.load_model("trained_model")
    main()