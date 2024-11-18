import streamlit as st
from fastai.vision.all import PILImage, load_learner

model = load_learner("/home/t1nnur/.fastai/data/mnist_png/mnist.pkl")

st.title("Digit prediction")


def main():
    file = st.file_uploader("Upload your file:")

    if not file:
        return

    img = PILImage.create(file)
    img.resize((28, 28))

    prediction, predict_id, percents = model.predict(img)

    st.success(f"Prediction: {prediction}")
    st.text(f"Percent: {percents[predict_id]}")

    st.image(img)


main()
