import numpy as np
import streamlit as st
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Load the trained CNN model
model = load_model('FV.h5')

# Dictionary of labels and corresponding fruit/vegetable names
labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}
fruits = ['apple', 'banana', 'bello pepper', 'chilli pepper', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'mango', 'orange',
          'paprika', 'pear', 'pineapple', 'pomegranate', 'watermelon']
vegetables = ['beetroot', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'corn', 'cucumber', 'eggplant', 'ginger',
              'lettuce', 'onion', 'peas', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato',
              'tomato', 'turnip']

# Dictionary of nutritional information for different types of fruits and vegetables
nutrition_info = {
    'apple': {'calories': 95, 'vitamins': ['Vitamin C', 'Vitamin K'], 'fat': 0.3, 'protein': 1},
    'banana': {'calories': 105, 'vitamins': ['Vitamin C', 'Vitamin B6'], 'fat': 0.4, 'protein': 1.3},
    'bell pepper': {'calories': 20, 'vitamins': ['Vitamin C', 'Vitamin A'], 'fat': 0.2, 'protein': 0.9},
    'chilli pepper': {'calories': 40, 'vitamins': ['Vitamin C', 'Vitamin A'], 'fat': 0.4, 'protein': 1.3},
    'grapes': {'calories': 62, 'vitamins': ['Vitamin C', 'Vitamin K'], 'fat': 0.2, 'protein': 0.6},
    'jalepeno': {'calories': 5, 'vitamins': ['Vitamin C', 'Vitamin A'], 'fat': 0.1, 'protein': 0.4},
    'kiwi': {'calories': 42, 'vitamins': ['Vitamin C', 'Vitamin K'], 'fat': 0.5, 'protein': 1},
    'lemon': {'calories': 29, 'vitamins': ['Vitamin C'], 'fat': 0.3, 'protein': 1.1},
    'mango': {'calories': 99, 'vitamins': ['Vitamin C', 'Vitamin A'], 'fat': 0.4, 'protein': 1},
    'orange': {'calories': 47, 'vitamins': ['Vitamin C'], 'fat': 0.1, 'protein': 0.9},
    'paprika': {'calories': 20, 'vitamins': ['Vitamin C', 'Vitamin A'], 'fat': 0.2, 'protein': 0.9},
    'pear': {'calories': 101, 'vitamins': ['Vitamin C', 'Vitamin K'], 'fat': 0.2, 'protein': 0.4},
    'pineapple': {'calories': 50, 'vitamins': ['Vitamin C'], 'fat': 0.1, 'protein': 0.5},
    'pomegranate': {'calories': 83, 'vitamins': ['Vitamin C', 'Vitamin K'], 'fat': 1.2, 'protein': 1.7},
    'watermelon': {'calories': 30, 'vitamins': ['Vitamin C', 'Vitamin A'], 'fat': 0.2, 'protein': 0.6},
    'beetroot': {'calories': 43, 'vitamins': ['Vitamin C', 'Vitamin A'], 'fat': 0.2, 'protein': 1.6},
    'cabbage': {'calories': 25, 'vitamins': ['Vitamin C', 'Vitamin K'], 'fat': 0.1, 'protein': 1.3},
    'capsicum': {'calories': 20, 'vitamins': ['Vitamin C', 'Vitamin A'], 'fat': 0.2, 'protein': 0.9},
    'carrot': {'calories': 41, 'vitamins': ['Vitamin C', 'Vitamin A'], 'fat': 0.2, 'protein': 0.9},
    'cauliflower': {'calories': 25, 'vitamins': ['Vitamin C', 'Vitamin K'], 'fat': 0.3, 'protein': 1.9},
    'corn': {'calories': 86, 'vitamins': ['Vitamin C', 'Vitamin B6'], 'fat': 1.4, 'protein': 3.3},
    'cucumber': {'calories': 16, 'vitamins': ['Vitamin C', 'Vitamin K'], 'fat': 0.1, 'protein': 0.7},
    'eggplant': {'calories': 25, 'vitamins': ['Vitamin C', 'Vitamin K'], 'fat': 0.2, 'protein': 1.2},
    'ginger': {'calories': 80, 'vitamins': ['Vitamin C', 'Vitamin B6'], 'fat': 0.8, 'protein': 1.8},
    'lettuce': {'calories': 15, 'vitamins': ['Vitamin C', 'Vitamin A'], 'fat': 0.1, 'protein': 0.9},
    'onion': {'calories': 40, 'vitamins': ['Vitamin C', 'Vitamin B6'], 'fat': 0.1, 'protein': 1.1},
    'peas': {'calories': 81, 'vitamins': ['Vitamin C', 'Vitamin A'], 'fat': 0.4, 'protein': 5.4},
    'potato': {'calories': 77, 'vitamins': ['Vitamin C', 'Vitamin B6'], 'fat': 0.1, 'protein': 2},
    'raddish': {'calories': 16, 'vitamins': ['Vitamin C', 'Vitamin A'], 'fat': 0.1, 'protein': 0.6},
    'soy beans': {'calories': 298, 'vitamins': ['Vitamin C', 'Vitamin K'], 'fat': 20, 'protein': 36.5},
    'spinach': {'calories': 23, 'vitamins': ['Vitamin C', 'Vitamin A'], 'fat': 0.4, 'protein': 2.9},
    'sweetcorn': {'calories': 86, 'vitamins': ['Vitamin C', 'Vitamin B6'], 'fat': 1.4, 'protein': 3.3},
    'sweetpotato': {'calories': 86, 'vitamins': ['Vitamin C', 'Vitamin A'], 'fat': 0.1, 'protein': 1.6},
    'tomato': {'calories': 18, 'vitamins': ['Vitamin C', 'Vitamin A'], 'fat': 0.2, 'protein': 0.9},
    'turnip': {'calories': 28, 'vitamins': ['Vitamin C', 'Vitamin A'], 'fat': 0.1, 'protein': 1.1}
}


def preprocess_image(uploaded_file_path):
    # Read the image file and resize it to the required input size for the model
    image = load_img(uploaded_file_path, target_size=(224, 224))
    # Convert the image to a numpy array
    image = img_to_array(image)
    # Normalize the pixel values
    image = image / 255
    image = np.expand_dims(image, [0])
    return image


def predict_label(image):
    # Make a prediction using the CNN model
    prediction = model.predict(image)
    # Get the index of the highest prediction
    label_index = np.argmax(prediction)
    # Get the corresponding label from the labels list
    label = labels[label_index]
    return label


def get_nutrition_info(label):
    # Get the nutrition information for the label
    info = nutrition_info[label]
    return info


# Define the main function
def main():
    st.title("Fruits🍍-Vegetable🍅 Classification")
    st.sidebar.title("Fruits🍍-Vegetable🍅 Classification")

    uploaded_file = st.sidebar.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).resize((250, 250))
        st.sidebar.image(image, use_column_width=False)
        uploaded_file_path = './upload_images/' + uploaded_file.name
        with open(uploaded_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        image = preprocess_image(uploaded_file_path)
        lable = predict_label(image)
        if lable in fruits:
            st.info('**Category : fruits**')
            info = get_nutrition_info(lable)
            # Display the predicted label and nutrition information
            st.write("Predicted label: ", lable)
            st.write("Calories: ", info['calories'])
            st.write("Vitamins: ", info['vitamins'])
            st.write("Fat: ", info['fat'])
            st.write("Protein: ", info['protein'])
        elif lable in vegetables:
            st.info('**Category : vegetables**')
            info = get_nutrition_info(lable)
            # Display the predicted label and nutrition information
            st.write("Predicted label: ", lable)
            st.write("Calories: ", info['calories'])
            st.write("Vitamins: ", info['vitamins'])
            st.write("Fat: ", info['fat'])
            st.write("Protein: ", info['protein'])
        else:
            st.write("Nutrition Data Cannot be identified Model it is Not Trained for uploaded image")


main()
