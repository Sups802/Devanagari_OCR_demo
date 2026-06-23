import os
import tensorflow as tf
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Paths
train_dir = "train_set_path"
test_dir = "test_set_path"
img_size = 96
batch_size = 32
epochs = 10

# Auto-label encoding based on folder names
def create_generators(img_size, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1,
        horizontal_flip=False
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='sparse' # labels will be float32 integers
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=False
    )
    return train_generator, test_generator
    
# Load data
train_gen, test_gen = create_generators(img_size, batch_size)
num_classes = len(train_gen.class_indices)
    
# Build model
base_model = MobileNetV2(input_shape=(img_size, img_size, 3), 
                         include_top=False, 
                         weights='imagenet')
base_model.trainable = False # Freeze base
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)

output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

# Save best model
checkpoint = ModelCheckpoint("mobilenetv2_dhcd.keras", 
                             monitor='val_accuracy',
                             save_best_only=True, 
                             verbose=1)

# Train
history = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=test_gen,
    callbacks=[checkpoint]
)

# Evaluate
loss, acc = model.evaluate(test_gen)
print(f"\nFinal Test Accuracy: {acc * 100:.2f}%")

model.save("mobilenetv2_dhcd_full.keras") #save model
print("Model saved as 'mobilenetv2_dhcd_full.keras'")

#Streamlit App code-
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
CORRECT_COLOR = (0, 255, 0)
INCORRECT_COLOR = (0, 0, 255)

# CHARACTER MAPPING
character_mapping = {
    0: 'क', 1: 'ख', 2: 'ग', 3: 'घ', 4: 'ङ',
    5: 'च', 6: 'छ', 7: 'ज', 8: 'झ', 9: 'ञ',
    10: 'ट', 11: 'ठ', 12: 'ड', 13: 'ढ', 14: 'ण',
    15: 'त', 16: 'थ', 17: 'द', 18: 'ध', 19: 'न',
    20: 'प', 21: 'फ', 22: 'ब', 23: 'भ', 24: 'म',
    25: 'य', 26: 'र', 27: 'ल', 28: 'व', 29: 'श',
    30: 'स', 31: 'ष', 32: 'ह', 33: 'क्ष', 34: 'त्र', 35: 'ज्ञ',
    36: '०', 37: '१', 38: '२', 39: '३', 40: '४',
    41: '५', 42: '६', 43: '७', 44: '८', 45: '९'
}

# LOAD MODEL
@st.cache_resource
def load_devanagari_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# IMAGE PREPROCESSING & SEGMENTATION
def segment_characters(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=20)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 35, 15
    )
    kernel = np.ones((2, 2), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    char_imgs, bboxes = [], []
    H, W = clean.shape
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 0.01*W or h < 0.01*H or w > 0.5*W or h > 0.5*H:
            continue
        roi = image[y:y+h, x:x+w]
        char_imgs.append(roi)
        bboxes.append((x, y, w, h))
    if char_imgs:
        sorted_pairs = sorted(zip(char_imgs, bboxes), key=lambda b: b[1][0])
        char_imgs, bboxes = zip(*sorted_pairs)
        return list(char_imgs), list(bboxes)
    else:
        return [], []
        
        
def preprocess_char(img, target_size=(96, 96)):
    resized = cv2.resize(img, target_size)
    norm = resized / 255.0
    return np.expand_dims(norm, axis=0)

# PREDICTION
def predict_characters(model, char_images):
    predictions = []
    for img in char_images:
        inp = preprocess_char(img)
        probs = model.predict(inp, verbose=0)
        idx = np.argmax(probs)
        predictions.append(character_mapping.get(idx+1, "?"))
    return predictions

# STREAMLIT UI
st.set_page_config(page_title="Devanagari OCR", layout="wide")
st.title("📜 Devanagari OCR — Character Recognition (Simple)")

model_path = st.text_input("🔹 Enter Model Path:",
"/workspaces/Devanagari_OCR_demo/mobilenetv2_dhcd_full.keras")

ground_truth = st.text_input("🔹 Ground Truth Text (optional):", "च")
uploaded_file = st.file_uploader("📤 Upload Devanagari manuscript image", type=["jpg", "png", "jpeg"])

if uploaded_file and model_path:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if img is not None:

        # Display uploaded image small
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image",
        width=300)
        model = load_devanagari_model(model_path)
        
        if model:
            char_imgs, bboxes = segment_characters(img)
            if not char_imgs:
                st.warning("No characters segmented. Prediction may fail.")
                predicted_text = ""
            else:
                #st.write("Predicted raw indices:", [np.argmax(model.predict(preprocess_char(img),
                #verbose=0)) for img in char_imgs])

                predicted = predict_characters(model, char_imgs)
                predicted_text = "".join(predicted)

        # Show results
        st.subheader(" OCR Result")
        st.write(f"**Ground Truth:** {ground_truth}")
        st.write(f"**Predicted :** {predicted_text}")
        if ground_truth == predicted_text:
            st.success("✅ Prediction matches Ground Truth")
        else:
            st.error("❌ Prediction does NOT match Ground Truth")
    else:
        st.error("Failed to read image.")
