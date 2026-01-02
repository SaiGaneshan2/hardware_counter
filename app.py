import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm

# --- MODEL DEFINITION ---
class MultiHeadCounterModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = timm.create_model(
            'tf_efficientnetv2_s',
            pretrained=False,
            num_classes=0
        )
        
        feat_dim = self.backbone.num_features
        
        def make_head():
            return nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1)
            )
        
        self.head_bolt = make_head()
        self.head_pin = make_head()
        self.head_nut = make_head()
        self.head_washer = make_head()
    
    def forward(self, x):
        feat = self.backbone(x)
        return torch.cat([
            self.head_bolt(feat),
            self.head_pin(feat),
            self.head_nut(feat),
            self.head_washer(feat),
        ], dim=1)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model = MultiHeadCounterModel()
    model.load_state_dict(
        torch.load('best_scratch_model.pth', map_location='cpu')
    )
    model.eval()
    return model

# --- PREPROCESSING ---
def preprocess_image(image):
    """Convert PIL Image to model input tensor"""
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Convert RGB to BGR if needed
    if img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    # Apply transformations
    transform = A.Compose([
        A.Resize(384, 384),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    transformed = transform(image=img_array)
    img_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
    
    return img_tensor

# --- PREDICTION ---
def predict(model, image_tensor):
    """Run inference and return counts"""
    with torch.no_grad():
        output = model(image_tensor)
        predictions = output.cpu().numpy()[0]
        
        # Round and clip to non-negative integers
        predictions = np.round(predictions)
        predictions = np.clip(predictions, 0, None).astype(int)
    
    return {
        'Bolts': int(predictions[0]),
        'Locating Pins': int(predictions[1]),
        'Nuts': int(predictions[2]),
        'Washers': int(predictions[3])
    }

# --- STREAMLIT UI ---
st.set_page_config(
    page_title="Hardware Parts Counter",
    page_icon="🔩",
    layout="centered"
)

st.title("🔩 Hardware Parts Counter")
st.markdown("""
Upload an image of hardware parts and the AI will count:
- **Bolts**
- **Locating Pins**
- **Nuts**
- **Washers**
""")

# Load model
with st.spinner("Loading model..."):
    model = load_model()

st.success("✅ Model loaded successfully!")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=['png', 'jpg', 'jpeg'],
    help="Upload a clear image of hardware parts"
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Uploaded Image")
        # FIX: Changed use_container_width to use_column_width
        st.image(image, use_column_width=True)
    
    # Make prediction
    with st.spinner("Counting parts..."):
        img_tensor = preprocess_image(image)
        results = predict(model, img_tensor)
    
    with col2:
        st.subheader("📊 Detection Results")
        
        # Display results with icons
        st.metric("🔩 Bolts", results['Bolts'])
        st.metric("📍 Locating Pins", results['Locating Pins'])
        st.metric("🔹 Nuts", results['Nuts'])
        st.metric("⭕ Washers", results['Washers'])
        
        # Total count
        total = sum(results.values())
        st.markdown(f"### Total Parts: **{total}**")
    
    # Download results as CSV
    st.divider()
    
    import pandas as pd
    df = pd.DataFrame([results])
    csv = df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="parts_count.csv",
        mime="text/csv"
    )

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>Powered by EfficientNetV2 | Built with Streamlit</small>
</div>
""", unsafe_allow_html=True)