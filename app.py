import numpy as np
from sklearn.linear_model import LinearRegression
import gradio as gr

# 1. Dataset: House Size vs House Price (in Lakhs)
X_size = np.array([[500], [700], [1000], [1200], [1500], [2000]])
y_price = np.array([25, 35, 50, 60, 75, 100])

# 2. Train the Linear Regression Model (y = wx + b)
model = LinearRegression()
model.fit(X_size, y_price)

# 3. Prediction Logic Function
def predict_price(house_size):
    input_data = np.array([[house_size]])
    prediction = model.predict(input_data)
    final_price = round(prediction[0], 2) # Extracts the raw numeric value
    return f"Estimated Valuation: ₹{final_price} Lakhs"

# 4. Construct the Simple Gradio User Interface
interface = gr.Interface(
    fn=predict_price,
    inputs=gr.Slider(500, 2000, value=1000, label="House Area (Square Feet)"),
    outputs=gr.Textbox(label="AI Predicted Price"),
    title="🏠 Simple House Price Predictor",
    description="Move the slider to predict house prices using a simple Linear Regression model."
)

interface.launch()

