import numpy as np
from sklearn.linear_model import LinearRegression
import gradio as gr

#House Size vs House Price (in Lakhs)
X_size = np.array([[500], [700], [1000], [1200], [1500], [2000]])
y_price = np.array([25, 35, 50, 60, 75, 100])

model = LinearRegression()
model.fit(X_size, y_price)

def predict_price(house_size):
    input_data = np.array([[house_size]])
    prediction = model.predict(input_data)
    final_price = round(prediction[0], 2)
    return f"Estimated Valuation: ₹{final_price} Lakhs"

interface = gr.Interface(
    fn=predict_price,
    inputs=gr.Slider(500, 2000, value=1000, label="House Area (Square Feet)"),
    outputs=gr.Textbox(label="AI Predicted Price"),
    title="Simple House Price Predictor",
    description="Move the slider to predict house prices using a simple Linear Regression model."
)

interface.launch()

