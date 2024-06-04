
import requests

ip = 'localhost'
port = 8000
url = f"http://{ip}:{port}/"

def test_model_prediction(url):
    # Read the image file
    image_path = 'fashion_mnist_images/image_9_categoria_7.jpg'
    with open(image_path, "rb") as file:
        image_data = file.read()

    # Define the URL of the FastAPI service
    service = "predict"
    url = url + service

    # Define the payload (you can add additional data if needed)
    payload = {"key": "value"}

    # Send a POST request with the image data
    response = requests.post(url, files={"input_image": image_data}, data=payload).json()
    print(response)
    try:
        assert isinstance(response, dict), "Error al calcular la prediccion"
        assert "prediction" in response, "Error al calcular la prediccion"
    except AssertionError as e:
        print("❌An error occurred:", e)
    else:
        print(f"✅No errors occurred in service {service}.")

test_model_prediction(url)