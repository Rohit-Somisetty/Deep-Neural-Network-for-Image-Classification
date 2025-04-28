import torch
from main import CNN, predict_image
from pathlib import Path

def main():
    # Load the trained model
    model = CNN()
    model.load_state_dict(torch.load('cats_vs_dogs_cnn.pth'))
    
    # Example: Predict on a test image
    test_image_path = 'test/cats/cat.1.jpg'  # Replace with your image path
    if Path(test_image_path).exists():
        prediction, confidence = predict_image(model, test_image_path)
        print(f"\nPrediction for {test_image_path}:")
        print(f"This image is a {prediction} with {confidence:.2%} confidence")
    else:
        print(f"Error: Image file not found at {test_image_path}")
        print("Please provide a valid image path")

if __name__ == '__main__':
    main() 