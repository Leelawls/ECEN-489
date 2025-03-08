import pygame
import numpy as np
import torch
import torchvision.transforms as transforms
from cpu_train import CNN  # Import trained model
from PIL import Image

# Initialize Pygame
pygame.init()

# Window settings
WIDTH, HEIGHT = 300, 300
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Real-Time Digit Recognition")
screen.fill(WHITE)  # Set background to white
drawing = False
predicted_digit = None  # Store the last predicted digit

# Load trained model
model = CNN()
model.load_state_dict(torch.load("cpu_cnn_model.pth"))
model.eval()

# Define image transformation (same as MNIST preprocessing)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict_digit():
    """Captures the canvas, processes it, and updates the prediction."""
    global predicted_digit

    # Save and process the drawn image
    pygame.image.save(screen, "digit.png")
    img = Image.open("digit.png").convert("L")  # Convert to grayscale
    img = transform(img).unsqueeze(0)  # Apply transformations

    # Predict using CNN
    with torch.no_grad():
        output = model(img)
        predicted_digit = torch.argmax(output, dim=1).item()

def draw_prediction():
    """Displays the predicted digit on the screen without overlap."""
    font = pygame.font.Font(None, 50)
    text_surface = font.render(f"Prediction: {predicted_digit}", True, BLACK)
    text_rect = text_surface.get_rect(center=(WIDTH // 2, 20))

    # Create a rectangle to erase old text
    pygame.draw.rect(screen, WHITE, (0, 0, WIDTH, 40))
    
    # Blit the new prediction
    screen.blit(text_surface, text_rect)

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Start drawing when mouse is pressed
        if event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True

        # Stop drawing when mouse is released
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False

        # Clear screen when 'C' is pressed
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                screen.fill(WHITE)  # Clear canvas
                predicted_digit = None  # Reset prediction

    # Draw when mouse moves while clicked
    if drawing:
        pygame.draw.circle(screen, BLACK, pygame.mouse.get_pos(), 10)
        predict_digit()  # Predict while drawing

    # Display prediction on the canvas (no overlap)
    if predicted_digit is not None:
        draw_prediction()

    pygame.display.flip()  # Update screen

pygame.quit()
