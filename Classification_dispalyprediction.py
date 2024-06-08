def predict_and_display(image_tensor):
    image_tensor = image_tensor.cuda()
    with torch.no_grad():
        output = model(image_tensor)
    
    # Get the top-5 predictions
    probs = torch.nn.functional.softmax(output, dim=1)
    top_5_probs, top_5_indices = probs.topk(5, dim=1)
    
    # Translate indices into class names
    top_5_labels = [label_map[idx.item()] for idx in top_5_indices[0]]
    top_5_probs = top_5_probs[0].cpu().numpy()

    return top_5_labels, top_5_probs

def display_predictions(image, top_5_labels, top_5_probs, true_label):
    plt.imshow(image)
    plt.title(f"True Label: {true_label}\nTop-5 Predictions:")
    plt.axis("off")
    plt.show()

    # Display the top-5 predictions with probabilities
    for label, prob in zip(top_5_labels, top_5_probs):
        print(f"{label}: {prob * 100:.2f}%")

# Dictionary to keep track of how many images per label have been displayed
label_count = defaultdict(int)

# Maximum number of images per label to display
max_images_per_label = 4

# Total number of labels
num_labels = 100

# Predict and display results for images in the test set
for i, (image, label) in enumerate(test_loader):
    true_label = label_map[label.item()]
    
    # Check if the max number of images for this label has been reached
    if label_count[true_label] >= max_images_per_label:
        continue
    
    print(f"\nImage {i + 1}:")

    # Add batch dimension if necessary
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    
    top_5_labels, top_5_probs = predict_and_display(image)
    
    # Convert the tensor back to a PIL image for display
    img = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # Unnormalize
    img = (img * 255).astype(np.uint8)
    
    display_predictions(img, top_5_labels, top_5_probs, true_label)  # Pass the true label
    
    # Increment the count for the current label
    label_count[true_label] += 1

    # Check if the max number of images per label has been reached for all labels
    if len(label_count) >= num_labels and all(count >= max_images_per_label for count in label_count.values()):
        break