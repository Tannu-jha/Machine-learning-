#create own custom image using numpy
width, height = 300, 300

# Create a 3D NumPy array of shape (height, width, 3) for an RGB image
image_array = np.zeros((height, width, 3), dtype=np.uint8)
for y in range(height):
    for x in range(width):
        image_array[y, x] = [x % 256, y % 256, (x + y) % 256]
image = Image.fromarray(image_array)
image.show()
image.save("custom_image.png")