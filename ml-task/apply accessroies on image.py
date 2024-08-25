#apply sunglass and cap on a image
image_path = r"C:\Users\tannu\Desktop\python task_files\IMG_8820.jpg" 
image = Image.open(image_path)

sunglasses_path = r"C:\Users\tannu\Desktop\sunglass and cap\pngwing.com (1).png"
cap_path = r"C:\Users\tannu\Desktop\sunglass and cap\pngwing.com (2).png"

sunglasses = Image.open(sunglasses_path)
cap = Image.open(cap_path)

sunglasses = sunglasses.resize((300, 150))  
cap = cap.resize((300, 150))  

sunglasses_position = (500, 240) 
image.paste(sunglasses, sunglasses_position, sunglasses)
cap_position = (300, 250)  
image.paste(cap, cap_position, cap)

image.show()
image.save("image_with_accessories.jpg")