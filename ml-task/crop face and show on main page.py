import cvlib as cv
from cvlib.object_detection import draw_bbox

def capture_image():
    cam = cv2.VideoCapture(0) 
    cv2.namedWindow("Capture Image")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow("Capture Image", frame)
        k = cv2.waitKey(1)
        if k % 256 == 27:
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            img_name = "captured_image.png"
            cv2.imwrite(img_name, frame)
            print(f"{img_name} saved!")
            break

    cam.release()
    cv2.destroyAllWindows()
    return img_name

def crop_face(image_path):
    img = cv2.imread(image_path)
    faces, confidences = cv.detect_face(img)
    if faces:
        for face in faces:
            (startX, startY) = face[0], face[1]
            (endX, endY) = face[2], face[3]
            cropped_face = img[startY:endY, startX:endX]
            
            cv2.imshow("Cropped Face", cropped_face)
            cropped_face_path = "cropped_face.png"
            cv2.imwrite(cropped_face_path, cropped_face)
            print(f"Cropped face saved as {cropped_face_path}")
            
            img[startY:endY, startX:endX] = cropped_face
            cv2.imshow("Image with Cropped Face", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return cropped_face_path
    else:
        print("No face detected")
        return None

if __name__ == "__main__":
    image_path = capture_image()
    if image_path:
        cropped_face_path = crop_face(image_path)
