import cv2
import numpy as np
import os

# Membuat directory untuk menyimpan gambar
output_dir = 'output_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

# Memuat detektor wajah (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Fungsi untuk menyimpan gambar
def save_images(original, grayscale, bw, cropped_face):
    cv2.imwrite(os.path.join(output_dir, 'original.jpg'), original)
    cv2.imwrite(os.path.join(output_dir, 'grayscale.jpg'), grayscale)
    cv2.imwrite(os.path.join(output_dir, 'blackwhite.jpg'), bw)
    cv2.imwrite(os.path.join(output_dir, 'cropped_face.jpg'), cropped_face)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah dalam gambar
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        # Ambil wajah pertama yang terdeteksi (untuk contoh ini hanya satu wajah)
        (x, y, w, h) = faces[0]

        # Crop wajah
        cropped_face = frame[y:y+h, x:x+w]

        # Ubah gambar asli menjadi grayscale
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Ubah gambar grayscale menjadi black and white
        _, bw = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY)

        # Simpan hasil gambar
        save_images(frame, grayscale, bw, cropped_face)
        
        print("Images saved in the folder 'output_images'")

        # Tampilkan gambar dengan wajah yang ter-crop
        cv2.imshow("Cropped Face", cropped_face)
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Grayscale Frame", grayscale)
        cv2.imshow("Black and White Frame", bw)
        
        # Tunggu input dari keyboard untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("No face detected")
        cv2.imshow("Original Frame", frame)
    
# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
