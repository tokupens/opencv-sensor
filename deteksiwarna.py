import cv2
import numpy as np

# Fungsi callback untuk menangani perubahan warna pada trackbar
def on_change(value):
    pass

# Inisialisasi jendela dan trackbar
cv2.namedWindow('Color Detector')
cv2.createTrackbar('Hue Lower', 'Color Detector', 0, 180, on_change)
cv2.createTrackbar('Hue Upper', 'Color Detector', 0, 180, on_change)
cv2.createTrackbar('Saturation Lower', 'Color Detector', 0, 255, on_change)
cv2.createTrackbar('Saturation Upper', 'Color Detector', 0, 255, on_change)
cv2.createTrackbar('Value Lower', 'Color Detector', 0, 255, on_change)
cv2.createTrackbar('Value Upper', 'Color Detector', 0, 255, on_change)

# Open video capture (0 for default camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Konversi frame ke format HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Ambil nilai trackbar
    hue_lower = cv2.getTrackbarPos('Hue Lower', 'Color Detector')
    hue_upper = cv2.getTrackbarPos('Hue Upper', 'Color Detector')
    sat_lower = cv2.getTrackbarPos('Saturation Lower', 'Color Detector')
    sat_upper = cv2.getTrackbarPos('Saturation Upper', 'Color Detector')
    val_lower = cv2.getTrackbarPos('Value Lower', 'Color Detector')
    val_upper = cv2.getTrackbarPos('Value Upper', 'Color Detector')

    # Tentukan batas warna dalam format HSV
    lower_bound = np.array([hue_lower, sat_lower, val_lower])
    upper_bound = np.array([hue_upper, sat_upper, val_upper])

    # Buat mask untuk deteksi warna
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Aplikasikan mask ke frame asli
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Tampilkan hasil
    cv2.imshow('Color Detector', np.hstack([frame, result]))

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
