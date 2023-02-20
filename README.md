# movingObjectDetection
import cv2

# Load the Haar Cascade for cars
car_cascade = cv2.CascadeClassifier('HaarCascade_car.xml')

# Open the video file
cap = cv2.VideoCapture('video_file.mp4')

while True:
# Read a frame from the video
    ret, frame = cap.read()

# Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect cars in the frame
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Draw a rectangle around each car
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Display the resulting frame
    cv2.imshow('frame', frame)

# Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
import cv2

# Load the Haar Cascade for license plates
plate_cascade = cv2.CascadeClassifier('HaarCascade_russian_plate_number.xml')

# Load the image of a car with a license plate
img = cv2.imread('car_with_plate.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect license plates in the image
plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Draw a rectangle around each license plate
for (x, y, w, h) in plates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Display the resulting image
cv2.imshow('img', img)
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()
