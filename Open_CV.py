import cv2

# Open the default camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

print("Camera opened successfully.")
print("Press 'q' to quit.")

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture a frame.")
        break

    # Display the frame
    cv2.imshow("Webcam Feed", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()





import cv2
from ultralytics import YOLO

# Load pretrained YOLO model
model= YOLO("yolov8n.pt")

cap=cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("camera is open")
    exit()
    
while True:
    ret, frame= cap.read()
    if not ret:
        break
    
    # Run object detection
    results = model(frame)

    # Draw results on frame
    annotated_frame= results[0].plot()
    
    cv2.imshow("Object Detection", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()



#************************* Webcam Viewer ***********************************

import cv2

cap =cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    ret,frame=cap.read()
    if not ret:
        break
    cv2.imshow("webcam",frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()

# ************************ Color Object Detector*****************************

import cv2
import numpy as np

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame=cap.read()
    if not ret:
        break
    
    # -----------------------------
    # Convert BGR to HSV
    # -----------------------------
    hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # -----------------------------
    # Threshold selected color
    # Example: Red color
    # -----------------------------
    lower_red1=np.array([0, 0, 200])
    upper_red1=np.array([180, 30, 255])
    
    lower_red2=np.array([170,0, 200])
    upper_red2=np.array([180,30, 255])
    
    mask1=cv2.inRange(hsv, lower_red1, upper_red1)
    mask2=cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Binary mask
    mask=mask1 | mask2
    
    # -----------------------------
    # Find contours
    # ----------------------------
    countours, _ =cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # -----------------------------
    # Draw bounding boxes
    # -----------------------------
    for countour in countours:
        area=cv2.contourArea(countour)
        if area>500:
            x,y,w,h=cv2.boundingRect(countour)
            
            # Draw green rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Label
            cv2.putText(
                frame,
                "white object",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 6),
                2,
            )
    # Display results
    cv2.imshow("Original", frame)
    cv2.imshow("Binary Mask", mask)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
# Release resources
cap.release()
cv2.destroyAllWindows()        
             
             
             
#  *********************QR Code Scanner***********************

import cv2

# Initialize webcam (you can also replace 0 with image path)
cap = cv2.VideoCapture(0)

# Create QR detector object
detector = cv2.QRCodeDetector()

while True:
    ret,frame = cap.read()
    if not ret:
        break
    # -----------------------------
    # Step 1: Find QR corners + decode
    # -----------------------------
    data,bbox, _=detector.detectAndDecode(frame)
    
    # -----------------------------
    # Step 2: If QR detected
    # -----------------------------
    if bbox is not None:
        # Convert bbox to integer
        bbox = bbox.astype(int)
        
        
        # Draw bounding box (corners)
        for i in range(len(bbox[0])):
            pt1 = tuple(bbox[0][i])
            pt2 = tuple(bbox[0][(i + 1) % len(bbox[0])])
            cv2.line(frame, pt1, pt2, (0, 255, 0), 3)
            
        
        # Put decoded text
        if data:
            cv2.putText(
                frame,
                data,
                (bbox[0][0][0], bbox[0][0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
            )

            print("Decoded QR Data:", data)
            
     # -----------------------------
    # Show result
    # -----------------------------
    cv2.imshow("QR Scanner", frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    

# Release resources
cap.release()
cv2.destroyAllWindows()

# ****************************Face Detector *******************************
import cv2

# Load the pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Draw a rectangle around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Face",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Show the result
    cv2.imshow("Face Detector", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
# Clean up
cap.release()
cv2.destroyAllWindows()



import cv2
import numpy as np

def order_position(pts):
    # Order points: top-left, top-right, bottom-right, bottom-left
    rect=np.zeros((4,2), dtype="float32")
    
    s=pts.sum(axis=1)
    rect[0]=pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)] 
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    
    return rect

def four_point_transform(image, pts):
    rect= order_position(pts)
    (tl, tr, br, bl)=rect
    
    widthA= np.linalg.norm(br-bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 600))
    orig = frame.copy()
    
    # 1. Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Blur to remove noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Edge detection
    edges = cv2.Canny(blur, 50, 150)
    
    # 4. Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    screenCnt = None
    
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # If document has 4 corners
        if len(approx) == 4:
            screenCnt = approx
            break
    
    # 5. Draw contour + warp perspective
    if screenCnt is not None:
        cv2.drawContours(frame, [screenCnt], -1, (0, 255, 0), 2)

        warped = four_point_transform(orig, screenCnt.reshape(4, 2))

        cv2.imshow("Scanned Document", warped)

    # Show original feed
    cv2.imshow("Camera", frame)
    cv2.imshow("Edges", edges)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()