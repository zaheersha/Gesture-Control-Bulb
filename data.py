import cv2
import mediapipe as mp 
import numpy as np #numerical python array

mp_drawing = mp.solutions.drawing_utils #hand landmarks and points joining line
mp_drawing_styles = mp.solutions.drawing_styles #To get hand image nicely
mp_hands = mp.solutions.hands #can use face, iris etc

#fun to store hand landmarks
def storeData(f,a):
  fo=open(f,'a')
  fo.write(str(a))
  fo.close()

cap = cv2.VideoCapture(0) #camera ON


with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5, #50% hand detected
    min_tracking_confidence=0.5) as hands:  #50% tracking
  while cap.isOpened():
    success, image = cap.read()
    imageWidth, imageHeight=image.shape[:2] #camera window size
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    #Camera will give us image in BGR, so we need convert into RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    results = hands.process(image) #Checking whether hands are there.

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    #Again converting to RGB to BGR.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks: 
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(), #drawing landmarks
            mp_drawing_styles.get_default_hand_connections_style()) #Connecting
        data=[]
        for point in mp_hands.HandLandmark:
 
              normalizedLandmark = hand_landmarks.landmark[point]
              pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, imageWidth, imageHeight)
              
              data.append(normalizedLandmark.x)
              data.append(normalizedLandmark.y)
              data.append(normalizedLandmark.z)
        print(len(data)) #63 points captured.
        data=str(data)
        data=data[1:-1]
        #image=cv2.flip(image,1)
        storeData('Gesture2.csv',data+', Gesture2\n')                  #Gesture2.csv and Gesture1.csv  should change as per the gestures
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release() 