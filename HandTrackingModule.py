import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
import math


class HandDetector:
    """
    Finds Hands using the mediapipe library. Exports the landmarks
    in pixel format. Adds extra functionalities like finding how
    many fingers are up or the distance between two fingers. Also
    provides bounding box info of the hand found.
    """

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5): 
        """
        :param mode: In static mode, detection is done on each image: slower
        :param maxHands: Maximum number of hands to detect
        :param detectionCon: Minimum Detection Confidence Threshold
        :param minTrackCon: Minimum Tracking Confidence Threshold
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands # hands module
        self.hands = self.mpHands.Hands(model_complexity=0, static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils # drawing utilities module
        self.mp_drawing_styles = mp.solutions.drawing_styles # added by kamal
        # self.mp_hands = mp.solutions.hands
        self.tipIds = [4, 8, 12, 16, 20] # landmarks of the tips of the fingers
        self.fingers = [] # list of fingers
        self.lmList = [] # list of landmarks

    def findHands(self, img, draw=True, flipType=True):
        """
        Finds hands in a BGR image.
        :param img: Image to find the hands in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert BGR to RGB
        self.results = self.hands.process(imgRGB) # process the image
        allHands = [] # list of hands
        h, w, c = img.shape # height, width, channel of the image
        if self.results.multi_hand_landmarks: # if hands are detected
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks): # for each hand
                myHand = {} # dictionary of hand info
                ## lmList
                mylmList = [] # list of landmarks
                xList = [] # list of x coordinates
                yList = [] # list of y coordinates
                zList = [] # list of z coordinates
                for id, lm in enumerate(handLms.landmark): # for each landmark
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w) # x, y, z coordinates of the landmark
                    mylmList.append([px, py, pz]) # append to the list of landmarks
                    xList.append(px) # append to the list of x coordinates
                    yList.append(py) # append to the list of y coordinates
                    zList.append(pz) # append to the list of z coordinates

                ## bbox
                xmin, xmax = min(xList), max(xList) # min and max x coordinates of the hand
                ymin, ymax = min(yList), max(yList) # min and max y coordinates of the hand
                boxW, boxH = xmax - xmin, ymax - ymin # width and height of the hand bounding box 
                bbox = xmin, ymin, boxW, boxH # bounding box info
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2) # center x, center y of the hand bounding box 

                myHand["lmList"] = mylmList # list of landmarks
                myHand["bbox"] = bbox # bounding box info
                myHand["center"] = (cx, cy) # center of the hand bounding box

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label 
                allHands.append(myHand)

                ## draw
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS,
                                               mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style()) # draw landmarks and connections
                    # cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                    #               (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                    #               (35, 35, 129), 2)
                    cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (35, 35, 129), 2) # put text on the image 
        if draw:
            return allHands, img
        else:
            return allHands

    def fingersUp(self, myHand):
        """
        Finds how many fingers are open and returns in a list.
        Considers left and right hands separately
        :return: List of which fingers are up
        """
        myHandType = myHand["type"]
        myLmList = myHand["lmList"]
        if self.results.multi_hand_landmarks:
            fingers = []
            # Thumb
            if myHandType == "Right":
                if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]: # if x coordinate of the thumb tip is greater than the x 
                                                                                  # coordinate of the thumb IP (index proximal) landmark
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if myLmList[self.tipIds[0]][0] < myLmList[self.tipIds[0] - 1][0]: # if x coordinate of the thumb tip is less than the x
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if myLmList[self.tipIds[id]][1] < myLmList[self.tipIds[id] - 2][1]: # if y coordinate of the finger tip is less than the y
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img=None):
        """
        Find the distance between two landmarks based on their
        index numbers.
        :param p1: Point1
        :param p2: Point2
        :param img: Image to draw on.
        :param draw: Flag to draw the output on the image.
        :return: Distance between the points
                 Image with output drawn
                 Line information
        """

        x1, y1, i1 = p1 # x, y, index 
        x2, y2, i2 = p2 # x, y, index 
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 # center x, center y
        length = math.hypot(x2 - x1, y2 - y1) # distance between two points
        info = (x1, y1, x2, y2, cx, cy) 
        if img is not None:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED) # draw circle
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED) # draw circle
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3) # draw line
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED) # draw circle
            return length, info, img 
        else:
            return length, info


def main():
    cap = cv2.VideoCapture(0) # camera id
    detector = HandDetector(detectionCon=0.8, maxHands=2) # detection confidence, max hands to detect, 
                                                          # confidence means how sure the model is that it has detected a hand 
    while True:
        # Get image frame
        success, img = cap.read() # success is a boolean, img is the image frame
        # Find the hand and its landmarks
        hands, img = detector.findHands(img)
        # hands = detector.findHands(img, draw=False)  # without draw

        if hands:
            # Hand 1
            hand1 = hands[0] # list of hands
            lmList1 = hand1["lmList"]  # List of 21 Landmark points
            bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
            centerPoint1 = hand1['center']  # center of the hand cx,cy
            handType1 = hand1["type"]  # Handtype Left or Right

            fingers1 = detector.fingersUp(hand1) # list of fingers

            if len(hands) == 2: # if two hands are detected
                # Hand 2
                hand2 = hands[1] 
                lmList2 = hand2["lmList"]  # List of 21 Landmark points
                bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
                centerPoint2 = hand2['center']  # center of the hand cx,cy
                handType2 = hand2["type"]  # Hand Type "Left" or "Right"

                fingers2 = detector.fingersUp(hand2)

                # Find Distance between two Landmarks. Could be same hand or different hands
                length, info, img = detector.findDistance(lmList1[8][0:2], lmList2[8][0:2], img)  # with draw
                # length, info = detector.findDistance(lmList1[8], lmList2[8])  # with draw
        # Display
        cv2.imshow("Image", img)
        cv2.waitKey(1)

