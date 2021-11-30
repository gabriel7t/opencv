import cv2
faceClassifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
mouthClassifier = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_mouth.xml')
capture = cv2.VideoCapture(0)

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
capture.set(cv2.CAP_PROP_FRAME_COUNT, 15)

while not cv2.waitKey(20) & 0xFF == ord("q"):
    ret, frame_color = capture.read()
    gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
    faces = faceClassifier.detectMultiScale(gray)
    for x, y, w, h in faces:
        cmask = cv2.rectangle(frame_color, (x, y), (x + w, y + h), (0, 255, 0), 10)
        cv2.putText(cmask, 'Com Mascara', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 3)
        mouth_rects = mouthClassifier.detectMultiScale(gray, 1.5, 5)

        for (mx, my, mw, mh) in mouth_rects:
            if(y < my < y + h):
                smask = cv2.rectangle(frame_color, (x, y), (x + w, y + h), (0, 0, 255), 10)
                cv2.rectangle(frame_color, (mx, my), (mx + mw, my + mh), (0, 0, 0), 0)
                cv2.putText(smask, 'Sem Mascara', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

                break

    cv2.imshow('Preview', frame_color)

capture.release()
cv2.destroyAllWindows()