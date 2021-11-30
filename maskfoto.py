import cv2

Face = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
Boca = cv2.CascadeClassifier("haarcascades/haarcascade_mcs_mouth.xml")
imagem = cv2.imread('fotos/imagem4.jpg')

imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
facesDetectadas = Face.detectMultiScale(imagemCinza)

for (x, y, l, a) in facesDetectadas:
    rosto = cv2.rectangle(imagem, (x, y), (x + l, y+ a), (0, 255, 0), 0)

    LocalBoca = rosto[y:y+a, x:x+l]
    LocalBocaCinza = cv2.cvtCor(LocalBoca, cv2.COLOR_BGR2GRAY)
    BocasDetectadas = Boca.detectMultiScale(LocalBocaCinza)
    for (ex,ey,ew,eh) in BocasDetectadas:
        if ey > (a/2):
            print("Sem mascara")
            cv2.rectangle(LocalBoca,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            cv2.rectangle(rosto, (x, y), (x + l, y+ a), (255, 255, 255), 2)
        else:
            print("Com mascara")
            cv2.rectangle(rosto, (x, y), (x + l, y+ a), (255,191,0), 2)

    cv2.imshow('Frame', imagem)

while True:
    if cv2.waitKey(1) == ord('s'):
        cv2.destroyAllWindows()
        exit()