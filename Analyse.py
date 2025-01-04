import Tools
import cv2

image, nom = Tools.Choix()
binary_image = Tools.Gaussian_Threshold(image)
color_image, _ = Tools.Components_detection(image, binary_image)

for i in range(10):
    roi, (x, y, w, h) = _[i]
    cv2.imshow("CC", roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# Tools.Display(color_image, nom)
