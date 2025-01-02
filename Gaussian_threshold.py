import Tools
import cv2

image, nom = Tools.Choix()
result = Tools.Gaussian_Threshold(image)
resized = Tools.resize_with_aspect_ratio(result)
cv2.imshow("Gaussian_Threshold", resized)
cv2.waitKey(0)
print(f"Ecriture dans Resultats/Gaussian.{nom.split('/',2)[2]}...")
cv2.imwrite(f'Resultats/Gaussian.{nom.split('/',2)[2]}', result)
cv2.destroyAllWindows()
