import Tools
import cv2


def start():
    image, nom = Tools.Choix()
    inp = input("1 - Threshold \n2 - Gaussian \n3 - Mean_Threshold\n")
    if inp == "1":
        binary_image = Tools.Threshold(image)
    elif inp == "2":
        binary_image = Tools.Gaussian_Threshold(image)
    elif inp == "3":
        binary_image = Tools.Mean_Threshold(image)
    binary_image = Tools.Mean_Threshold(image)
    color_image, _ = Tools.Components_detection(image, binary_image, nom)
    return nom, _
