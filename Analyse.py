import Tools

image, nom = Tools.Choix()
input = input("1 - Threshold \n2 - Gaussian \n3 - Mean_Threshold\n")
if input == "1":
    binary_image = Tools.Threshold(image)
elif input == "2":
    binary_image = Tools.Gaussian_Threshold(image)
elif input == "3":
    binary_image = Tools.Mean_Threshold(image)
binary_image = Tools.Mean_Threshold(image)
color_image, _ = Tools.Components_detection(image, binary_image, nom)
Tools.Display(color_image, nom)
