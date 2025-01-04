import Tools

image, nom = Tools.Choix()
binary_image = Tools.Gaussian_Threshold(image)
color_image, _ = Tools.Components_detection(image, binary_image)
Tools.Display(color_image, nom)
