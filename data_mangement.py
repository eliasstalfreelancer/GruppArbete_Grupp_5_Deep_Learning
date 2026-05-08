import os
import keras
from PIL import Image

def data_import():

    if  os.path.exists("data/fashion_mnist_images"):
        print("Data has aldready been imported")
        return
    #  Ladda dataset
    (x_train, y_train) = keras.datasets.fashion_mnist.load_data()

    # Klassnamn (viktigt för mappar)
    class_names = [
        "T-shirt_top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle_boot"
    ]

    # Root folder
    base_path = "data/fashion_mnist_images"

    # Skapa mappar per klass
    for class_name in class_names:
        os.makedirs(os.path.join(base_path, class_name), exist_ok=True)

    # Spara träningsbilder
    for i, (img, label) in enumerate(zip(x_train, y_train)):
        class_name = class_names[label]
        path = os.path.join(base_path, class_name, f"{i}.png")
        
        image = Image.fromarray(img)  # numpy → bild
        image.save(path)

    print("Done with importing data")