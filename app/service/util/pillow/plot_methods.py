from PIL import Image


def plot_image(file_name):
    image = Image.open(file_name)
    image.show()
