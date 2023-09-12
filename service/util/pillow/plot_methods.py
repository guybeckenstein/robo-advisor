def plot_image(file_name: str):
    from PIL import Image

    image = Image.open(file_name)
    image.show()
