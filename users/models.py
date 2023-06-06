from django.contrib.auth.models import User
from django.db import models
from PIL import Image

IMG_SIZE = 256


class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    image = models.ImageField(default='default.jpg', upload_to='profile_pics')

    class Meta:
        db_table = 'Profile'

    def __str__(self):
        return f"{self.user.username}'s Profile"

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)

        img = Image.open(self.image.path)
        if (img.height > IMG_SIZE) or (img.width > IMG_SIZE):
            output_size = (IMG_SIZE, IMG_SIZE)
            img.thumbnail(output_size)
            img.save(self.image.path)
