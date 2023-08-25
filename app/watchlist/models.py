from django.db import models


class TopStock(models.Model):
    id = models.BigAutoField(primary_key=True, verbose_name="ID")
    sector_name = models.CharField(max_length=70)
    img_src = models.CharField(max_length=130)
    img_alt = models.CharField(max_length=50)

    class Meta:
        app_label = 'watchlist'
        db_table = 'TopStock'
        ordering = ['sector_name']
        verbose_name = 'Top Stock'
        verbose_name_plural = 'Top Stock'
