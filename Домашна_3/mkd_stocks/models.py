# from django.db import models
#
# # Create your models here.
# from django.db import models
#
# class Stock(models.Model):
#     issuer = models.CharField(max_length=255,primary_key=True)
#     date = models.CharField(max_length=255)  # Keep as text for now
#     last_price = models.CharField(max_length=255)  # Text, will be converted
#     max = models.CharField(max_length=255)
#     min = models.CharField(max_length=255)
#     average = models.CharField(max_length=255)
#     percent_change = models.CharField(max_length=255)
#     quantity = models.CharField(max_length=255)
#     best_trade = models.CharField(max_length=255)
#     total_trade = models.CharField(max_length=255)
#
#     class Meta:
#         db_table = 'stock_prices'


from django.db import models

class Stock(models.Model):
    issuer = models.CharField(max_length=255, primary_key=True)
    date = models.CharField(max_length=255)  # Оставено како CharField
    last_price = models.CharField(max_length=255)  # Оставено како CharField
    max = models.CharField(max_length=255)  # Оставено како CharField
    min = models.CharField(max_length=255)  # Оставено како CharField
    average = models.CharField(max_length=255)  # Оставено како CharField
    percent_change = models.CharField(max_length=255)  # Оставено како CharField
    quantity = models.CharField(max_length=255)  # Оставено како CharField
    best_trade = models.CharField(max_length=255)  # Оставено како CharField
    total_trade = models.CharField(max_length=255)  # Оставено како CharField

    class Meta:
        db_table = 'stock_prices'

