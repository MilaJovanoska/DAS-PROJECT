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
    date = models.DateField()  # Change to DateField for actual date format
    last_price = models.DecimalField(max_digits=10, decimal_places=2)  # Use DecimalField for prices
    max = models.DecimalField(max_digits=10, decimal_places=2)  # Use DecimalField for prices
    min = models.DecimalField(max_digits=10, decimal_places=2)  # Use DecimalField for prices
    average = models.DecimalField(max_digits=10, decimal_places=2)  # Use DecimalField for average
    percent_change = models.DecimalField(max_digits=5, decimal_places=2)  # Use DecimalField for percentages
    quantity = models.IntegerField()  # Use IntegerField for quantity
    best_trade = models.DecimalField(max_digits=10, decimal_places=2)  # Use DecimalField for trade amounts
    total_trade = models.DecimalField(max_digits=15, decimal_places=2)  # Use DecimalField for total trade

    class Meta:
        db_table = 'stock_prices'

