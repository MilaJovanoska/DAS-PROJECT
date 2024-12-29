from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='index.html'),
    path("about_us/", views.about_us, name='about_us.html'),
    path('news/', views.news, name='news.html'),
    path('sign_up/', views.sign_up, name='sign_up.html'),
    path('contact/', views.contact, name='contact.html'),
    path('stocks/', views.stock_list, name='stock_list'),
    path('login/', views.login_view, name='log_in.html'),
    path('technical-analysis/', views.technical_analysis, name='technical_analysis.html'),
    path('get-indicators/', views.get_indicators, name='get_indicators'),
    path('nlp-analysis/', views.nlp_analysis, name='nlp_analysis.html'),
    path('company_view/', views.company_view, name='company_view'),  # Додај ја рутата
    path('lstm-analysis/', views.lstm_analysis, name='lstm_analysis'),
    path('get-stock-data/', views.get_stock_data, name='get_stock_data'),
    # path('prepare-data/', views.prepare_data_for_training, name='prepare_data'),
    path('train-model/<str:issuer_name>/', views.train_model_view, name='train_model'),
    path('predict/<str:issuer_name>/', views.predict_stock_prices, name='predict')

]
