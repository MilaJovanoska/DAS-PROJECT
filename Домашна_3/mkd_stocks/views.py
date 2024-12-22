# Create your views here.
def home(request):
    return render(request, 'index.html')
def about_us(request):
    return render(request, 'about_us.html')
def sign_up(request):
    return render(request, 'sign_up.html')
def news(request):
    return render(request, 'stock_data.html')
def contact(request):
    return render(request, 'contact.html')

def technical_analysis(request):
    return render(request, 'technical_analysis.html')

def nlp_analysis(request):
    return render(request, 'nlp_analysis.html')

def lstm_analysis(request):
    return render(request, 'lstm_analysis.html')

from django.shortcuts import render
from django.contrib import messages
from .models import Stock
from datetime import datetime

def stock_list(request):
    query = request.GET.get('query', '').strip()
    date = request.GET.get('date', '').strip()
    table_data = []
    graph_data = []

    original_date = date


    if date:
        try:

            date = datetime.strptime(date, '%Y-%m-%d').strftime('%d.%m.%Y')

            day, month, year = date.split('.')
            date = f"{day}.{int(month)}.{year}"
        except ValueError:
            messages.error(request, "Invalid date format. Please use a valid date.")
            date = ''


    if date and query:

        table_data = Stock.objects.filter(issuer__icontains=query, date=date).order_by('date')
        graph_data = table_data[:10]
    elif query:

        table_data = Stock.objects.filter(issuer__icontains=query).order_by('-date')[:15]
        all_graph_data = Stock.objects.filter(issuer__icontains=query).order_by('-date')
        graph_data = all_graph_data[:10]
    elif date:

        messages.error(request, "Please provide an issuer along with the date.")
    else:

        table_data = Stock.objects.all().order_by('-date')[:15]


    return render(request, 'stock_data.html', {
        'stocks': table_data,
        'graph_data': graph_data,
        'query': query,
        'date': original_date
    })

def login_view(request):
    return render(request, 'log_in.html')


















