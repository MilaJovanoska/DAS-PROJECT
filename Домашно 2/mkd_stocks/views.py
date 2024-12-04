# Create your views here.
def home(request):
    return render(request, 'mkd_stocks/index.html')
def about_us(request):
    return render(request, 'mkd_stocks/about_us.html')
def sign_up(request):
    return render(request, 'mkd_stocks/sign_up.html')
def news(request):
    return render(request, 'mkd_stocks/news.html')
def contact(request):
    return render(request, 'mkd_stocks/contact.html')


# from django.shortcuts import render
# from .models import Stock
#
# def stock_list(request):
#     query = request.GET.get('query', '')  # Get the search query from the request
#     if query:
#         stocks = Stock.objects.filter(issuer__icontains=query)[:15]  # Filter by issuer (case-insensitive), limit to 15 rows
#     else:
#         stocks = Stock.objects.all()[:15]  # Default to the first 15 rows if no query is provided
#     return render(request, 'mkd_stocks/stock_data.html', {'stocks': stocks, 'query': query})
# from django.shortcuts import render
# from .models import Stock
#
# def stock_list(request):
#     query = request.GET.get('query', '')  # Get the issuer name from the search bar
#     table_data = []
#     graph_data = []
#
#     if query:
#         # Fetch only the first 15 rows for the table
#         table_data = Stock.objects.filter(issuer__icontains=query).order_by('date')[:15]
#         # Fetch all rows for the graph
#         graph_data = Stock.objects.filter(issuer__icontains=query).order_by('date')
#     else:
#         # Show default empty table and graph if no query
#         table_data = Stock.objects.all()[:15]
#         graph_data = []
#
#     return render(request, 'mkd_stocks/stock_data.html', {
#         'stocks': table_data,   # Data for the table
#         'graph_data': graph_data,  # Full data for the graph
#         'query': query
#     })
from django.shortcuts import render
from .models import Stock

def stock_list(request):
    query = request.GET.get('query', '')  # Get the issuer name from the search bar
    table_data = []
    graph_data = []

    if query:
        # Fetch only the first 15 rows for the table
        table_data = Stock.objects.filter(issuer__icontains=query).order_by('date')[:15]
        # Fetch all rows for the graph and sample every 5th row
        all_graph_data = Stock.objects.filter(issuer__icontains=query).order_by('date')
        graph_data = all_graph_data[::5]  # Include every 5th row
    else:
        # Show default empty table and graph if no query
        table_data = Stock.objects.all()[:15]
        graph_data = []

    return render(request, 'mkd_stocks/stock_data.html', {
        'stocks': table_data,   # Data for the table
        'graph_data': graph_data,  # Sampled data for the graph
        'query': query
    })


