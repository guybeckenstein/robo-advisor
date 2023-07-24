from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required


def homepage(request):
    return render(request, 'form/index.html', {})


def about(request):
    return render(request, 'form/about.html', {'title': 'About Us'})


def contact(request):
    return render(request, 'form/contact.html', {'title': 'Contact Us'})


#@login_required#TODO: fix register and login and then remove this
def services(request):
    return render(request, 'form/services.html', {'title': 'Services'})

#@login_required#TODO: fix register and login and then remove this
def createPortfolio(request):
    if request.method == 'POST':
        investment_amount = request.POST.get('user_answer_1')
        stats_model = request.POST.get('user_answer_2')
        machine_learning = request.POST.get('user_answer_3')
        # Redirect to the "form" view and pass the data as query parameters
        return redirect('form', investment_amount=investment_amount, stats_model=stats_model, machine_learning=machine_learning)
    return render(request, 'form/createPortfolio.html', {'title': 'createPortfolio'})

#@login_required#TODO: fix register and login and then remove this
def form(request):
    # Retrieve the data from the query parameters
    investment_amount = request.GET.get('investment_amount')
    stats_model = request.GET.get('stats_model')
    machine_learning = request.GET.get('machine_learning')

    if request.method == 'POST':
        # Process the form data here
        user_answer_1 = request.POST.get('user_answer_1')
        user_answer_2 = request.POST.get('user_answer_2')
        user_answer_3 = request.POST.get('user_answer_3')

        # Perform any necessary operations with the data

        # Render the success or result page
        return render(request, 'form/success.html',
                      {'user_answer_1': user_answer_1, 'user_answer_2': user_answer_2, 'user_answer_3': user_answer_3})

    return render(request, 'form/form.html', {'title': 'Form', 'investment_amount': investment_amount, 'stats_model': stats_model, 'machine_learning': machine_learning})
