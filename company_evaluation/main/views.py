from django.shortcuts import render

from .forms import PredictPostForm
from .predictors import predict


def home(request):
    if request.method == 'POST':
        form = PredictPostForm(request.POST,
                               initial={'search': 'Классный у вас банк!',
                                        'ml_algorithm': 'model1'}
                               )
        context = {'form': form,
                   'is_predicted': True}
        if form.is_valid():
            sentence = form.cleaned_data['sentence']
            ml_algorithm = form.cleaned_data['ml_algorithm']
            predictions = predict(sentence, ml_algorithm)
            context.update(predictions)
    else:
        form = PredictPostForm()
        context = {'form': form,
                   'is_predicted': False}
    return render(request, 'main/home.html', context)

