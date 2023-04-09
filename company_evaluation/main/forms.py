from django import forms


class PredictPostForm(forms.Form):
    MODEL_CHOICES = (
        ('m1', 'model1'),
        ('m2', 'model2'),
    )
    sentence = forms.CharField(label='Введите ваш отзыв',
                               required=False)
    ml_algorithm = forms.ChoiceField(choices=MODEL_CHOICES,
                                     label='Выберите модель',
                                     required=False)
