from django import forms


class PredictPostForm(forms.Form):
    MODEL_CHOICES = (
        ('bert', 'bert'),
        ('cat_boost', 'cat_boost'),
    )
    sentence = forms.CharField(label='Введите ваш отзыв',
                               required=False)
    ml_algorithm = forms.ChoiceField(choices=MODEL_CHOICES,
                                     label='Выберите модель',
                                     required=False)
