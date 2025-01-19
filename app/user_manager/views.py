from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm

from .forms import RegistrationForm


# Представление для регистрации
def register_view(request):
    if request.method == "POST":
        form = RegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)  # Автоматический вход
            return redirect("home")  # Перенаправление на главную страницу
    else:
        form = RegistrationForm()

    return render(request, "user_manager/register.html", {"form": form})


# Представление для авторизации
def login_view(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect("home")
    else:
        form = AuthenticationForm()
    return render(request, "user_manager/login.html", {"form": form})


# Представление для выхода
def logout_view(request):
    logout(request)
    return redirect("login")
