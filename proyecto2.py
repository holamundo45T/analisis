import pandas as pd
import seaborn as sns
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Cargar los datos del CSV
csv_file_path = "inteligente.csv"
df = pd.read_csv(csv_file_path)

# Estadísticas descriptivas
print(df.describe())

# Función para mostrar gráficos
def plot_histogram(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['IQ'], bins=10, kde=True)
    plt.title('Distribución de IQ')
    plt.xlabel('IQ')
    plt.ylabel('Frecuencia')
    plt.grid(True)
    plt.show()

def plot_gender_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Gender', data=df)
    plt.title('Distribución de Género')
    plt.xlabel('Género')
    plt.ylabel('Frecuencia')
    plt.grid(True)
    plt.show()

def plot_iq_by_field(df):
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Field of Expertise', y='IQ', data=df)
    plt.title('Distribución de IQ por Campo de Especialización')
    plt.xlabel('Campo de Especialización')
    plt.ylabel('IQ')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.show()

def plot_achievements_count(df):
    plt.figure(figsize=(14, 8))
    sns.countplot(y='Achievements', data=df, order=df['Achievements'].value_counts().index)
    plt.title('Conteo de Logros')
    plt.xlabel('Frecuencia')
    plt.ylabel('Logros')
    plt.grid(True)
    plt.show()

def plot_linear_regression(df):
    if 'Years of Experience' in df.columns and 'IQ' in df.columns:
        X = df[['Years of Experience']]
        y = df['IQ']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(X_test, y_test, color='blue', label='Datos reales')
        plt.plot(X_test, y_pred, color='red', linewidth=2, label='Línea de regresión')
        plt.title('Regresión Lineal de IQ vs. Años de Experiencia')
        plt.xlabel('Años de Experiencia')
        plt.ylabel('IQ')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("Las columnas 'Years of Experience' y 'IQ' no están presentes en los datos.")

# Mostrar gráficos
plot_histogram(df)
plot_gender_distribution(df)
plot_iq_by_field(df)
plot_achievements_count(df)
plot_linear_regression(df)
