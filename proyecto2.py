import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Cargar los datos del CSV
csv_file_path = "inteligente.csv"
df = pd.read_csv(csv_file_path)

# Función para mostrar gráficos en Streamlit
def plot_histogram(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['IQ'], bins=10, kde=True)
    plt.title('Distribución de IQ')
    plt.xlabel('IQ')
    plt.ylabel('Frecuencia')
    plt.grid(True)
    st.pyplot(plt.gcf())
    plt.close()

def plot_gender_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Gender', data=df)
    plt.title('Distribución de Género')
    plt.xlabel('Género')
    plt.ylabel('Frecuencia')
    plt.grid(True)
    st.pyplot(plt.gcf())
    plt.close()

def plot_iq_by_field(df):
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Field of Expertise', y='IQ', data=df)
    plt.title('Distribución de IQ por Campo de Especialización')
    plt.xlabel('Campo de Especialización')
    plt.ylabel('IQ')
    plt.xticks(rotation=90)
    plt.grid(True)
    st.pyplot(plt.gcf())
    plt.close()

def plot_achievements_count(df):
    plt.figure(figsize=(14, 8))
    sns.countplot(y='Achievements', data=df, order=df['Achievements'].value_counts().index)
    plt.title('Conteo de Logros')
    plt.xlabel('Frecuencia')
    plt.ylabel('Logros')
    plt.grid(True)
    st.pyplot(plt.gcf())
    plt.close()

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
        st.pyplot(plt.gcf())
        plt.close()
    else:
        st.write("Las columnas 'Years of Experience' y 'IQ' no están presentes en los datos.")

# Crear la interfaz de Streamlit
st.title("Análisis de Datos")

st.sidebar.header("Selecciona el gráfico para mostrar")
option = st.sidebar.selectbox(
    "Selecciona una opción:",
    ["Distribución de IQ", "Distribución de Género", "IQ por Campo de Especialización", "Conteo de Logros", "Regresión Lineal"]
)

if option == "Distribución de IQ":
    plot_histogram(df)
elif option == "Distribución de Género":
    plot_gender_distribution(df)
elif option == "IQ por Campo de Especialización":
    plot_iq_by_field(df)
elif option == "Conteo de Logros":
    plot_achievements_count(df)
elif option == "Regresión Lineal":
    plot_linear_regression(df)
