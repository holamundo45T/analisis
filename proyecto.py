import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Cargar los datos del CSV
csv_file_path = "inteligente.csv"
df = pd.read_csv(csv_file_path)

# Obtener el directorio del archivo CSV
output_dir = os.path.dirname(csv_file_path)

# Estadísticas descriptivas
print(df.describe())

# Gráfico de distribución de IQ
plt.figure(figsize=(10, 6))
sns.histplot(df['IQ'], bins=10, kde=True)
plt.title('Distribución de IQ')
plt.xlabel('IQ')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'distribucion_iq.png'))
plt.close()

# Gráfico de barras de género
plt.figure(figsize=(10, 6))
sns.countplot(x='Gender', data=df)
plt.title('Distribución de Género')
plt.xlabel('Género')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'distribucion_genero.png'))
plt.close()

# Gráfico de caja de IQ por campo de especialización
plt.figure(figsize=(14, 8))
sns.boxplot(x='Field of Expertise', y='IQ', data=df)
plt.title('Distribución de IQ por Campo de Especialización')
plt.xlabel('Campo de Especialización')
plt.ylabel('IQ')
plt.xticks(rotation=90)
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'iq_por_campo.png'))
plt.close()

# Gráfico de conteo de logros
plt.figure(figsize=(14, 8))
sns.countplot(y='Achievements', data=df, order=df['Achievements'].value_counts().index)
plt.title('Conteo de Logros')
plt.xlabel('Frecuencia')
plt.ylabel('Logros')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'conteo_logros.png'))
plt.close()
