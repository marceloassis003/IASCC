import random
import numpy as np
import matplotlib.pyplot as plt
from biblioteca_aula  import Neuronio, CamadaDensa, RedeNeuronal
import pandas as pd 



def tah(x):
    return np.tanh(x)


# implementação do problema usando tangente um dataset de seguros de acordo com idade
#dataset simples para o problema 
np.random.seed(42)
idades = np.linspace(17, 60, 15)  
valores_seguro = 0.01 * (idades - 17)**2 + 0.2 + np.random.normal(0, 0.02, len(idades))  



# Normalização de dados 
idades_norm = (idades - min(idades)) / (max(idades) - min(idades))  # Normaliza para [0, 1]
valores_seguro_norm = (valores_seguro - min(valores_seguro)) / (max(valores_seguro) - min(valores_seguro))

dataset = {"idades": idades, "valores_seguro": valores_seguro}
df = pd.DataFrame(dataset)

print(df)


# Separar dados em entrada (X) e saída (Y)
X = idades_norm.reshape(-1, 1)  # Idades (entrada)
Y = valores_seguro_norm.reshape(-1, 1)  # Valores do seguro (saída)

# aplicando a biblioteca para treinamento da rede 
forma = [1, 15, 1]   # 17    = 90 % 
rede = RedeNeuronal(forma, tah)  

# parametros 
n_epocas = 10000#5000        # Número de épocas
epsilon_max = 0.01     # Erro máximo permitido
alpha = 0.03            # Taxa de aprendizado # 0.5 ideal até agora 
beta = 0.9              # Factor de momento        #0.8     

# Treinar a rede                                        # maximo acerto 86%
historico_erro = rede.treinar(X, Y, n_epocas, epsilon_max, alpha, beta)


predicoes = rede.prever(X)

#porcentagem de acerto
# calculo erro medio absoluto
accuracy = np.mean(np.abs(Y - predicoes))
percent_acerto = 100 * (1 - accuracy)

# Exibir a porcentagem de acerto
print(f"Acerto: {percent_acerto:.2f}%")

# configurar previsões 
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, color='magenta', label='Observado') 
plt.plot(X, predicoes, color='blue', label='Previsões')  
plt.title('Previsão de Seguro')
plt.xlabel('Idade')
plt.ylabel('Valor do Seguro')
plt.legend()
plt.grid(True)
plt.show()
