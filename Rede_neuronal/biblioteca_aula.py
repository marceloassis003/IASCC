import random
import numpy as np
import matplotlib.pyplot as plt


# calculo diferencial entre dois pontos 
    

def diferencial(f,x, h=1e-5):
        return (f(x + h) - f(x)) / h


class Neuronio:

    def __init__(self,d,phi):       
        #super.__init__()
        self.d = d
        self.phi = phi 

        self.w = np.random.uniform(-1, 1, d)

        self.b = np.random.uniform(-1, 1)
        
        self.delta_w = np.zeros(d)

        self.delta_b = 0

        self.h = 0 
        self.y = 0
        self.y1 = 0
    
    
    def propagar(self,x):
        self.h = np.dot(self.w, x) + self.b # np.dot representa o escalar 
        self.y = self.phi(self.h)
        self.y1 = diferencial(self.phi,self.h)
        return self.y 
       
    

    def adaptar(self, delta, y_anterior, alpha, beta): 
        y_anterior = np.array(y_anterior)
        Mw = beta * self.delta_w
        self.deltaw = -alpha * self.y1 * delta * y_anterior + beta + Mw
        self.w += self.delta_w

        Mb = beta * self.delta_b
        self.delta_b = -alpha * self.y1 * delta + Mb
        self.b += self.delta_b

        


class CamadaDensa:
    def __init__(self, de, ds, phi):
        self.de = de 
        self.ds = ds
        self.phi = phi 

        self.neuronios = [Neuronio(de, phi) for _ in range(ds)]

    def propagar(self, x):
        self.y = [neuronio.propagar(x) for neuronio in self.neuronios]
        return self.y 
    
    def saida(self):
        return [neuronio.y for neuronio in self.neuronios]
     
    def adaptar(self, delta_n, y_anterior, alpha, beta):
        for j in range(self.ds):
            self.neuronios[j].adaptar(delta_n[j], y_anterior, alpha, beta)
    
class CamadaEntrada:
    def __init__(self, ds):
        self.ds = ds  
        self.y = [0 for _ in range(ds)]

    def propagar(self, x):
        self.y = x  
        return self.y

class RedeNeuronal:
    def __init__(self, forma, phi):
        self.camadas = []   
        self.N = len(forma)  
        ds1 = forma[0]
        camada_entrada = CamadaEntrada(ds1)
        self.camadas.append(camada_entrada)

        for n in range(1, self.N):
            de_n = forma[n - 1]  
            ds_n = forma[n]  
            camada_densa = CamadaDensa(de_n, ds_n, phi)
            self.camadas.append(camada_densa)

    def delta_saida(self, yN, y):
        return [yN[k] - y[k] for k in range(len(y))]
    
    def retropropagar(self, deltaN, alpha, beta):
        delta_atual = deltaN
        
        for n in range(len(self.camadas) -1, 0, -1): 
            camada_atual = self.camadas[n]
            camada_anterior = self.camadas[n - 1]


            y_anterior = camada_anterior.y
            #derivada_anterior = [neuronio.y1 for neuronio in camada_anterior.neuronios]
            if isinstance(camada_anterior, CamadaDensa):
                derivada_anterior = [neuronio.y1 for neuronio in camada_anterior.neuronios]
            else:
                derivada_anterior = [1] * len(camada_anterior.y)  # Derivada trivial para a entrada

            delta_anterior = []
            for i in range(len(camada_anterior.y)):
                soma = sum(
                    camada_atual.neuronios[J].w[i] * delta_atual[J]
                    for J in range(len(camada_atual.neuronios))
                )
                delta_anterior.append(soma * derivada_anterior[i])

            camada_atual.adaptar(delta_atual, y_anterior, alpha, beta)
            delta_atual = delta_anterior
    
    def adaptar(self, x, y, alpha, beta):
        yN = self.propagar(x)
        delta_n = self.delta_saida(yN, y)

        self.retropropagar(delta_n, alpha, beta)

        k = len(delta_n)
        erro_medio = (1/k) * sum(delta ** 2 for delta in delta_n)

        return erro_medio

    def treinar(self, x, y, n_epocas, epsilon_max, alpha, beta):
        historico_erro = []

        for epoca in range(n_epocas):
            erro_epoca = 0

            for entrada,saida in zip(x,y):
                erro_x = self.adaptar(entrada,saida, alpha, beta)
                erro_epoca = max(erro_epoca, erro_x)

            historico_erro.append(erro_epoca)
            print(f"Época {epoca + 1}, Erro: {erro_epoca}")

            if erro_epoca <= epsilon_max:
                print("Treinamento concluído. Erro aceitável atingido.")
            #break

        return historico_erro
    
    def prever(self,X):
        Y = [self.propagar(x) for x in X]
        return Y

    def propagar(self, x):
        y = x
        for camada in self.camadas:
            y = camada.propagar(y)
        return y
    
    
    
""" 
def sigmoid(x):
    return (1 / (1+np.exp(-x))) # np.exp significa o exponencial do valor

def sigmoid_derivada(x):
    s = sigmoid(x)
    return s * (1-s)

def tah(x):
    return np.tanh(x)

def tah_derivada(x):
    return 1 - np.tanh(x) **2


# ambiente testes para o problema XOR

#forma = [2, 3, 1]
forma = [2, 3, 1]

rede = RedeNeuronal(forma, sigmoid)
#rede.camadas[-1].neuronios = [Neuronio(3, sigmoid)]

#X = np.array([
#    [-1, -1],
#    [-1, 1],
#    [1, -1],
#    [1, 1]
#]) 

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

Y = np.array([
    [0],
    [1],
    [1],
    [0]
])

#n_epocas = 5000        # Número de épocas
#n_epocas = 700
#epsilon_max = 0.03      # Erro máximo permitido
#alpha = 0.1               # Taxa de aprendizado
#beta = 0.5         # Fator de momento 

#n_epocas = 5000        # Número de épocas
n_epocas = 10000
epsilon_max = 0.01     # Erro máximo permitido  
alpha = 0.5            # Taxa de aprendizado # rate learning 
beta = 0.7         # Factor de momento

historico_erro = rede.treinar(X, Y, n_epocas, epsilon_max, alpha, beta)

# Exibir o histórico do erro
plt.plot(historico_erro)
plt.title('Evolução do Erro')
plt.xlabel('Épocas')
plt.ylabel('Erro')
plt.show()


predicoes = rede.prever(X)

for i, (entrada, saida) in enumerate(zip(X, predicoes)):
    print(f"Entrada: {entrada}, Saída esperada: {Y[i]}, Previsão: {saida}")

"""   