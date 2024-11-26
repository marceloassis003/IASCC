""" 
########################################  APRENDIZAGEM POR REFORÇO - IMPLEMENTAÇÃO - PART 1 #####################################################################
# exemplo declarações variaveis: 
        self._numero_conta = numero_conta  # Atributo protegido: uso indicado dentro da classe e subclasses
        self.__saldo = 0  # Atributo privado: modificação restrita dentro da classe
#
# 
#
#___________________________________________________________________________________________________________________________________________________________________________#
"""

from abc import ABC, abstractmethod
import random
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import simpy



# classes de objetos 
class Accao(ABC):
    def __init__(self, accoes: list):
        self.accoes = accoes
        

class Estado(ABC):
    def __init__(self):
        self.Estado = Estado 

class MemoriaAssociativa(ABC):
    def __init__(self):
        super().__init__()

# interfaces 

class MemoriaAprend(ABC):
    def __init__(self, s: Estado, a: Accao, q:float):
        self.s = s
        self.a = a
        self.q = q

    @abstractmethod
    def actualizar(self, s, a, q):
        pass

    @abstractmethod
    def Q(self, s, a):
        pass 

# classes desenvolvimento 


class aprendizagem(ABC):
    def __init__(self, s):
        pass 

    
    @abstractmethod
    def seleccionar_accao(self):
        s = Estado
        s = Accao

    @abstractmethod
    def aprender(self): 
        s = Estado
        a = Accao
        r = float
        sn = Estado
        an = Accao

        
class SelAccao(ABC):
    def __init__(self, mem_aprend: MemoriaAprend ):
        self.mem_aprend = mem_aprend

    @abstractmethod
    def seleccionar_accao(self, s: Estado) -> Accao:
        pass

    @abstractmethod
    def max_accao(self, s: Estado) -> Accao:
        pass 


class AprendRef(ABC):
    def __init__(self, mem_aprend: MemoriaAprend, sel_accao: SelAccao, alfa:float, gama:float):
        self.mem_aprend = mem_aprend
        self.sel_accao = sel_accao
        self.alfa = alfa
        self.gama = gama

    @abstractmethod
    def aprender(self, s: Estado, a: Accao, r: float, sn: Estado, an: Accao=None):
        self.s = s
        self.a = a
        self.r = r 
        self.sn = sn
        self.an = an
    
     
class MecAprendRef(aprendizagem, AprendRef, SelAccao):
    def __init__(self, accoes: Accao):
        self.accoes = accoes
        

    def aprender(self, s: Estado, a: Accao, r: float, sn: Estado, an: Accao=None):
        self.s = s
        self.a = a
        self.r = r
        self.sn = sn
        self.an = an
        return self.s, self.a, self.r, self.sn, self.an


    @classmethod
    def selecionar_accao(self, s: Estado):
       self.s = Accao
       return self.s
        

# classes principais 


class EGreedy(SelAccao, Accao):
    def __init__(self, mem_aprend: MemoriaAprend, accoes: list[Accao], episolon: float):
        self.mem_aprend = mem_aprend
        self.accoes = accoes
        self.episolon = episolon

    def max_accao(self, s: Estado) -> Accao:
        random.shuffle(self.accoes)
        return max(self.accoes, key=lambda a: self.mem_aprend.Q(s, a))
    
    def aproveitar(self, s):
        return self.max_accao(s)
    
    def explorar(self):
        return random.choice(self.accoes)
    
    def seleccionar_accao(self, s):
        if random.random() > self.episolon:
            accao = self.aproveitar(s)
        else:
            accao = self.explorar()
        return accao
     

class MemoriaEsparsa(MemoriaAprend):
    def __init__(self, valor_omissao = 0.0):
        self.valor_omissao = valor_omissao
        self.memoria = {}

    def Q(self, s, a):
        return self.memoria.get((s, a), self.valor_omissao)
        
    
    def actualizar(self, s, a, q):
        self.memoria[(s, a)] = q

    def obter_memoria(self):
        return self.memoria


class SARSA(AprendRef):
    def __init__(self, mem_aprend, sel_accao, alfa, gama):
        super().__init__(mem_aprend, sel_accao, alfa, gama)


    def aprender(self, s, a, r, sn, an):
        qsa = self.mem_aprend.Q(s, a)
        qsnan = self.mem_aprend.Q(sn, an)
        q = qsa + self.alfa * (r + self.gama * qsnan - qsa)
        self.mem_aprend.actualizar(s, a, q)


class QLearning(AprendRef):
    def __init__(self, mem_aprend, sel_accao, alfa, gama):
        super().__init__(mem_aprend, sel_accao, alfa, gama)   

    def aprender(self, s, a, r, sn):
        an = self.sel_accao.max_accao(sn)
        qsa = self.mem_aprend.Q(s, a)
        qsnan = self.mem_aprend.Q(sn, an)
        q = qsa + self.alfa * (r + self.gama * qsnan - qsa)
        self.mem_aprend.actualizar(s, a, q)


class ModeloTR():
    def __init__(self):
        self.T = {}
        self.R = {}

    def actualizar(self, s, a, r, sn):
        self.T[(s, a)] = sn
        self.R[(s, a)] = r

    def amostrar(self):
        s, a = random.choice(list(self.T.keys()))
        sn = self.T[(s, a)]
        r = self.R[(s, a)]
        return s, a, r, sn 


class DynaQ(QLearning):
    def __init__(self, mem_aprend, sel_accao, alfa, gama, num_sim):
        super().__init__(mem_aprend, sel_accao, alfa, gama)
        self.num_sim = num_sim
        self.modelo = ModeloTR()     

    def aprender(self, s, a, r, sn):
        super().aprender(s, a, r, sn)
        self.modelo.actualizar(s, a, r, sn)
        self.Simular()

    def Simular(self):
        for _ in range(self.num_sim):
            s, a, r, sn = self.modelo.amostrar()
            super().aprender(s, a, r, sn)


class QME(QLearning):  #
    def __init__(self, mem_aprend, sel_accao, alfa, gama, num_sim, dim_max):
        super().__init__(mem_aprend, sel_accao, alfa, gama)
        self.num_sim = num_sim
        self.memoria_experiencia = MemoriaExperiencia(dim_max)

    def aprender(self, s, a, r, sn):
        super().aprender(s, a, r, sn)
        e = (s, a, r, sn)
        self.memoria_experiencia.actualizar(e)
        #simular()

    def simular(self):
        amostras = self.memoria_experiencia.amostrar(self.num_sim)
        for (s, a, r, sn) in amostras:
            super().aprender(s, a, r, sn)

    
class MemoriaExperiencia():
    def __init__(self, dim_max):
        self.dim_max = dim_max
        self.memoria = []

    def actualizar(self, e):
        if len(self.memoria) == self.dim_max:
            self.memoria.pop(0)
        self.memoria.append(e)

    def amostrar(self, n):
        n_amostras = min(n, len(self.memoria))
        return random.sample(self.memoria, n_amostras)




# criando ambiente de testes e testando 
""" 
#--- ambiente de testes da lib

class Ambiente:
    def __init__(self,env, tamanho=7):
        self.env = env
        self.tamanho = tamanho
        self.estado = self.tamanho // 2

    def reset(self):
        self.estado = self.tamanho // 2
        return self.estado
    
    def passo(self, acao):
        recompensa = 0 
        if acao == 'direita':
            if self.estado < self.tamanho - 1:
                self.estado += 1
            #else:
                #recompensa = 10
        elif acao == 'esquerda':
            if self.estado > 0: 
                self.estado -= 1
            #else:
                #recompensa = -10 
        if  self.estado == self.tamanho -1:
            recompensa = 10 
        
        elif self.estado == 0: 
            recompensa = -10 
        else: 
            recompensa = -1


        print(self.estado)
        print(recompensa)
        return self.estado, recompensa
    
    
def simular_agente(env, agente, amb, sel_accao, num_episodios, ax, patch):
    for episodio in range(num_episodios):
        estado = amb.reset()
        accao = sel_accao.seleccionar_accao(estado)
        recompensas = []
        trajetoria = []

        for _ in range(20):
            yield env.timeout(1) 
            novo_estado, recompensa = amb.passo(accao.accoes[0])
            nova_accao = sel_accao.seleccionar_accao(novo_estado)

            agente.aprender(estado, accao, recompensa, novo_estado)
            estado = novo_estado
            accao = nova_accao 

            recompensas.append(recompensa) 
            #trajetoria.append(estado) 

            patch.set_x(estado)
            plt.pause(0.1)
                
            if estado == amb.tamanho - 1:
                break

        todas_recompensas.append(recompensas) 
        todas_trajetorias.append(trajetoria)


env = simpy.Environment()
amb = Ambiente(env)

dim_max = 1000
num_sim = 10
alfa = 0.1
gama = 0.9
epsilon = 0.1

# Criando a memória de aprendizagem e seleção de ação (e-greedy)
mem_aprend = MemoriaEsparsa() 
accoes = [Accao(['esquerda']), Accao(['direita'])]
sel_accao = EGreedy(mem_aprend, accoes, epsilon)
#[Accao(['esquerda', 'direita'])], epsilon)

agente = QME(mem_aprend, sel_accao, alfa, gama, num_sim, dim_max) #DynaQ(mem_aprend, sel_accao, alfa, gama, num_sim)    

num_episodios = 10000
sucess = 0
#recompensas_acumuladas = []
todas_recompensas = []
todas_trajetorias = []

# iniciando o grafico 
fig, ax  = plt.subplots()
ax.set_xlim(0, amb.tamanho) 
ax.set_ylim(-1, 1) 
patch = patches.Rectangle((amb.estado, 0), 1, 0.5, fc='black')
ax.add_patch(patch)


# iniciando ambiente 
env.process(simular_agente(env, agente, amb, sel_accao, num_episodios, ax, patch))
env.run()

plt.show()
"""
#____________________________________________versão 1 graficos - discard_____________________________________________________________________________________________________________________________________________#
""" 
for episodio in range(num_episodios):
    estado = amb.reset()
    accao = sel_accao.seleccionar_accao(estado)
    #recompensa_acumulada = 0
    recompensas = []
    trajetoria = []
    trajeto_sucess = False


    for _ in range(20):  # Limite de passos por episódio
        novo_estado, recompensa = amb.passo(accao.accoes[0])
        nova_accao = sel_accao.seleccionar_accao(novo_estado)

        # Aprendizado pelo método Q-Learning
        agente.aprender(estado, accao, recompensa, novo_estado)
        
        # Atualiza o estado e a ação atuais
        estado = novo_estado
        accao = nova_accao

        recompensas.append(recompensa)
        trajetoria.append(estado)

        #recompensa_acumulada += recompensa

        # Verifica se o objetivo foi alcançado
        if estado == amb.tamanho - 1:
            trajeto_sucess = True
            break
    if trajeto_sucess:
        sucess += 1

    #recompensas_acumuladas.append(recompensa_acumulada)
    todas_recompensas.append(recompensas)
    todas_trajetorias.append(trajetoria)


accuracy = sucess / num_episodios

print("Treinamento concluído!")
print(f"Accuracy do agente: {accuracy * 100: .2f}%")
print("Tabela Q-Learning:")
for k, v in mem_aprend.obter_memoria().items():
    print(f"Estado {k[0]}, Ação {k[1]}: {v}")
#print(mem_aprend.obter_memoria())


fig, axs = plt.subplots(2, figsize=(10,10))


for recompensas, trajetoria in zip(todas_recompensas, todas_trajetorias):
    axs[0].plot(recompensas, alpha=0.3)
    axs[1].plot(trajetoria, alpha=0.3)

axs[0].set_title('Recompensas por Passo') 
axs[0].set_xlabel('Passos') 
axs[0].set_ylabel('Recompensa') 

axs[1].set_title('Trajetórias por Passo') 
axs[1].set_xlabel('Passos') 
axs[1].set_ylabel('Estado') 

plt.tight_layout()
plt.show()


"""
































