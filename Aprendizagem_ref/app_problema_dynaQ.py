"""------------------------------------------------------------------------------------------------------------------------------------------------------------------ 
# IMPLENTAÇÃO DA AI - ALGORITMO POR REFORÇO  . DynaQ
#                                                                     GAME - CHEGAR NO CIRCULO
# Objectivo: Desviar dos obtaculos e chegar no circulo 
# 
# 
#_____________________________________________________________________________________________________________________________________________________________________________"""



import pygame
from pygame.locals import *
from sys import exit
import numpy as np
from pratica_pt2 import Accao, MemoriaEsparsa, EGreedy, QLearning, DynaQ

# Configurações do jogo
ALTURA = 640
LARGURA = 550
VEL = 10
RAIO_AGENTE = 20
RAIO_OBJECTIVO = 30
NUM_EPISODIOS = 1000
NUM_SIM = 10
ALFA = 0.1
GAMA = 0.9
EPSILON = 0.1

# Cores
PRETO = (0, 0, 0)
AMARELO = (255, 255, 0)
VERDE = (0, 255, 0)
AZUL = (0, 255, 255)


class Ambiente:
    def __init__(self, altura, largura, vel):
        self.altura = altura
        self.largura = largura
        self.vel = vel
        self.recompensas = np.zeros((altura // vel, largura // vel))  # Matriz de recompensas
        self.obstaculos = []  
        

        # Definir o objetivo antes de criar o Rect
        self.objetivo = (27, 54)  # Posição do objetivo (linha, coluna)
        self.recompensas[self.objetivo[0]][self.objetivo[1]] = 10

        # Criar um Rect para o objetivo
        self.objectivo = pygame.Rect(
            self.objetivo[1] * vel,  # Coordenada x
            self.objetivo[0] * vel,  # Coordenada y
            vel,  # Largura
            vel   # Altura
        )

        # Inicializar o ambiente
        self.configurar_ambiente()

    def configurar_ambiente(self):
        # Definir o objetivo
        self.objetivo = (27, 54)  # Posição do objetivo (matriz)
        self.recompensas[self.objetivo[0]][self.objetivo[1]] = 10

        # Criar os obstáculos (C invertido)
        for i in range(21, 45):
            self.recompensas[i][32] = -1  # Penalização para bater na parede
            self.obstaculos.append(pygame.Rect(32 * VEL, i * VEL, VEL, VEL))

        self.obstaculos.append(pygame.Rect(30 * VEL , 21 * VEL, 3 * VEL, VEL))  # Parte superior
        self.obstaculos.append(pygame.Rect(30 * VEL , 44 * VEL, 3 * VEL, VEL))  # Parte inferior

        # Adicionar margens como obstáculos
        margem_espessura = 5  # Ajuste conforme necessário

        # Margem superior
        self.obstaculos.append(pygame.Rect(0, 0, self.largura + 100, margem_espessura))
        # Margem inferior
        self.obstaculos.append(pygame.Rect(0, self.altura - 95, self.largura + 100, margem_espessura))
        # Margem esquerda
        self.obstaculos.append(pygame.Rect(0, 0, margem_espessura, self.altura))
        # Margem direita
        self.obstaculos.append(pygame.Rect(self.largura + 85, 0, margem_espessura, self.altura))

        # Atualizar penalizações para as bordas na matriz de recompensas
        for i in range(self.altura // self.vel):  # Altura em passos
            for j in range(self.largura // self.vel):  # Largura em passos
                # Penalizar células na margem superior ou inferior
                if i < margem_espessura // self.vel or i >= (self.altura - margem_espessura) // self.vel:
                    self.recompensas[i][j] = -10
                # Penalizar células na margem esquerda ou direita
                if j < margem_espessura // self.vel or j >= (self.largura - margem_espessura) // self.vel:
                    self.recompensas[i][j] = -10

    def desenhar(self, tela):
        
        # Desenhar obstáculos e margem
        for obstaculo in self.obstaculos:
            pygame.draw.rect(tela, AMARELO, obstaculo)


        # Desenhar o objetivo
        objetivo_centro = (self.objetivo[1] * VEL + VEL // 2, self.objetivo[0] * VEL + VEL // 2)
        pygame.draw.circle(tela, VERDE, objetivo_centro, RAIO_OBJECTIVO)

    def verificar_colisao(self, x, y):
        # Verificar colisão com os obstáculos
        for obstaculo in self.obstaculos:
            if obstaculo.collidepoint(x, y):
                x = x
                y = y 
                return x,y
        return False
    
    def obter_estado(self, pos):
        return (pos[1] // self.vel, pos[0] // self.vel)

    def obter_recompensa(self, estado):
        if estado[0] < 0 or estado[0] >= self.altura //self.vel or estado[1] < 0 or estado[1] >= self.largura // self.vel:
            return - 10
        return self.recompensas[estado[0]][estado[1]]
    


class Agente:
    def __init__(self, ambiente):
        self.ambiente = ambiente
        self.x = 100
        self.y = 100
        self.mem_aprend = MemoriaEsparsa()
        self.accoes = [Accao(['esquerda']),Accao(['direita']),Accao(['cima']),Accao(['baixo'])]

        self.sel_accao = EGreedy(self.mem_aprend, self.accoes, EPSILON)
        self.dynaq = DynaQ(self.mem_aprend, self.sel_accao, ALFA, GAMA, NUM_SIM)
        

    def mover(self, acao):
        move_x, move_y = 0, 0
        if acao == 'esquerda':
            move_x = -VEL
        elif acao == 'direita':
            move_x = VEL
        elif acao == 'cima':
            move_y = -VEL
        elif acao == 'baixo':
            move_y = VEL

        novo_x = self.x + move_x
        novo_y = self.y + move_y

        margem_espessura = 5
        novo_x = max(margem_espessura + RAIO_AGENTE, min(novo_x, self.ambiente.largura - margem_espessura - RAIO_AGENTE))
        novo_y = max(margem_espessura + RAIO_AGENTE, min(novo_y, self.ambiente.altura - margem_espessura - RAIO_AGENTE))

        
        # Verificar colisão com obstáculos
        if self.ambiente.verificar_colisao(novo_x, novo_y):
            return self.x, self.y  # Não se move ao colidir

        return novo_x, novo_y

    def treinar(self, num_episodios):
        for episodio in range(num_episodios):
            self.x, self.y = 100, 100  # Reiniciar posição inicial
            estado = self.ambiente.obter_estado((self.x, self.y))
            accao = self.sel_accao.seleccionar_accao(estado)
            total_recompensa = 0
            passos = 0 
            for passo in range(500):
                novo_x, novo_y = self.mover(accao.accoes[0])
                novo_estado = self.ambiente.obter_estado((novo_x, novo_y))
                recompensa = self.ambiente.obter_recompensa(novo_estado)
                total_recompensa += recompensa # Atualizar recompensa total e passos , para testes no print 
                
                
                if self.ambiente.objectivo.colliderect(pygame.Rect(novo_x, novo_y, RAIO_AGENTE, RAIO_AGENTE)):
                    recompensa = 10 # Recompensa máxima ao atingir o objetivo
                    self.dynaq.aprender(estado,accao, recompensa, novo_estado)
                    break  # Encerrar o episódio, mas continuar o treinamento

                self.x, self.y = novo_x, novo_y
                estado = novo_estado
                accao = self.sel_accao.seleccionar_accao(estado)
                #print(self.mem_aprend.memoria)
                #print(estado)
                #print(recompensa)
 

    def executar_politica(self):
        estado = self.ambiente.obter_estado((self.x, self.y))
        accao = self.sel_accao.seleccionar_accao(estado)  # Política aprendida
        self.x, self.y = self.mover(accao.accoes[0])

        # Verificar se atingiu o objetivo
        if self.ambiente.objectivo.colliderect(pygame.Rect(self.x, self.y, RAIO_AGENTE, RAIO_AGENTE)):
            self.x, self.y = 100, 100  # Reiniciar para o início



# Configuração do jogo
pygame.init()
screen = pygame.display.set_mode((ALTURA, LARGURA))
pygame.display.set_caption("Aprendizagem por Reforço")
relogio = pygame.time.Clock()

# Inicializar ambiente e agente
ambiente = Ambiente(ALTURA, LARGURA, VEL)
agente = Agente(ambiente)

# Treinamento do agente
agente.treinar(NUM_EPISODIOS)

# Loop principal
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            exit()

    # Agente segue a política aprendida
    agente.executar_politica()

    # Desenhar na tela
    screen.fill(PRETO)
    ambiente.desenhar(screen)
    pygame.draw.circle(screen, AZUL, (agente.x, agente.y), RAIO_AGENTE)
    pygame.display.update()
    relogio.tick(30) 