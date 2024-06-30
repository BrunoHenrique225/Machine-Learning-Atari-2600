import gymnasium
import random as rd

env = gymnasium.make("ALE/Freeway-ram-v5", render_mode="human", mode=0)
percorrido = 0
total = []
for episodio in range(5): # 10
    recompensa_episodio = 0
    percorrido = 0
    obs, info = env.reset()    
    for passo in range(1000):
        acao = 1    
        obs, recompensa, fim, truncado, info = env.step(acao)
        if acao == 1: percorrido += 1
        if acao == 2: percorrido -= 1

        if recompensa == 1: percorrido = 0
        #recompensa_episodio += recompensa
        if (percorrido > 43 or percorrido < 0) and recompensa != 1: recompensa_episodio += -1
        else: recompensa_episodio += recompensa
        if fim or truncado:
            #print("=====================")
            break
    total.append(recompensa_episodio)

print(total)
print(sum(total))