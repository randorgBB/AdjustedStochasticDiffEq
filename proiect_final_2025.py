import numpy as np
import matplotlib.pyplot as plt

def Monte_Carlo_Simulation(S_0, T , h , r, sigma, N, plot = False):
    
    # S_0        - pretul initial al activului cand t = 0
    # r          - rata de crestere/drift a procesului stocastic(in cazul activelor financiare r este rata dobanzii de referinta)
    # sigma      - volatilitatea procesului stocastic(in cazul activelor financiare 
    #             putem alege sigma drept abaterea standard a randamentelor activlui calculate la fiecare moment de timp)
    # N          - numar de simulari
    # h          - granularitatea schemei Euler
    # T          - nr termenului final de timp T (T este exprimat in ani)
    # dt         - parametru de scalare a procesului la fiecare moment de timp
    # timeSteps  - numarul momentelor de timp h * T, 
    #              am declarat variabila in mod redundat, alegand "timeSteps" in loc de "h", avand ca obiectiv claritatea
    # plot       - parametru boolean: True - Functia genereaza ploturi ale traiectoriilor, ale distributiei la momentul T, ale randamentelor la fiecare moment dt, si a distributiei randamentelor

    timeSteps = int(h*T)
    dt = 1/timeSteps

    r_scalat = r * T
    # initializam pasul intiliat dSt cu 0
    dSt = np.zeros(N)

    # declaram si initializam matricea care stocheaza traiectorile la fiecare moment de timp
    St_vector = np.zeros((N, timeSteps+1))
    St_vector[ : , 0 ] = S_0
    dS_vector = np.zeros((N, timeSteps+1))

    for i in range(1,timeSteps+1):

        # ecuatia stocastica dSt = St*r*dt + St*sigma*N(0,1)*dt
        dSt = St_vector[: , i - 1] * r_scalat * dt + St_vector[: , i - 1] * np.random.normal(0,1,N) * sigma * np.sqrt(dt)
        
        # inserarea randamentelor intr-un vector, in vederea analizei acestora
        dS_vector[:, i] = dSt

        # actualizarea schemei Euler
        St_vector[:, i] = St_vector[:, i - 1] + dSt

    # plot
    # if(plot == False) => functia nu ploteaza traiectoriile
    # if(plot == True)  => functia ploteaza traiectoriile
    if(plot == True):

        # declaram dimenisunile graficului
        plt.figure(figsize=(10,6))
        for i in range(N):  # Ploteaza fiecare traiectorie a pretului
            plt.plot(St_vector[i, :], linewidth=1)

        # Setarile Graficului
        plt.title(f"Simulare Monte Carlo a Procesului Stocastic Simplu", fontsize=16)
        plt.xlabel("Time Steps", fontsize=12)
        plt.ylabel("Traiectoria procesului stocastic", fontsize=12)
        plt.grid(True)
        plt.show()

        plt.hist(St_vector[ : , -1], bins=50, color='skyblue', edgecolor='black')
        plt.xlabel('Distributia lui ST')
        plt.ylabel('Densitate')
        plt.title('Graficul Densitatii lui ST')
        plt.show()

        plt.hist(dS_vector[ : , -1], bins=50, color='green', edgecolor='black')
        plt.xlabel('Distributia randamentelor dS')
        plt.ylabel('Densitate')
        plt.title('Graficul Densitatii lui dS')
        plt.show()

    # Monte_Carlo_Simulation()[0] - vectorul celor N traiectori simulate la momentul final T
    # Monte_Carlo_Simulation()[1] - vectorul celor N randamente simulate la momentul final T
    return St_vector[:,-1], dS_vector[ : , -1]

def Euler_Ajustat_Simulation(S_0, T , h , r, sigma, N, alpha, j, plot = False):
    
    # S_0        - pretul initial al activului cand t = 0
    # r          - rata de crestere/drift a procesului stocastic(in cazul activelor financiare r este rata dobanzii de referinta)
    # sigma      - volatilitatea procesului stocastic(in cazul activelor financiare 
    #             putem alege sigma drept abaterea standard a randamentelor activlui calculate la fiecare moment de timp)
    # N          - numar de simulari
    # h          - granularitatea schemei Euler
    # T          - nr termenului final de timp T (T este exprimat in ani)
    # dt         - parametru de scalare a procesului la fiecare moment de timp
    # timeSteps  - numarul momentelor de timp h * T, 
    #              am declarat variabila in mod redundat, alegand "timeSteps" in loc de "h", avand ca obiectiv claritatea
    # alpha      - parametru real care scaleaza procesul de ajustare prin raportarea la minimul/maximul traiectoriei pana la momentul [i-j]
    # j          - parametru natural - lungimea ferestrei in care evaluam minimum/maximul traiectoriei
    # plot       - parametru boolean: True - Functia genereaza ploturi ale traiectoriilor, ale distributiei la momentul T, ale randamentelor la fiecare moment dt, si a distributiei randamentelor


    timeSteps = int(h*T)
    dt = 1/timeSteps

    r_scalat = r * T
    # initializam pasul intiliat dSt cu 0
    dSt = np.zeros(N)

    # declaram si initializam matricea care stocheaza traiectorile la fiecare moment de timp
    St_vector = np.zeros((N, timeSteps+1))
    St_vector[ : , 0 ] = S_0
    dS_vector = np.zeros((N, timeSteps+1))

    for i in range(1,timeSteps+1):

        # Minimul si Maximul calculate conform unei ferestre de timp j mobile
        max_mobil = np.max(St_vector[:, max(0, i-j):i], axis=1)
        min_mobila = np.min(St_vector[:, max(0, i-j):i], axis=1)

        # Compute correction term
        correction = alpha * ((max_mobil + min_mobila - 2 * St_vector[:, i - 1]) ** 2)
        correction = np.clip(correction, -1e6, 1e6) 

        # ecuatia stocastica dSt = St*r*correction*dt + St*sigma*N(0,1)*dt
        dSt = (St_vector[: , i - 1] + correction) * r_scalat * dt + St_vector[: , i - 1] * np.random.normal(0,1,N) * sigma * np.sqrt(dt)
        # inserarea randamentelor intr-un vector, in vederea analizei acestora
        dS_vector[:, i] = dSt

        # actualizarea schemei Euler
        St_vector[:, i] = St_vector[:, i-1] + dSt

    # plot
    # if(plot == False) => functia nu ploteaza traiectoriile
    # if(plot == True)  => functia ploteaza traiectoriile
    if(plot == True):

        # declaram dimenisunile graficului
        plt.figure(figsize=(10,6))
        for i in range(N):  # Ploteaza fiecare traiectorie a pretului
            plt.plot(St_vector[i, :], linewidth=1)

        # Setarile Graficului
        plt.title(f"Simulare Monte Carlo a Procesului Stocastic Ajustat min/max", fontsize=16)
        plt.xlabel("Time Steps", fontsize=12)
        plt.ylabel("Stock Price at t (St)", fontsize=12)
        plt.grid(True)
        plt.show()

        plt.hist(St_vector[ : , -1], bins=50, color='skyblue', edgecolor='black')
        plt.xlabel('Distributia lui ST')
        plt.ylabel('Densitate')
        plt.title('Graficul Densitatii lui ST')
        plt.show()

        plt.hist(dS_vector[ : , -1], bins=50, color='green', edgecolor='black')
        plt.xlabel('Distributia randamentelor dS')
        plt.ylabel('Densitate')
        plt.title('Graficul Densitatii lui dS')
        plt.show()

    # Euler_Ajustat_Simulation()[0] - vectorul celor N traiectori simulate la momentul final T
    # Euler_Ajustat_Simulation()[1] - vectorul celor N randamente simulate la momentul final T
    return St_vector[:,-1], dS_vector[:,-1]

def Euler_Ajustat_Patratic_Sim(S_0, T , h , r, sigma, N, alpha, j, plot = False):

    # S_0        - pretul initial al activului cand t = 0
    # r          - rata de crestere/drift a procesului stocastic(in cazul activelor financiare r este rata dobanzii de referinta)
    # sigma      - volatilitatea procesului stocastic(in cazul activelor financiare 
    #             putem alege sigma drept abaterea standard a randamentelor activlui calculate la fiecare moment de timp)
    # N          - numar de simulari
    # h          - granularitatea schemei Euler
    # T          - nr termenului final de timp T (T este exprimat in ani)
    # dt         - parametru de scalare a procesului la fiecare moment de timp
    # timeSteps  - numarul momentelor de timp h * T, 
    #              am declarat variabila in mod redundat, alegand "timeSteps" in loc de "h", avand ca obiectiv claritatea
    # alpha      - parametru real care scaleaza procesul de ajustare prin raportarea la minimul/maximul traiectoriei pana la momentul [i-j]
    # j          - parametru natural - lungimea ferestrei in care evaluam minimum/maximul traiectoriei
    # plot       - parametru boolean: True - Functia genereaza ploturi ale traiectoriilor, ale distributiei la momentul T, ale randamentelor la fiecare moment dt, si a distributiei randamentelor

    timeSteps = int(h*T)
    dt = 1/timeSteps

    r_scalat = r * T
    # initializam pasul intiliat dSt cu 0
    dSt = np.zeros(N)

    # declaram si initializam matricea care stocheaza traiectorile la fiecare moment de timp
    St_vector = np.zeros((N, timeSteps+1))
    St_vector[ : , 0 ] = S_0
    dS_vector = np.zeros((N, timeSteps+1))

    for i in range(1,timeSteps+1):

        # Minimul si Maximul calculate conform unei ferestre de timp j mobile
        max_mobil = np.max(St_vector[:, max(0, i-j):i], axis=1)
        min_mobila = np.min(St_vector[:, max(0, i-j):i], axis=1)

        # Compute correction term
        
        correction = alpha * ((max_mobil + min_mobila - 2 * St_vector[:, i - 1]) ** 2)
        correction = np.clip(correction, -1e6, 1e6) 

        # ecuatia stocastica dSt = St*r*correction*dt + St*sigma*N(0,1)*dt
        dSt = (St_vector[: , i - 1] + correction) * r_scalat * dt + St_vector[: , i - 1] * np.random.normal(0,1,N) * sigma * np.sqrt(dt)
        
        # inserarea randamentelor intr-un vector, in vederea analizei acestora
        dS_vector[:, i] = dSt

        # actualizarea schemei Euler
        St_vector[:, i] = St_vector[:, i-1] + dSt

    # plot
    # if(plot == False) => functia nu ploteaza traiectoriile
    # if(plot == True)  => functia ploteaza traiectoriile
    if(plot == True):

        # declaram dimenisunile graficului
        plt.figure(figsize=(10,6))
        for i in range(N):  # Ploteaza fiecare traiectorie a pretului
            plt.plot(St_vector[i, :], linewidth=1)

        # Setarile Graficului
        plt.title(f"Simulare Monte Carlo a Procesului Stocastic Ajustat min/max", fontsize=16)
        plt.xlabel("Time Steps", fontsize=12)
        plt.ylabel("Stock Price at t (St)", fontsize=12)
        plt.grid(True)
        plt.show()

        plt.hist(St_vector[ : , -1], bins=50, color='skyblue', edgecolor='black')
        plt.xlabel('Distributia lui ST')
        plt.ylabel('Densitate')
        plt.title('Graficul Densitatii lui ST')
        plt.show()

        plt.hist(dS_vector[ : , -1], bins=50, color='green', edgecolor='black')
        plt.xlabel('Distributia randamentelor dS')
        plt.ylabel('Densitate')
        plt.title('Graficul Densitatii lui dS')
        plt.show()

    # Euler_Ajustat_Patratic_Sim()[0] - vectorul celor N traiectori simulate la momentul final T
    # Euler_Ajustat_Patratic_Sim()[1] - vectorul celor N randamente simulate la momentul final T
    return St_vector[:,-1], dS_vector[ : , -1]

def Euler_Ajustat_Algoritmic_Patratic_Sim(S_0, T , h , r, sigma, N, alpha, j, plot = False):

    # S_0        - pretul initial al activului cand t = 0
    # r          - rata de crestere/drift a procesului stocastic(in cazul activelor financiare r este rata dobanzii de referinta)
    # sigma      - volatilitatea procesului stocastic(in cazul activelor financiare 
    #             putem alege sigma drept abaterea standard a randamentelor activlui calculate la fiecare moment de timp)
    # N          - numar de simulari
    # h          - granularitatea schemei Euler
    # T          - nr termenului final de timp T (T este exprimat in ani)
    # dt         - parametru de scalare a procesului la fiecare moment de timp
    # timeSteps  - numarul momentelor de timp h * T, 
    #              am declarat variabila in mod redundat, alegand "timeSteps" in loc de "h", avand ca obiectiv claritatea
    # alpha      - parametru real care scaleaza procesul de ajustare prin raportarea la minimul/maximul traiectoriei pana la momentul [i-j]
    # j          - parametru natural - lungimea ferestrei in care evaluam minimum/maximul traiectoriei
    # plot       - parametru boolean: True - Functia genereaza ploturi ale traiectoriilor, ale distributiei la momentul T, ale randamentelor la fiecare moment dt, si a distributiei randamentelor

    timeSteps = int(h*T)
    dt = 1/timeSteps

    r_scalat = r * T
    # initializam pasul intiliat dSt cu 0
    dSt = np.zeros(N)

    # declaram si initializam matricea care stocheaza traiectorile la fiecare moment de timp
    St_vector = np.zeros((N, timeSteps+1))
    St_vector[ : , 0 ] = S_0
    dS_vector = np.zeros((N, timeSteps+1))

    for i in range(1,timeSteps+1):

        # Minimul si Maximul calculate conform unei ferestre de timp j mobile
        max_mobil = np.max(St_vector[:, max(0, i - j):i], axis=1)
        min_mobil = np.min(St_vector[:, max(0, i - j):i], axis=1)

        # Creăm măști booleene pentru actualizare element-wise
        mask_max = St_vector[:, i - 1] > max_mobil
        mask_min = St_vector[:, i - 1] < min_mobil

        # Inițializăm Max_local și Min_local cu valorile calculate anterior
        Max_local = np.copy(max_mobil)
        Min_local = np.copy(min_mobil)

        # Aplicăm condițiile pe vector
        Max_local[mask_max] = St_vector[:, i - 1][mask_max] + (max_mobil[mask_max] - min_mobil[mask_max])
        Min_local[mask_min] = St_vector[:, i - 1][mask_min] - (max_mobil[mask_min] - min_mobil[mask_min])

        # Termenul de corecție
        correction = alpha * ((Max_local + Min_local - 2 * St_vector[:, i - 1]) ** 2)
        correction = np.clip(correction, -1e6, 1e6) 

        # ecuatia stocastica dSt = St*r*correction*dt + St*sigma*N(0,1)*dt
        dSt = (St_vector[: , i - 1] + correction) * r_scalat * dt + St_vector[: , i - 1] * np.random.normal(0,1,N) * sigma * np.sqrt(dt)
        
        # inserarea randamentelor intr-un vector, in vederea analizei acestora
        dS_vector[:, i] = dSt

        # actualizarea schemei Euler
        St_vector[:, i] = St_vector[:, i-1] + dSt

    # plot
    # if(plot == False) => functia nu ploteaza traiectoriile
    # if(plot == True)  => functia ploteaza traiectoriile
    if(plot == True):

        # declaram dimenisunile graficului
        plt.figure(figsize=(10,6))
        for i in range(N):  # Ploteaza fiecare traiectorie a pretului
            plt.plot(St_vector[i, :], linewidth=1)

        # Setarile Graficului
        plt.title(f"Simulare Monte Carlo a Procesului Stocastic Ajustat min/max", fontsize=16)
        plt.xlabel("Time Steps", fontsize=12)
        plt.ylabel("Stock Price at t (St)", fontsize=12)
        plt.grid(True)
        plt.show()

        plt.hist(St_vector[ : , -1], bins=50, color='skyblue', edgecolor='black')
        plt.xlabel('Distributia lui ST')
        plt.ylabel('Densitate')
        plt.title('Graficul Densitatii lui ST')
        plt.show()

        plt.hist(dS_vector[ : , -1], bins=50, color='green', edgecolor='black')
        plt.xlabel('Distributia randamentelor dS')
        plt.ylabel('Densitate')
        plt.title('Graficul Densitatii lui dS')
        plt.show()

    # Euler_Ajustat_Patratic_Sim()[0] - vectorul celor N traiectori simulate la momentul final T
    # Euler_Ajustat_Patratic_Sim()[1] - vectorul celor N randamente simulate la momentul final T
    return St_vector[:,-1], dS_vector[ : , -1]

def OutputSim():
    np.random.seed(42)
    S_0 = 100
    T = 1
    h = 10000
    sigma = 0.2
    N = 10000
    
    # in functie de alegerea parametrilor pot aparea explozii si instabilitate numerica.
    r = 0.0001
    sigma = 0.2
    # alpha 1 este ponderea corectiei in modelul liniar
    alpha1 = 30
    # alpha 2 este ponderea corectiei in modelul patratic
    alpha2 = 100
    j = 50
    
    plot = True

    ST_v1 = Monte_Carlo_Simulation(S_0, T , h , r, sigma, N, plot)[0]
    ST_v2 = Euler_Ajustat_Simulation(S_0, T , h , r, sigma, N, alpha1, j, plot)[0]
    ST_v3 = Euler_Ajustat_Patratic_Sim(S_0, T , h , r, sigma, N, alpha2, j, plot)[0]
    ST_v4 = Euler_Ajustat_Algoritmic_Patratic_Sim(S_0, T , h , r, sigma, N, alpha2, j, plot)[0]

    dS_v1 = Monte_Carlo_Simulation(S_0, T , h , r, sigma, N)[1]
    dS_v2 = Euler_Ajustat_Simulation(S_0, T , h , r, sigma, N, alpha1, j)[1]
    dS_v3 = Euler_Ajustat_Patratic_Sim(S_0, T , h , r, sigma, N, alpha2, j)[1]
    dS_v4 = Euler_Ajustat_Algoritmic_Patratic_Sim(S_0, T , h , r, sigma, N, alpha2, j)[1]

    print("\n")
    print(f"{'Parametrii Simulare Model Simplu de difuze cu drift:'}\n{'S_0 = '}{S_0}{', r = '}{r}{', sigma = '}{sigma}{', h = '}{h}{' zile/an '}{', T = '}{T}{' an(i), N = '}{N}{' simulari'}")
    print(f"{'Parametru estimat':<25}{'Timp (T ani)':<20}{'Valoare':<15}")
    print("="*80)
    print(f"{'Mean of St at T':<30}{T:<20}{np.mean(ST_v1):<15}")
    print(f"{'Stdev/Volatility of St at T':<30}{T:<20}{np.std(ST_v1):<15}")
    print(f"{'Variance of St at T':<30}{T:<20}{np.var(ST_v1):<15}")
    print("\n")

    print("\n")
    print(f"{'Parametrii Simulare Model de difuze cu drift ajustat dupa min/max:'}\n{'S_0 = '}{S_0}{', r = '}{r}{', sigma = '}{sigma}{', h = '}{h}{' zile/an '}{', T = '}{T}{' an(i), N = '}{N}{' simulari, coef. de pondere: '}{alpha1}{' ,fereastra mobila de '}{j}{' zile.'}")
    print(f"{'Parametru estimat':<25}{'Timp (T ani)':<20}{'Valoare':<15}")
    print("="*80)
    print(f"{'Mean of St at T':<30}{T:<20}{np.mean(ST_v2):<15}")
    print(f"{'Stdev/Volatility of St at T':<30}{T:<20}{np.std(ST_v2):<15}")
    print(f"{'Variance of St at T':<30}{T:<20}{np.var(ST_v2):<15}")
    print("\n")

    print("\n")
    print(f"{'Parametrii Simulare Model de difuze cu drift ajustat patratic dupa min/max:'}\n{'S_0 = '}{S_0}{', r = '}{r}{', sigma = '}{sigma}{', h = '}{h}{' zile/an '}{', T = '}{T}{' an(i), N = '}{N}{' simulari, coef. de pondere: '}{alpha2}{' ,fereastra mobila de '}{j}{' zile.'}")
    print(f"{'Parametru estimat':<25}{'Timp (T ani)':<20}{'Valoare':<15}")
    print("="*80)
    print(f"{'Mean of St at T':<30}{T:<20}{np.mean(ST_v3):<15}")
    print(f"{'Stdev/Volatility of St at T':<30}{T:<20}{np.std(ST_v3):<15}")
    print(f"{'Variance of St at T':<30}{T:<20}{np.var(ST_v3):<15}")
    print("\n")

    print("\n")
    print(f"{'Parametrii Simulare Model de difuze cu drift ajustat patratic si algoritmic dupa min/max:'}\n{'S_0 = '}{S_0}{', r = '}{r}{', sigma = '}{sigma}{', h = '}{h}{' zile/an '}{', T = '}{T}{' an(i), N = '}{N}{' simulari, coef. de pondere: '}{alpha2}{' ,fereastra mobila de '}{j}{' zile.'}")
    print(f"{'Parametru estimat':<25}{'Timp (T ani)':<20}{'Valoare':<15}")
    print("="*80)
    print(f"{'Mean of St at T':<30}{T:<20}{np.mean(ST_v4):<15}")
    print(f"{'Stdev/Volatility of St at T':<30}{T:<20}{np.std(ST_v4):<15}")
    print(f"{'Variance of St at T':<30}{T:<20}{np.var(ST_v4):<15}")
    print("\n")

    print("\n")
    print(f"{'Parametrii Simulare Model Simplu de difuze cu drift:'}\n{'S_0 = '}{S_0}{', r = '}{r}{', sigma = '}{sigma}{', h = '}{h}{' zile/an '}{', T = '}{T}{' an(i), N = '}{N}{' simulari'}")
    print(f"{'Parametru estimat':<25}{'Timp (T ani)':<20}{'Valoare':<15}")
    print("="*80)
    print(f"{'Mean of St at T':<30}{T:<20}{np.mean(dS_v1):<15}")
    print(f"{'Stdev/Volatility of St at T':<30}{T:<20}{np.std(dS_v1):<15}")
    print(f"{'Variance of St at T':<30}{T:<20}{np.var(dS_v1):<15}")
    print("\n")

    print("\n")
    print(f"{'Parametrii Simulare Model de difuze cu drift ajustat dupa min/max:'}\n{'S_0 = '}{S_0}{', r = '}{r}{', sigma = '}{sigma}{', h = '}{h}{' zile/an '}{', T = '}{T}{' an(i), N = '}{N}{' simulari, coef. de pondere: '}{alpha1}{' ,fereastra mobila de '}{j}{' zile.'}")
    print(f"{'Parametru estimat':<25}{'Timp (T ani)':<20}{'Valoare':<15}")
    print("="*80)
    print(f"{'Mean of St at T':<30}{T:<20}{np.mean(dS_v2):<15}")
    print(f"{'Stdev/Volatility of St at T':<30}{T:<20}{np.std(dS_v2):<15}")
    print(f"{'Variance of St at T':<30}{T:<20}{np.var(dS_v2):<15}")
    print("\n")

    print("\n")
    print(f"{'Parametrii Simulare Model de difuze cu drift ajustat patratic dupa min/max:'}\n{'S_0 = '}{S_0}{', r = '}{r}{', sigma = '}{sigma}{', h = '}{h}{' zile/an '}{', T = '}{T}{' an(i), N = '}{N}{' simulari, coef. de pondere: '}{alpha2}{' ,fereastra mobila de '}{j}{' zile.'}")
    print(f"{'Parametru estimat':<25}{'Timp (T ani)':<20}{'Valoare':<15}")
    print("="*80)
    print(f"{'Mean of St at T':<30}{T:<20}{np.mean(dS_v3):<15}")
    print(f"{'Stdev/Volatility of St at T':<30}{T:<20}{np.std(dS_v3):<15}")
    print(f"{'Variance of St at T':<30}{T:<20}{np.var(dS_v3):<15}")
    print("\n")

    print("\n")
    print(f"{'Parametrii Simulare Model de difuze cu drift ajustat patratic si algoritmic dupa min/max:'}\n{'S_0 = '}{S_0}{', r = '}{r}{', sigma = '}{sigma}{', h = '}{h}{' zile/an '}{', T = '}{T}{' an(i), N = '}{N}{' simulari, coef. de pondere: '}{alpha2}{' ,fereastra mobila de '}{j}{' zile.'}")
    print(f"{'Parametru estimat':<25}{'Timp (T ani)':<20}{'Valoare':<15}")
    print("="*80)
    print(f"{'Mean of St at T':<30}{T:<20}{np.mean(dS_v4):<15}")
    print(f"{'Stdev/Volatility of St at T':<30}{T:<20}{np.std(dS_v4):<15}")
    print(f"{'Variance of St at T':<30}{T:<20}{np.var(dS_v4):<15}")
    print("\n")

    # Analizand parametrii distributiilor ce rezulta in urma simularilor putem observa ca factorul de corectie poate modifica media procesului la momentul final T,
    # randamentele procesului, volatilitatea observata a procesului si forma distributiei randamentelor si a preturilor, "aplatizand" sau "ingustand" distributiile rezultate.

def main():
    #np.random.seed(42)
    OutputSim()

if __name__ == "__main__":
    main()