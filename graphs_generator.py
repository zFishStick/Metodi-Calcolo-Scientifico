import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("risultati_win_julia.csv")
df = df.sort_values("dimensione")

# TEMPO vs DIMENSIONE
plt.figure()
plt.plot(df["dimensione"], df["tempo"], marker='o')
plt.xlabel("Dimensione matrice")
plt.ylabel("Tempo (s)")
plt.title("Tempo vs Dimensione")
plt.xscale("log")
plt.grid()
plt.savefig("tempo_vs_dimensione.png")

# ERRORE vs DIMENSIONE
plt.figure()
plt.plot(df["dimensione"], df["errore"], marker='o')
plt.xlabel("Dimensione matrice")
plt.ylabel("Errore relativo")
plt.title("Errore vs Dimensione")
plt.yscale("log") 
plt.xscale("log")
plt.grid()
plt.savefig("errore_vs_dimensione.png")

# MEMORIA vs DIMENSIONE
plt.figure()
plt.plot(df["dimensione"], df["memoria"], marker='o')
plt.xlabel("Dimensione matrice")
plt.ylabel("Memoria (MB)")
plt.title("Memoria vs Dimensione")
plt.xscale("log")
plt.grid()
plt.savefig("memoria_vs_dimensione.png")

plt.show()