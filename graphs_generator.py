import pandas as pd
import matplotlib.pyplot as plt

df_rust = pd.read_csv(r"C:\Users\Diagon\Desktop\Progetti\Metodi-Calcolo-Scientifico\rust\risultati_rust.csv").sort_values("dimensione")
df_matlab = pd.read_csv("risultati_win_matlab.csv").sort_values("dimensione")
df_julia = pd.read_csv("risultati_win_julia.csv").sort_values("dimensione")

# TEMPO x DIMENSIONE
plt.figure()
plt.plot(df_rust["dimensione"], df_rust["tempo"], marker='o', label="Rust")
plt.plot(df_matlab["dimensione"], df_matlab["tempo"], marker='s', label="MATLAB")
plt.plot(df_julia["dimensione"], df_julia["tempo"], marker='^', label="Julia")

plt.xlabel("Dimensione matrice")
plt.ylabel("Tempo (s)")
plt.title("Tempo vs Dimensione")
plt.xscale("log")
plt.yscale("log")
plt.grid()
plt.legend()
plt.savefig("confronto_tempo.png")

# # ERRORE x DIMENSIONE
plt.figure()
plt.plot(df_rust["dimensione"], df_rust["errore"], marker='o', label="Rust")
plt.plot(df_matlab["dimensione"], df_matlab["errore"], marker='s', label="MATLAB")
plt.plot(df_julia["dimensione"], df_julia["errore"], marker='^', label="Julia")

plt.xlabel("Dimensione matrice")
plt.ylabel("Errore relativo")
plt.title("Errore vs Dimensione")
plt.xscale("log")
plt.yscale("log")
plt.grid()
plt.legend()
plt.savefig("confronto_errore.png")

# MEMORIA x DIMENSIONE
plt.figure()
plt.plot(df_rust["dimensione"], df_rust["memoria"], marker='o', label="Rust")
plt.plot(df_matlab["dimensione"], df_matlab["memoria"], marker='s', label="MATLAB")
plt.plot(df_julia["dimensione"], df_julia["memoria"], marker='^', label="Julia")

plt.xlabel("Dimensione matrice")
plt.ylabel("Memoria (MB)")
plt.title("Memoria vs Dimensione")
plt.yscale("log")
plt.grid()
plt.legend()
plt.savefig("confronto_memoria.png")

plt.show()