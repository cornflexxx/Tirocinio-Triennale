import numpy as np
import os

# Configurazione
output_dir = "datasets"
output_file = "smooth_100M.bin"
num_elements = 100_000_000  # 100 milioni di float32

# Crea la directory se non esiste
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, output_file)

# Genera dati: gaussiana + rumore lieve
x = np.linspace(-5, 5, num_elements, dtype=np.float32)
data = np.exp(-x**2) + 0.01 * np.random.randn(num_elements).astype(np.float32)

# Salva il dataset in formato binario
data.tofile(output_path)

print(f"Dataset salvato in: {output_path}")
