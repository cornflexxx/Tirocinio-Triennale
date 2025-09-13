# HomComp: Compressore omomorfico basato su cuSZp2
La repository contiene *HomComp* il compressore lossy _Gpu-centrico_ basato su cuSZp2 e una sua integrazione all'interno di una Ring Allreduce. Il lavoro è stato eseguito durante il tirocinio per la Laurea Triennale in Informatica (A.A. 2024/2025).
Lo script python `dataset_generator.py` genera dei dataset di `float32` di dimensione variabile, che possono essere utilizzati in `test_allreduce.cu` per testare le collettive compresse.
