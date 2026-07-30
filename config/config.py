# =======================================
# PARAMETERS FOR SIMULATION
# =======================================

# --------- Circuit parameters ------------------------------------
N_QUBITS = 4
N_LAYERS = 4
N_BITS   = N_QUBITS * N_LAYERS * 5

# --------- Data parameters ---------------------------------------
TTS_SEED = 42
SEED = 50
N_POINTS = 20 

N_HAAR_SAMPLES = 2000
N_HAAR_BINS    = 75
LAMBDA_HAAR    = 1.0

# --------- Optuna parameters -------------------------------------
N_TRIALS = 100

# --------- Mealpy parameters -------------------------------------
N_EPOCH = 30
N_POPSIZE = 20
N_PM = 0.1

