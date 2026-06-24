import numpy as np
from .genes import Gene, MealpyGene, OptunaGene

class Candidate:
    """
    Base candidate class. Represents a proposal for a quantum circuit
    """
    def __init__(self, n_qubits: int, n_layers: int): #, x):
         self.n_qubits = n_qubits
         self.n_layers = n_layers
         self.n_genes = n_qubits * n_layers
         #self.x = x
         self.genes: list = []   # Will hold the list of genes
    
    def get_ith_gene(self, i:int) -> Gene:
        return self.genes[i]
    
class MealpyCandidate(Candidate):
    """
    Candidate class for Mealpy optimization.
    Splits the bitstring into 5-bit slices and obtains the candidate information.
    """
    def __init__(self, n_qubits:int , n_layers:int, bitstring: np.ndarray):
        super().__init__(n_qubits, n_layers)

        # Build the gene dictionary for this candidate by calling the MealpyGene class
        for i in range(self.n_genes):
            gene = MealpyGene(bitstring[i*5 : (i+1)*5])
            self.genes.append(gene)

class OptunaCandidate(Candidate):
    """
    Candidate class for Optuna optimization.
    """
    def __init__(self, n_qubits:int, n_layers:int, gates:list, angles:list):
        super().__init__(n_qubits, n_layers)

        # Build the gene dictionary for this candidate by calling the OptunaGene class
        for gate, angle in zip(gates, angles):
            gene = OptunaGene(gate, angle)
            self.genes.append(gene)