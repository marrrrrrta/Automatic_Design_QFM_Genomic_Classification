import numpy as np

GATE_MAP = {
    '000': 'H',
    '001': 'CNOT',
    '010': 'RX',
    '011': 'RY',
    '100': 'RZ',
}

ANGLE_MAP = {
    '00': np.pi,
    '10': np.pi / 2,
    '01': np.pi / 4,
    '11': np.pi / 8,
}

class Gene:
    """
    Base class that links one gene with a quantum gate.
    """
    def get_gene_info(self) -> dict:
        raise NotImplementedError

class MealpyGene(Gene):
    """
    Gene class for Mealpy optimization.
    The gene is encoded as a binary array, which will give info about the type of gate and its parameters.
    From a certain gene_bits candidate, the information of the gate and its parameters can be extracted.
    """
    def __init__(self, gene_bits:np.ndarray):
        self.bits = gene_bits
        gate_bits  = ''.join(str(int(bit)) for bit in gene_bits[:3])
        angle_bits = ''.join(str(int(bit)) for bit in gene_bits[3:5])
        self.gate = GATE_MAP.get(gate_bits, "I")
        self.angle = ANGLE_MAP.get(angle_bits, 0)
    
    def get_gene_info(self) -> dict:
        return {"bits": self.bits, "gate": self.gate, "angle": self.angle}

class OptunaGene(Gene):
    """
    Gene class for Optuna optimization.
    The gene is encoded as a gate type (string), and the angle parameter (float).
    """
    def __init__(self, gate:str, angle:float):
        self.gate = gate
        self.angle = angle

    def get_gene_info(self) -> dict:
        return {"gate": self.gate, "angle": self.angle}