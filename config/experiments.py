from dataclasses import dataclass, field

@dataclass
class ExperimentConfig:
    name: str                       # also doubles as the save/load filename
    subset_method: str              # 'random' | 'nystrom_global' | 'nystrom_stratified'
    optimizer: str                  # 'optuna' | 'mealpy'
    subset_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    load_from: str | None = None    # set this to skip training and load a past run instead