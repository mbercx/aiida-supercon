from enum import Enum

class RestartType(Enum):
    """Defines the supported restart modes for the EpwIntpWorkChain."""
    FROM_SCRATCH = 'FROM_SCRATCH'
    RESTART_WANNIER = 'RESTART_WANNIER'
    RESTART_PHONON = 'RESTART_PHONON'
    RESTART_EPW = 'RESTART_EPW'
    RESTART_A2F = 'RESTART_A2F'
    RESTART_ISO = 'RESTART_ISO'
    RESTART_ANISO = 'RESTART_ANISO'
    RESTART_TRANSPORT = 'RESTART_TRANSPORT'
    # You can add more modes here later
    
