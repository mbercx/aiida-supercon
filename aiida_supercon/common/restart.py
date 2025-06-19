from enum import Enum
from aiida.common import AttributeDict

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
    
class RestartState(AttributeDict):
    """
    A class to store the state of the restart.
    """
    def __init__(self, namespaces, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set the default state: run everything
        for namespace in namespaces:
            self.setdefault(namespace, True)
