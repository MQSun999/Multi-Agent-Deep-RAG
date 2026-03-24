from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

class Categ(str, Enum):
    DEFAULT = "Industrial_Automation_Safety"
    OTHER = "Other"

class Tech(str, Enum):
    GENERAL = "General"
    PRESSURE_TRANSMITTER = "Pressure_Transmitter"
    TEMPERATURE_TRANSMITTER = "Temperature_Transmitter"
    CONTROL_VALVE = "Control_Valve"
    FLOWMETER = "Flowmeter"
    SOLENOID_VALVE = "Solenoid_Valve"
    LOOP_REGULATOR = "Loop_Regulator"
    ELECTRIC_ACTUATOR = "Electric_Actuator"
    DIGITAL_DISPLAY = "Digital_Display"
    RECORDING_INSTRUMENT = "Recording_Instrument"

class ChunkMetaData(BaseModel):
    category: Optional[Categ] = Field(
        default=None, description="问题类别"
    )
    technology: Optional[Tech] = Field(
        default=None,
        description="问题涉及的技术类型('Pressure_Transmitter', 'Temperature_Transmitter', 'Control_Valve', etc.)",
    )
