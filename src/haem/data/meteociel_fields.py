"""
Meteociel Data Fields Configuration.

Defines all available meteorological fields from Meteociel,
organized by category as shown in the Meteociel interface.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class FieldCategory(str, Enum):
    """Categories of meteorological fields from Meteociel."""

    TEMPERATURES = "temperatures"
    PRECIPITATIONS = "precipitations"
    VENT = "vent"
    DYNAMIQUE_ALTITUDE = "dynamique_altitude"
    AUTRES = "autres"


class MeteocielField(str, Enum):
    """
    All available Meteociel fields.

    Organized according to Meteociel's interface categories.
    """

    # === TEMPÉRATURES ===
    TEMP_2M = "temp_2m"                    # Temperature at 2 meters
    TEMP_850HPA = "temp_850hpa"            # Temperature at 850 hPa
    TEMP_850HPA_ANOMALY = "temp_850hpa_anomaly"  # 850 hPa temperature anomaly
    TEMP_500HPA = "temp_500hpa"            # Temperature at 500 hPa
    TEMP_10HPA_STRAT = "temp_10hpa_strat"  # Stratospheric temperature at 10 hPa

    # === PRÉCIPITATIONS ===
    PRECIPITATION = "precipitation"         # Precipitation
    CUMUL_PRECIP = "cumul_precip"          # Cumulative precipitation
    SNOW_HEIGHT = "snow_height"            # Snow height (Hauteur neige)
    PRECIP_SUMMARY = "precip_summary"      # Precipitation summary (Résumé)

    # === VENT ===
    WIND_10M = "wind_10m"                  # Wind at 10 meters
    JET_STREAM = "jet_stream"              # Jet Stream

    # === DYNAMIQUE D'ALTITUDE ===
    ALTITUDE_1_5_PVU = "altitude_1_5_pvu"  # 1.5 PVU altitude (tropopause)
    THETA_E_850HPA = "theta_e_850hpa"      # Equivalent potential temperature 850 hPa
    THETA_W_850HPA = "theta_w_850hpa"      # Wet-bulb potential temperature 850 hPa

    # === AUTRES ===
    PRESSURE_GEOP_500HPA = "pressure_geop_500hpa"  # Pressure/Geopotential 500 hPa
    ANOMALY_GEOP_500HPA = "anomaly_geop_500hpa"    # Geopotential anomaly 500 hPa
    ISO0_Z500_1000 = "iso0_z500_1000"              # 0°C isotherm and Z500/1000
    SBCAPE_LI = "sbcape_li"                        # SBCAPE and Lifted Index

    @property
    def category(self) -> FieldCategory:
        """Return the category this field belongs to."""
        categories = {
            # Temperatures
            self.TEMP_2M: FieldCategory.TEMPERATURES,
            self.TEMP_850HPA: FieldCategory.TEMPERATURES,
            self.TEMP_850HPA_ANOMALY: FieldCategory.TEMPERATURES,
            self.TEMP_500HPA: FieldCategory.TEMPERATURES,
            self.TEMP_10HPA_STRAT: FieldCategory.TEMPERATURES,
            # Precipitations
            self.PRECIPITATION: FieldCategory.PRECIPITATIONS,
            self.CUMUL_PRECIP: FieldCategory.PRECIPITATIONS,
            self.SNOW_HEIGHT: FieldCategory.PRECIPITATIONS,
            self.PRECIP_SUMMARY: FieldCategory.PRECIPITATIONS,
            # Vent
            self.WIND_10M: FieldCategory.VENT,
            self.JET_STREAM: FieldCategory.VENT,
            # Dynamique d'altitude
            self.ALTITUDE_1_5_PVU: FieldCategory.DYNAMIQUE_ALTITUDE,
            self.THETA_E_850HPA: FieldCategory.DYNAMIQUE_ALTITUDE,
            self.THETA_W_850HPA: FieldCategory.DYNAMIQUE_ALTITUDE,
            # Autres
            self.PRESSURE_GEOP_500HPA: FieldCategory.AUTRES,
            self.ANOMALY_GEOP_500HPA: FieldCategory.AUTRES,
            self.ISO0_Z500_1000: FieldCategory.AUTRES,
            self.SBCAPE_LI: FieldCategory.AUTRES,
        }
        return categories.get(self, FieldCategory.AUTRES)

    @property
    def description(self) -> str:
        """Human-readable description."""
        descriptions = {
            self.TEMP_2M: "Temperatura a 2 metri dal suolo",
            self.TEMP_850HPA: "Temperatura a 850 hPa (~1500m)",
            self.TEMP_850HPA_ANOMALY: "Anomalia temperatura 850 hPa rispetto alla climatologia",
            self.TEMP_500HPA: "Temperatura a 500 hPa (~5500m)",
            self.TEMP_10HPA_STRAT: "Temperatura stratosferica a 10 hPa",
            self.PRECIPITATION: "Precipitazione istantanea",
            self.CUMUL_PRECIP: "Precipitazione cumulata",
            self.SNOW_HEIGHT: "Altezza neve al suolo",
            self.PRECIP_SUMMARY: "Sintesi precipitazioni",
            self.WIND_10M: "Vento a 10 metri",
            self.JET_STREAM: "Corrente a getto (Jet Stream)",
            self.ALTITUDE_1_5_PVU: "Altitudine della tropopausa dinamica (1.5 PVU)",
            self.THETA_E_850HPA: "Temperatura potenziale equivalente a 850 hPa",
            self.THETA_W_850HPA: "Temperatura potenziale del bulbo umido a 850 hPa",
            self.PRESSURE_GEOP_500HPA: "Pressione e geopotenziale a 500 hPa",
            self.ANOMALY_GEOP_500HPA: "Anomalia del geopotenziale a 500 hPa",
            self.ISO0_Z500_1000: "Isoterma 0°C e spessore 500/1000 hPa",
            self.SBCAPE_LI: "CAPE superficiale e Lifted Index (instabilità)",
        }
        return descriptions.get(self, self.value)

    @property
    def units(self) -> str:
        """Field units."""
        units = {
            self.TEMP_2M: "°C",
            self.TEMP_850HPA: "°C",
            self.TEMP_850HPA_ANOMALY: "°C",
            self.TEMP_500HPA: "°C",
            self.TEMP_10HPA_STRAT: "°C",
            self.PRECIPITATION: "mm/h",
            self.CUMUL_PRECIP: "mm",
            self.SNOW_HEIGHT: "cm",
            self.PRECIP_SUMMARY: "mm",
            self.WIND_10M: "km/h",
            self.JET_STREAM: "kt",
            self.ALTITUDE_1_5_PVU: "hPa",
            self.THETA_E_850HPA: "K",
            self.THETA_W_850HPA: "K",
            self.PRESSURE_GEOP_500HPA: "dam",
            self.ANOMALY_GEOP_500HPA: "dam",
            self.ISO0_Z500_1000: "m / dam",
            self.SBCAPE_LI: "J/kg / °C",
        }
        return units.get(self, "")

    @property
    def meteorological_significance(self) -> str:
        """Why this field is important for forecasting."""
        significance = {
            self.TEMP_2M: "Temperatura percepita al suolo, fondamentale per previsioni locali",
            self.TEMP_850HPA: "Indica la massa d'aria in quota, utile per fronti e avvezioni",
            self.TEMP_850HPA_ANOMALY: "Rivela anomalie termiche rispetto al clima medio",
            self.TEMP_500HPA: "Struttura termica della media troposfera",
            self.TEMP_10HPA_STRAT: "Monitoraggio stratosfera, sudden warming events",
            self.PRECIPITATION: "Precipitazione in corso",
            self.CUMUL_PRECIP: "Totale precipitazione attesa",
            self.SNOW_HEIGHT: "Accumulo nevoso previsto",
            self.WIND_10M: "Vento al suolo, impatto diretto",
            self.JET_STREAM: "Guida i sistemi meteorologici, chiave per previsioni",
            self.ALTITUDE_1_5_PVU: "Tropopausa dinamica, intrusioni stratosferiche",
            self.THETA_E_850HPA: "Instabilità convettiva, masse d'aria umide",
            self.THETA_W_850HPA: "Potenziale temporalesco, fronti",
            self.PRESSURE_GEOP_500HPA: "Struttura sinottica principale",
            self.ANOMALY_GEOP_500HPA: "Pattern anomali, blocking, cut-off",
            self.ISO0_Z500_1000: "Quota neve e spessore aria calda/fredda",
            self.SBCAPE_LI: "Energia convettiva, potenziale temporali severi",
        }
        return significance.get(self, "Campo meteorologico standard")


class FieldSelection(BaseModel):
    """User's selection of fields to analyze."""

    # Temperatures
    temp_2m: bool = Field(default=True, description="Include 2m temperature")
    temp_850hpa: bool = Field(default=True, description="Include 850 hPa temperature")
    temp_850hpa_anomaly: bool = Field(default=False, description="Include 850 hPa anomaly")
    temp_500hpa: bool = Field(default=False, description="Include 500 hPa temperature")
    temp_10hpa_strat: bool = Field(default=False, description="Include stratospheric temperature")

    # Precipitations
    precipitation: bool = Field(default=True, description="Include precipitation")
    cumul_precip: bool = Field(default=True, description="Include cumulative precipitation")
    snow_height: bool = Field(default=False, description="Include snow height")

    # Wind
    wind_10m: bool = Field(default=True, description="Include 10m wind")
    jet_stream: bool = Field(default=True, description="Include jet stream")

    # Upper air dynamics
    altitude_1_5_pvu: bool = Field(default=False, description="Include 1.5 PVU altitude")
    theta_e_850hpa: bool = Field(default=False, description="Include Theta-E 850 hPa")
    theta_w_850hpa: bool = Field(default=False, description="Include Theta-W 850 hPa")

    # Others
    pressure_geop_500hpa: bool = Field(default=True, description="Include Z500")
    anomaly_geop_500hpa: bool = Field(default=False, description="Include Z500 anomaly")
    iso0_z500_1000: bool = Field(default=False, description="Include 0°C isotherm")
    sbcape_li: bool = Field(default=False, description="Include CAPE/LI")

    def get_selected_fields(self) -> list[MeteocielField]:
        """Return list of selected MeteocielField enums."""
        field_mapping = {
            "temp_2m": MeteocielField.TEMP_2M,
            "temp_850hpa": MeteocielField.TEMP_850HPA,
            "temp_850hpa_anomaly": MeteocielField.TEMP_850HPA_ANOMALY,
            "temp_500hpa": MeteocielField.TEMP_500HPA,
            "temp_10hpa_strat": MeteocielField.TEMP_10HPA_STRAT,
            "precipitation": MeteocielField.PRECIPITATION,
            "cumul_precip": MeteocielField.CUMUL_PRECIP,
            "snow_height": MeteocielField.SNOW_HEIGHT,
            "wind_10m": MeteocielField.WIND_10M,
            "jet_stream": MeteocielField.JET_STREAM,
            "altitude_1_5_pvu": MeteocielField.ALTITUDE_1_5_PVU,
            "theta_e_850hpa": MeteocielField.THETA_E_850HPA,
            "theta_w_850hpa": MeteocielField.THETA_W_850HPA,
            "pressure_geop_500hpa": MeteocielField.PRESSURE_GEOP_500HPA,
            "anomaly_geop_500hpa": MeteocielField.ANOMALY_GEOP_500HPA,
            "iso0_z500_1000": MeteocielField.ISO0_Z500_1000,
            "sbcape_li": MeteocielField.SBCAPE_LI,
        }

        selected = []
        for attr_name, field_enum in field_mapping.items():
            if getattr(self, attr_name, False):
                selected.append(field_enum)

        return selected

    def get_fields_by_category(self) -> dict[FieldCategory, list[MeteocielField]]:
        """Return selected fields organized by category."""
        selected = self.get_selected_fields()
        by_category: dict[FieldCategory, list[MeteocielField]] = {}

        for field in selected:
            cat = field.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(field)

        return by_category


# Preset configurations
class FieldPresets:
    """Preset field selections for common use cases."""

    @staticmethod
    def basic() -> FieldSelection:
        """Basic fields for general forecasting."""
        return FieldSelection(
            temp_2m=True,
            temp_850hpa=True,
            precipitation=True,
            cumul_precip=True,
            wind_10m=True,
            pressure_geop_500hpa=True,
        )

    @staticmethod
    def synoptic_analysis() -> FieldSelection:
        """Fields for synoptic-scale analysis."""
        return FieldSelection(
            temp_850hpa=True,
            temp_850hpa_anomaly=True,
            temp_500hpa=True,
            jet_stream=True,
            altitude_1_5_pvu=True,
            pressure_geop_500hpa=True,
            anomaly_geop_500hpa=True,
        )

    @staticmethod
    def convective_analysis() -> FieldSelection:
        """Fields for convective/thunderstorm analysis."""
        return FieldSelection(
            temp_2m=True,
            temp_850hpa=True,
            precipitation=True,
            wind_10m=True,
            theta_e_850hpa=True,
            theta_w_850hpa=True,
            sbcape_li=True,
            iso0_z500_1000=True,
        )

    @staticmethod
    def winter_weather() -> FieldSelection:
        """Fields for winter weather analysis."""
        return FieldSelection(
            temp_2m=True,
            temp_850hpa=True,
            precipitation=True,
            cumul_precip=True,
            snow_height=True,
            wind_10m=True,
            iso0_z500_1000=True,
            pressure_geop_500hpa=True,
        )

    @staticmethod
    def full_analysis() -> FieldSelection:
        """All fields for comprehensive analysis."""
        return FieldSelection(
            temp_2m=True,
            temp_850hpa=True,
            temp_850hpa_anomaly=True,
            temp_500hpa=True,
            temp_10hpa_strat=True,
            precipitation=True,
            cumul_precip=True,
            snow_height=True,
            wind_10m=True,
            jet_stream=True,
            altitude_1_5_pvu=True,
            theta_e_850hpa=True,
            theta_w_850hpa=True,
            pressure_geop_500hpa=True,
            anomaly_geop_500hpa=True,
            iso0_z500_1000=True,
            sbcape_li=True,
        )
