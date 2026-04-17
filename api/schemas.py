
from typing import Optional, Literal
from pydantic import BaseModel, Field

NEIGHBORHOODS = [
    "CENTRAL", "DELRIDGE", "DOWNTOWN", "EAST", "GREATER DUWAMISH",
    "LAKE UNION", "MAGNOLIA / QUEEN ANNE", "NORTH", "NORTHEAST",
    "NORTHWEST", "Other", "SOUTHEAST", "SOUTHWEST",
]


class BuildingInput(BaseModel):


    year_built: int = Field(..., ge=1800, le=2016, description="Year building was constructed")
    property_gfa_total: float = Field(..., gt=0, le=1e8, description="Total floor area (sq ft)")
    number_of_floors: int = Field(..., ge=1, le=200, description="Number of floors")
    primary_property_type: Literal["Office", "Retail", "Industrial", "Institutional", "Hospitality", "Other"] = Field(..., description="Main use of building")
    neighborhood: str = Field(..., min_length=1, max_length=50, description="Neighborhood name")
    energystar_score: Optional[float] = Field(None, ge=1, le=100, description="ENERGY STAR score if certified")
    property_gfa_parking: Optional[float] = Field(default=0, ge=0, le=1e8, description="Parking area (sq ft)")
    number_of_buildings: Optional[int] = Field(default=1, ge=1, le=100)
    largest_property_use_type_gfa: Optional[float] = Field(default=None, ge=0, le=1e8)
    council_district_code: Optional[float] = Field(default=7.0, ge=1, le=10)
    latitude: Optional[float] = Field(default=47.6062, ge=47.0, le=48.0)
    longitude: Optional[float] = Field(default=-122.3321, ge=-123.0, le=-122.0)
    has_electricity: Optional[int] = Field(default=1, ge=0, le=1)
    has_natural_gas: Optional[int] = Field(default=1, ge=0, le=1)
    has_steam: Optional[int] = Field(default=0, ge=0, le=1)
    has_multiple_uses: Optional[int] = Field(default=0, ge=0, le=1)
    use_type_count: Optional[int] = Field(default=1, ge=1, le=5)
    building_type: Optional[str] = Field(default="NonResidential", description="Building type")


