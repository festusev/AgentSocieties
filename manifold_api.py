import requests
from typing import Optional, Any
from pydantic import BaseModel, Field, TypeAdapter

class LiteMarket(BaseModel):
    id: str
    creatorId: str
    creatorUsername: str
    creatorName: str
    creatorAvatarUrl: Optional[str] = None

    createdTime: int
    closeTime: Optional[int] = None
    question: str

    url: str

    outcomeType: str
    mechanism: str

    probability: Optional[float] = None
    pool: Optional[dict[str, float]] = None
    p: Optional[float] = None
    totalLiquidity: Optional[float] = None

    # Renamed min and max to avoid conflicts
    minValue: Optional[float] = Field(None, alias="min")
    maxValue: Optional[float] = Field(None, alias="max")
    isLogScale: Optional[bool] = None

    volume: float
    volume24Hours: float

    isResolved: bool
    resolutionTime: Optional[int] = None
    resolution: Optional[str] = None
    resolutionProbability: Optional[float] = None

    uniqueBettorCount: int
    lastUpdatedTime: Optional[int] = None
    lastBetTime: Optional[int] = None

    token: Optional[str] = Field(None, description="Either 'MANA' or 'CASH'")
    siblingContractId: Optional[str] = None


class FullMarket(LiteMarket):
    answers: Optional[list[dict[str, Any]]] = None
    shouldAnswersSumToOne: Optional[bool] = None
    addAnswersMode: Optional[str] = Field(
        None, description="Options: 'ANYONE', 'ONLY_CREATOR', 'DISABLED'"
    )

    options: Optional[list[dict[str, str | int]]] = None

    totalBounty: Optional[float] = None
    bountyLeft: Optional[float] = None

    description: dict[str, Any]
    textDescription: str
    coverImageUrl: Optional[str] = None
    groupSlugs: Optional[list[str]] = None



BASE_URL = "https://api.manifold.markets"
MARKET_URL = BASE_URL + "/v0/markets"
def get_markets(**kwargs) -> list[LiteMarket]:
    resp = requests.get(MARKET_URL, kwargs)

    ta = TypeAdapter(list[LiteMarket])
    return ta.validate_json(resp.text)

def get_market(marketId: str, **kwargs) -> FullMarket:
    resp = requests.get(MARKET_URL + "/" + marketId, kwargs)

    ta = TypeAdapter(list[FullMarket])
    return ta.model_validate_json(resp.text)
