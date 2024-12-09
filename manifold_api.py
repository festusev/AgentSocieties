import requests
from typing import Optional, Dict, Any, List, Union


class LiteMarket:
    def __init__(self, data: Dict[str, Any]):
        self.id: str = data['id']

        self.creatorId: str = data['creatorId']
        self.creatorUsername: str = data['creatorUsername']
        self.creatorName: str = data['creatorName']
        self.creatorAvatarUrl: Optional[str] = data.get('creatorAvatarUrl')

        self.createdTime: int = data['createdTime']
        self.closeTime: Optional[int] = data.get('closeTime')
        self.question: str = data['question']

        self.url: str = data['url']

        self.outcomeType: str = data['outcomeType']
        self.mechanism: str = data['mechanism']

        self.probability: float = data['probability']
        self.pool: Dict[str, float] = data['pool']
        self.p: Optional[float] = data.get('p')
        self.totalLiquidity: Optional[float] = data.get('totalLiquidity')

        self.value: Optional[float] = data.get('value')
        self.min: Optional[float] = data.get('min')
        self.max: Optional[float] = data.get('max')
        self.isLogScale: Optional[bool] = data.get('isLogScale')

        self.volume: float = data['volume']
        self.volume24Hours: float = data['volume24Hours']

        self.isResolved: bool = data['isResolved']
        self.resolutionTime: Optional[int] = data.get('resolutionTime')
        self.resolution: Optional[str] = data.get('resolution')
        self.resolutionProbability: Optional[float] = data.get('resolutionProbability')

        self.uniqueBettorCount: int = data['uniqueBettorCount']
        self.lastUpdatedTime: Optional[int] = data.get('lastUpdatedTime')
        self.lastBetTime: Optional[int] = data.get('lastBetTime')

        self.token: Optional[str] = data.get('token')
        self.siblingContractId: Optional[str] = data.get('siblingContractId')

class FullMarket(LiteMarket):
    def __init__(self, data: Dict[str, Any]):
        super().__init__(data)

        # Multi markets only
        self.answers: Optional[List[Dict[str, Any]]] = data.get('answers')
        self.shouldAnswersSumToOne: Optional[bool] = data.get('shouldAnswersSumToOne')
        self.addAnswersMode: Optional[str] = data.get('addAnswersMode')

        # Poll-only attributes
        self.options: Optional[List[Dict[str, Union[str, int]]]] = data.get('options')

        # Bounty-only attributes
        self.totalBounty: Optional[float] = data.get('totalBounty')
        self.bountyLeft: Optional[float] = data.get('bountyLeft')

        # Rich text and other metadata
        self.description: Dict[str, Any] = data['description']
        self.textDescription: str = data['textDescription']
        self.coverImageUrl: Optional[str] = data.get('coverImageUrl')
        self.groupSlugs: Optional[List[str]] = data.get('groupSlugs')


BASE_URL = "api.manifold.markets"
MARKET_URL = BASE_URL + "/v0/markets"
def get_markets(**kwargs) -> LiteMarket:
    resp = requests.get(MARKET_URL, kwargs)
    data = resp.json()

    return LiteMarket(data)

def get_market(marketId: str, **kwargs) -> FullMarket:
    resp = requests.get(MARKET_URL + "/" + marketId, kwargs)
    data = resp.json()

    return FullMarket(data)
