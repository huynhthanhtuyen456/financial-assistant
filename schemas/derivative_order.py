from datetime import date, datetime
from pydantic import BaseModel, Json, ConfigDict


"""
Data Model
"""
class DerivativeOrder(BaseModel):
    id: int
    side: str
    accountNo: str
    investorId: str
    symbol: str
    price: float
    quantity: int
    orderType: str
    orderStatus: str
    fillQuantity: int
    lastQuantity: int
    lastPrice: float
    averagePrice: float
    transDate: date
    createdDate: datetime
    modifiedDate: datetime
    taxRate: float
    feeRate: float
    leaveQuantity: int
    canceledQuantity: int
    priceSecure: float
    custody: str
    channel: str
    loanPackageId: int
    initialRate: float
    error: str | None = None
    text: str | None = None


class DerivativeOrderListModel(BaseModel):
    id: int
    side: str
    accountNo: str
    investorId: str
    symbol: str
    price: float
    quantity: int
    orderType: str
    orderStatus: str
    fillQuantity: int
    lastQuantity: int
    lastPrice: float
    averagePrice: float
    transDate: date
    createdDate: datetime
    modifiedDate: datetime
    reports: list
    taxRate: float
    exchangeFeeRate: float
    feeRate: float
    leaveQuantity: int
    canceledQuantity: int
    error: str | None = None
    text: str | None = None
    priceSecure: float
    custody: str
    execType: str
    maker: str
    channel: str
    metadata: Json
    exDestination: str
    loanPackageId: int
    initialRate: float
    extraInfo: dict


class DerivativeLoanPackage(BaseModel):
    id: float
    name: str
    source: str
    initialRate: float
    maintenanceRate: float
    liquidRate: float
    tradingFee: dict
    symbolTypes: list[str]


class DerivativePPSEModel(BaseModel):
    investorAccountId: str
    ppse: float
    price: float
    qmaxLong: float
    qmaxShort: float


class DerivativeOrderModel(BaseModel):
    id: int
    side: str
    accountNo: str
    investorId: str
    symbol: str
    price: float
    quantity: float
    orderType: str
    orderStatus: str
    fillQuantity: int
    lastQuantity: int
    lastPrice: float
    averagePrice: float
    transDate: str
    createdDate: str
    modifiedDate: str
    taxRate: float
    feeRate: float
    leaveQuantity: int
    canceledQuantity: int
    priceSecure: float
    custody: str
    channel: str
    loanPackageId: int
    initialRate: float
    error: str | None


class DerivativeDealModel(BaseModel):
    id: int
    symbol: str
    accountNo: str
    status: str
    loanPackageId: int
    side: str
    secure: float
    accumulateQuantity: int
    closedQuantity: int
    openQuantity: int
    closedQuantity: int
    overNightQuantity: int
    breakEvenPrice: float
    costPrice: float
    costPriceVM: float
    averageCostPrice: float
    averageClosePrice: float
    totalUnrealizedProfit: float
    totalUnrealizedTaxAndFee: float
    totalRealizedProfit: float
    totalRealizedTaxAndFee: float
    estimateRemainFee: float
    estimateRemainTax: float
    totalRealizedPositionFee: float
    totalUnrealizedPositionFee: float
    maturityFee: float
    createdDate: str
    modifiedDate: str


class DerivativeDealRisk(BaseModel):
    id: int
    dealId: int
    accountNo: str
    status: str
    takeProfitEnabled: bool
    stopLossEnabled: bool
    takeProfitStrategy: str
    stopLossStrategy: str
    takeProfitOrderType: str
    stopLossOrderType: str
    takeProfitRate: float
    stopLossRate: float
    takeProfitDeltaPrice: float
    takeProfitDeltaPrice: float
    takeProfitOrderDeltaPrice: float
    stopLossOrderDeltaPrice: float
    autoHandleWarning: bool
    createdDate: str
    modifiedDate: str


class DerivativeDealRiskAccountModel(BaseModel):
    accountNo: int
    autoHandleWarning: int
    status: str
    takeProfitEnabled: bool
    stopLossEnabled: bool
    takeProfitStrategy: str
    stopLossStrategy: str
    takeProfitOrderType: str
    stopLossOrderType: str
    takeProfitRate: float
    stopLossRate: float
    takeProfitDeltaPrice: float
    takeProfitOrderDeltaPrice: float
    stopLossOrderDeltaPrice: float
    autoHandleWarning: bool
    createdDate: str
    modifiedDate: str


"""
Request model
"""
class DerivativeOrdersRequestModel(BaseModel):
    symbol: str
    side: str
    orderType: str
    price: float
    quantity: float
    loanPackageId: int
    accountNo: str


class DerivativeDealRiskRequestModel(BaseModel):
    takeProfitEnabled: bool
    stopLossEnabled: bool
    takeProfitStrategy: str
    stopLossStrategy: str
    takeProfitOrderType: str
    stopLossOrderType: str
    takeProfitRate: float
    stopLossRate: float
    takeProfitDeltaPrice: float
    takeProfitDeltaPrice: float
    takeProfitOrderDeltaPrice: float
    stopLossOrderDeltaPrice: float


class DerivativeDealRiskAccountRequestModel(BaseModel):
    status: str
    takeProfitEnabled: bool
    stopLossEnabled: bool
    takeProfitStrategy: str
    stopLossStrategy: str
    takeProfitRate: float
    stopLossRate: float
    takeProfitDeltaPrice: float
    takeProfitDeltaPrice: float


"""
Response Model
"""
class DerivativePPSEResponseModel(BaseModel):
    status: bool
    message: str
    data: DerivativePPSEModel | None = None


class DerivativeOrdersResponseModel(BaseModel):
    status: bool
    message: str
    data: dict | None = None
    # data: DerivativeOrder | None = None


class ListDerivativeOrdersResponseModel(BaseModel):
    status: bool
    message: str
    data: list[DerivativeOrderListModel]



class DerivativeDealsResponseModel(BaseModel):
    status: bool
    message: str
    data: list[DerivativeDealModel] | None = None


class DerivativeDealRiskResponseModel(BaseModel):
    status: bool
    message: str
    data: DerivativeDealRiskRequestModel | None = None


class DerivativeDealRiskAccountResponseModel(BaseModel):
    status: bool
    message: str
    data: DerivativeDealRiskAccountRequestModel | None = None


class ClosedDerivativeDealResponseModel(BaseModel):
    status: bool
    message: str
    data: dict | None = None


class DerivativeLoanPackageResponseModel(BaseModel):
    status: bool
    message: str
    data: list[DerivativeLoanPackage] | None = None
