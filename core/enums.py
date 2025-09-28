from enum import Enum


class FinanceTypes(Enum):
    INCOME_STATEMENT = ("incomestatement", "Kết quả kinh doanh")
    BALANCE_SHEET = ("balancesheet", "Bảng cân đối kế toán")
    CASH_FLOW = ("cashflow", "Bảng dòng tiền")

    @property
    def id(self) -> str:
        return self.value[0]

    @property
    def name(self) -> str:
        return self.value[1]


class IncomeStatement(Enum):
    REVENUE = "Doanh thu thuần"
    YEAR_REVENUE_GROWTH = "TT DT YoY%"
    QUARTER_REVENUE_GROWTH = "TT DT QoQ%"
    COST_OF_GOOD_SOLD = "Giá vốn hàng bán"
    GROSS_PROFIT = "Lợi nhuận gộp"
    OPERATION_EXPENSE = "Chi phí hoạt động"
    OPERATION_PROFIT = "Lợi nhuận hoạt động"
    YEAR_OPERATION_PROFIT_GROWTH = "TT LNHĐ YoY%"
    QUARTER_OPERATION_PROFIT_GROWTH = "TT LNHĐ QoQ%"
    INTEREST_EXPENSE = "Chi phí lãi vay"
    PRE_TAX_PROFIT = "Lợi nhuận trước thuế"
    POST_TAX_PROFIT = "Lợi nhuận sau thuế"
    SHARE_HOLDER_INCOME = "Lợi nhuận dành cho cổ đông"
    YEAR_SHARE_HOLDER_INCOME_GROWTH = "TT LNCĐ YoY%"
    QUARTER_SHARE_HOLDER_INCOME_GROWTH = "TT LNCĐ QoQ%"
    INVEST_PROFIT = "Lợi nhuận từ đầu tư"
    SERVICE_PROFIT = "Lợi nhuận từ dịch vụ"
    OTHER_PROFIT = "Lợi nhuận khác"
    PROVISION_EXPENSE = "Chi phí dự phòng"
    OPERATION_INCOME = "Thu nhập từ hoạt động kinh doanh"
    EBITDA = "EBITDA"  # Lợi nhuận trước lãi vay, thuế và khấu hao


class BalanceSheet(Enum):
    SHORT_ASSET = "Tài sản ngắn hạn"
    CASH = "Tiền và các khoản tương đương tiền"
    SHORT_INVEST = "Đầu tư ngắn hạn"
    SHORT_RECEIVABLE = "Phải thu ngắn hạn"
    INVENTORY = "Hàng tồn kho"
    LONG_ASSET = "Tài sản dài hạn"
    FIXED_ASSET = "Tài sản cố định"
    ASSET = "Tổng tài sản"
    DEBT = "Nợ phải trả"
    SHORT_DEBT = "Nợ ngắn hạn"
    LONG_DEBT = "Nợ dài hạn"
    EQUITY = "Vốn chủ sở hữu"
    CAPITAL = "Vốn điều lệ"
    CENTRAL_BANK_DEPOSIT = "Tiền gửi tại Ngân hàng Trung ương"
    OTHER_BANK_DEPOSIT = "Tiền gửi tại Ngân hàng khác"
    OTHER_BANK_LOAN = "Vay từ Ngân hàng khác"
    STOCK_INVEST = "Đầu tư chứng khoán"
    CUSTOMER_LOAN = "Cho vay khách hàng"
    BAD_LOAN = "Nợ xấu"
    PROVISION = "Dự phòng rủi ro tín dụng"
    NET_CUSTOMER_LOAN = "Cho vay khách hàng ròng"
    OTHER_ASSET = "Tài sản khác"
    OTHER_BANK_CREDIT = "Tín dụng từ Ngân hàng khác"
    OWE_OTHER_BANK = "Nợ Ngân hàng khác"
    OWE_CENTRAL_BANK = "Nợ Ngân hàng Trung ương"
    VALUABLE_PAPER = "Giấy tờ có giá"
    PAYABLE_INTEREST = "Lãi phải trả"
    RECEIVABLE_INTEREST = "Lãi phải thu"
    DEPOSIT = "Tiền gửi"
    OTHER_DEBT = "Nợ khác"
    FUND = "Quỹ"
    UN_DISTRIBUTED_INCOME = "Lợi nhuận chưa phân phối"
    MINOR_SHARE_HOLDER_PROFIT = "Lợi nhuận của cổ đông thiểu số"
    PAYABLE = "Phải trả"


class CashFlow(Enum):
    INVEST_COST = "Chi phí đầu tư"
    FROM_INVEST = "Dòng tiền từ hoạt động đầu tư"
    FROM_FINANCIAL = "Dòng tiền từ hoạt động tài chính"
    FROM_SALE = "Dòng tiền từ hoạt động kinh doanh"
    FREE_CASH_FLOW = "Dòng tiền tự do"