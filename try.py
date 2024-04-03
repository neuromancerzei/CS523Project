from ib_insync import IB, Future, util, Order

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=2)
ib
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

class IBClient(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
    
    def error(self, reqId, errorCode, errorString):
        print("Error:", reqId, errorCode, errorString)

    def contractDetails(self, reqId, contractDetails):
        symbol = contractDetails.contract.symbol
        print("Symbol:", symbol)

def main():
    client = IBClient()
    client.connect("127.0.0.1", 7497, clientId=123)

    # 请求所有合同详细信息
    contract = Contract()
    client.reqContractDetails(0, contract)

    client.run()

if __name__ == "__main__":
    main()


