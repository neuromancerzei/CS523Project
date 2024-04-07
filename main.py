from ib_insync import IB, Future, util, Order



ib = IB()
ib.connect('127.0.0.1', 7497, clientId=4)
ib
# Update the contract details as needed
contract = Future(symbol='ES', lastTradeDateOrContractMonth='202406', exchange='CME')

# Attempt to qualify the contract
try:
    qualified_contract = ib.qualifyContracts(contract)[0]
    print("Contract qualified successfully:", qualified_contract)
except Exception as e:
    print(f"Error qualifying contract: {e}")
    qualified_contract = None

if qualified_contract:
    # Request historical data for the qualified contract
    historical_data = ib.reqHistoricalData(
        qualified_contract, endDateTime='', durationStr='90 D',
        barSizeSetting='1 hour', whatToShow='MIDPOINT', useRTH=True)

    if historical_data:
        df = util.df(historical_data)
        print(df.head())
    else:
        print("No historical data returned.")
else:
    print("Unable to request historical data due to contract issues.")


# ib.reqMarketDataType(3)  # Switch to delayed data if necessary
#
# historical_data = ib.reqHistoricalData(
#     contract, endDateTime='', durationStr='30 D',
#     barSizeSetting='1 hour', whatToShow='MIDPOINT', useRTH=True)
#
# df = util.df(historical_data)
# print(df.head())
account = ib.accountSummary()
balance = next((item for item in account if item.tag == 'TotalCashBalance'), None)
print("Account Balance:", balance.value)


# order = Order(action='SELL', totalQuantity=1, orderType='LIMIT', lmtPrice=6000)
# trade = ib.placeOrder(contract, order)
#
# account = ib.accountSummary()
# balance = next((item for item in account if item.tag == 'TotalCashBalance'), None)
# print("Account Balance:", balance.value)


if __name__ == '__main__':
    print("ok")
