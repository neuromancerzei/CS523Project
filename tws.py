from ib_insync import IB, Future, util, Order
from gensim.models import Word2Vec
from sentiment_model import NeuralNetwork,load
from lda_model import get_topics
import numpy
import torch
def client():
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


    account = ib.accountSummary()
    balance = next((item for item in account if item.tag == 'TotalCashBalance'), None)
    print("Account Balance:", balance.value)
    return ib, contract

nn_model = load()

latest_news = get_topics()
vocab_model = Word2Vec.load("dataset/word2vec300.model")

input = [numpy.mean([vocab_model.wv[w] for w in row], axis=0) for row in latest_news]
output = nn_model(torch.tensor(input))

_, predicted = torch.max(output, 1)

sentiment = torch.mean(predicted.to(torch.float)).item()


mean_price = 6000
action='SELL'
low_price = mean_price - 50
high_price = mean_price + 50


price = (sentiment - 1) * 50 + mean_price
order = Order(action=action, totalQuantity=1, orderType='LIMIT', lmtPrice=price)
ib, contract = client()
trade = ib.placeOrder(contract, order)

# order = Order(action='SELL', totalQuantity=1, orderType='LIMIT', lmtPrice=6000)
# trade = ib.placeOrder(contract, order)
#
# account = ib.accountSummary()
# balance = next((item for item in account if item.tag == 'TotalCashBalance'), None)
# print("Account Balance:", balance.value)


if __name__ == '__main__':
    print("ok")