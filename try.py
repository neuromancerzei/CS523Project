from ib_insync import IB, Future, util, Order

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=2)
ib

if __name__ == '__main__':
    print("ok")
