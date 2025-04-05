from ibapi.client import *
from ibapi.wrapper import *
import time
import threading

port = 7496

class ConnectionApp(EClient, EWrapper):
    def __init__(self):
        EClient.__init__(self, self)
        self.connection_event = threading.Event()

    def nextValidId(self, orderId: OrderId):
        self.orderId = orderId
        self.connection_event.set()  # Signal that connection is established
    
    def nextId(self):
        self.orderId += 1
        return self.orderId
    
    def error(self, reqId, errorCode, errorString, advancedOrderReject=""):
        print(f"reqId: {reqId}, errorCode: {errorCode}, errorString: {errorString}, orderReject: {advancedOrderReject}")
    
    def historicalData(self, reqId, bar):
        print(bar.date, bar.close, bar.open, bar.high, bar.low, bar.volume)
        #bar.close
    
    def historicalDataEnd(self, reqId, start, end):
        print(f"Historical Data Ended for {reqId}. Started at {start}, ending at {end}")
        self.cancelHistoricalData(reqId)
        
#app = TestApp()
#app.connect("127.0.0.1", port, 0)
#threading.Thread(target=app.run).start()
#time.sleep(1)