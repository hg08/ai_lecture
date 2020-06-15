import pickle


class account:
    def __init__(self, id, balance):
        self.id = id
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount

    def withdraw(self, amount):
        self.balance -= amount


myac = account('100', 100)
myac.deposit(800)
myac.withdraw(500)
fd = open('archive', 'wb')
pickle.dump(myac, fd)
fd.close()          # 400
myac.deposit(200)
fd = open('archive', 'rb')
myac = pickle.load(fd)
fd.close()
print(myac.balance)
