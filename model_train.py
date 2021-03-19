from main import *

x = np.array([[1, 5]])
y = np.array([[0, 0]])

s = Sigmoid()
class Model():
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.W1 = np.random.randn(2,3)
        self.b1 = np.random.randn(3)

        self.W2 = np.random.randn(3,2)
        self.b2 = np.random.randn(2)
    def forward(self):
        self.z1 = np.dot(self.x,self.W1)+self.b1
        self.s1 = s.forward(self.z1)
        self.z2 = np.dot(self.s1,self.W2)+self.b2
        self.s2 = s.forward(self.z2)
        self.cost = np.sum((1/(2*len(self.x)))*((self.s2-self.y)**2))

    def test(self):
        self.forward()
        print(self.W1,self.b1,self.W2,self.b1)
        print(self.cost)
        a = 0.1
        dcds = self.s2-self.y
        dsdz = (1-s.forward(self.z2))*s.forward(self.z2)
        self.W2 -= a*(np.dot((dcds*dsdz).T,self.s1).T)
        self.b2 -= a*(dcds*dsdz*1).flatten()
        dedl2 = np.dot(dcds*dsdz,self.W2.T)
        dl2dz =(1-s.forward(self.z1))*s.forward(self.z1)
        self.b1 -= a*(dedl2*dl2dz).flatten()
        self.W1 -= a*np.dot(self.x.T,dedl2*dl2dz)
        self.forward()
        print(self.W1,self.b1,self.W2,self.b1)
        print(self.cost)

        l1 = (self.s2 - self.y) * (self.s2 * (1 - self.s2))


        lx = np.dot(l1,self.W2.T)




    def train(self):
        alpha = 0.01

        a = 0.1
        for i in range(10000):

            self.forward()


            dcds = self.s2 - self.y
            dsdz = (1 - s.forward(self.z2)) * s.forward(self.z2)
            self.W2 -= a * (np.dot((dcds * dsdz).T, self.s1).T)
            self.b2 -= a * (dcds * dsdz * 1).flatten()
            dedl2 = np.dot(dcds * dsdz, self.W2.T)
            dl2dz = (1 - s.forward(self.z1)) * s.forward(self.z1)
            self.b1 -= a * (dedl2 * dl2dz).flatten()
            self.W1 -= a * np.dot(self.x.T, dedl2 * dl2dz)
            if(i%100==0):
                print(self.W1, self.b1, self.W2, self.b1)
                print(self.cost)


m = Model(x,y)

m.forward()
print(m.cost)
m.train()



