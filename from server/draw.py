import matplotlib.pyplot as plt
import numpy as np
import re
def getacc(file,epochs=100):
    file = open(file,encoding='UTF-8')

    t = file.read()
    file.close()
    # pattern = re.compile('Epoch: \[.*\]\[1250\].*Loss1 .*Loss2 .*')

    pattern = re.compile('Epoch: \[.*\].*Acc@1 .* Acc@5')

    result = pattern.findall(t)[:epochs]
    epochs = []
    # train_loss = []
    # train_loss1 = []
    # train_loss2 = []
    acc = []
    for tt in result:
        it = tt.split()
        r = it[1].find(']')
        ep = it[1][1:r]
        epochs.append(eval(ep))
        acc.append(eval(it[3]))
    return acc,epochs
def getblacc(file):
    file = open(file,encoding='UTF-8')

    t = file.read()
    file.close()
    # pattern = re.compile('Epoch: \[.*\]\[1250\].*Loss1 .*Loss2 .*')

    pattern = re.compile("\*   Acc@1 .*Acc@5")

    result = pattern.findall(t)[:100]
    epochs = []
    # train_loss = []
    # train_loss1 = []
    # train_loss2 = []
    acc = []
    for tt in result:
        it = tt.split()
        # r = it[1].find(']')
        # ep = it[1][1:r]
        # epochs.append(eval(ep))
        # print(it)
        acc.append(eval(it[2]))
    return acc
plt.figure()
plt.xlabel("epoch")
plt.ylabel("acc")
plt.title("Acc@1")
acc3k,epochs3k = getacc("dcv2_k5.log")
# acc5kcos,epochs5kcos = getacc("dcv2_best.log",120)
# acc3kcos,epochs3kcos = getacc("dcv2_3.log")
acc5k,epochs5k = getacc("dcv2_k5_noema.log")
# acc8k,epochs8k = getacc("dcv2_k8000_a0.2_n.log")
# accbl = getblacc("b50.log")

plt.plot(epochs3k,np.float32(acc3k),label = 'ema')
# plt.plot(epochs3kcos,np.float32(acc3kcos),label = '3000+cosine')
# plt.plot(epochs5kcos,np.float32(acc5kcos),label = '5000+cosine')
plt.plot(epochs5k,np.float32(acc5k),label = 'no_ema')
# plt.plot(epochs8k,np.float32(acc8k),label = '8000')
# plt.plot(range(len(accbl)),np.float32(accbl),label = 'baseline')
# l1 = plt.plot(epochs,np.float32(train_loss1),label = 'loss1')
# l2 = plt.plot(epochs,np.float32(train_loss2),label = 'loss2')
plt.legend()

plt.show()

