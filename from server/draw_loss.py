import matplotlib.pyplot as plt
import numpy as np
import re
def getloss(file,epochs=-1):
    file = open(file,encoding='UTF-8')

    t = file.read()
    file.close()
    pattern = re.compile('Epoch: \[.*\]\[.*\].*Loss .*')

    # pattern = re.compile('Epoch: \[.*\].*Acc@1 .* Acc@5')

    result = pattern.findall(t)[:epochs]
    epochs = []
    train_loss = []
    # train_loss1 = []
    # train_loss2 = []
    acc = []
    for tt in result:
        it = tt.split()
        # print(it)
        r = it[1].find(']')
        ep = it[1][1:r]
        epochs.append(eval(ep))
        train_loss.append(eval(it[10]))
        # train_loss1.append(eval(it[13]))
        # train_loss2.append(eval(it[16]))
        # acc.append(eval(it[3]))
    
    return epochs,train_loss

plt.figure()
plt.xlabel("epoch")
plt.ylabel("loss")
# plt.title("loss_k8k_cos_nn")
epochs0,tl0 = getloss("main_swav_test_30_ema.log")
epochs,tl = getloss("main_swav_test_30.log")
# epochs5,tl5,tl15,tl25 = getloss("dcv2_best.log")
# epochs8,tl8,tl18,tl28 = getloss("dcv2_best_8.log")



# plt.plot(epochs,np.float32(tl),label = 'loss0')
plt.plot(epochs0,np.float32(tl0),label = 'ema')
plt.plot(epochs,np.float32(tl),label = 'no_ema')
# plt.plot(epochs0,np.float32(tl0),label = '0loss')
# plt.plot(epochs,np.float32(tl2),label = 'loss')
# plt.plot(epochs5,np.float32(tl25),label = 'loss5')
# plt.plot(epochs8,np.float32(tl28),label = 'loss8')
plt.legend()

plt.show()

