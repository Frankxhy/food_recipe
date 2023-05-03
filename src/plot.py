import matplotlib.pyplot as plt
import math
import matplotlib.ticker as mtick


# print(math.sqrt(2000))


# cs7643 project
# fixed head=4
x=[4,8,12,16,20,24]
y=[8.67,8.61,8.54,8.51,8.53,8.52] # perplexity
plt.plot(x,y,'s-',color = 'r')
# plt.plot(x,k2,'o-',color = 'g',label="CNN-RLSTM")
# plt.xticks(x)
# plt.yticks(y)
plt.xlabel("the number of blocks")
plt.ylabel("perplexity")
plt.title("recipe perplexity with different number of blocks")
# plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%1.2e'))

# plt.legend(loc = "plot1")
plt.show()


x=[1,2,3,4,5]
y=[8.63,8.55,8.52,8.51,8.52] # perplexity
plt.plot(x,y,'ro-',color = 'b')
plt.xlabel("the number of multi-head")
plt.ylabel("perplexity")
plt.title("recipe perplexity with different number of multi-head")

plt.show()


