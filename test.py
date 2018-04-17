import matplotlib.pyplot as plt
x3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y3 = [30, 31, 31, 32, 33, 35, 35, 40, 47, 62]

x4 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y4 = [32, 32, 32, 33, 34, 34, 34, 34, 38, 43]
plt.title('broadcast(b) vs join(r)')
plt.xlabel('data size')
plt.ylabel('time(s)')
plt.plot(x3, y3, 'r', label='broadcast')
plt.plot(x4, y4, 'b', label='join')
plt.xticks(x3, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rotation=0)

plt.legend(bbox_to_anchor=[0.3, 1])
plt.show()