import random
import time
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sys import setrecursionlimit

def InsertionSort(A):
    n = len(A)
    for i in range(n-1):
        if ( A[i] > A[i+1]):
            for j in range(i+1, 0, -1):
                if (A[j-1] > A[j]):
                    A[j-1], A[j] = A[j], A[j-1]
                else: break
    return A

def SelectionSort(A):
    n = len(A)
    for i in range(n-1):
        minEl = i
        for j in range(i+1, n-1):
             if (A[j] < A[minEl]): minEl = j
        A[i], A[minEl] = A[minEl], A[i]
    return A

def BubbleSort(A):
    n = len(A)
    for i in range(1, n):
        flag = True
        for j in range(0, n-1):
            if ( A[j] > A[j + 1] ):
                A[j+1], A[j] = A[j], A[j+1]
                flag = False
        if (flag): break
    return A

def MergeSort(A):
    n = len(A)
    if ( n <= 1): return A
    leftBorder = MergeSort(A[:n//2])
    rightBorder = MergeSort(A[n//2:])
    i = j = k = 0
    C = [0] * n
    leftLen, rightLen = len(leftBorder), len(rightBorder)

    while (i < leftLen and j < rightLen):
        if (leftBorder[i] <= rightBorder[j]):
            C[k] = leftBorder[i]
            i += 1
        else:
            C[k] = rightBorder[j]
            j += 1
        k += 1

    while ( i < leftLen):
        C[k] = leftBorder[i]
        i += 1
        k += 1
    
    while (j < rightLen):
        C[k] = rightBorder[j]
        j += 1
        k += 1
    
    for x in range(len(A)):
        A[x] = C[x]

    return A

def ShellSort(A, gapvar):
    n = len(A)
    gaps = []
    if (gapvar == 1):
        gap = n
        while (gap//2 > 0):
            gaps.append(gap//2)
            gap = gap // 2
    elif (gapvar == 2):
        gap = 1
        while (2**gap - 1 <= n):
            gaps.append(2**gap - 1)
            gap = gap + 1
        gaps = gaps[::-1]
    else:
        gap = 0
        i = 0
        while (2**gap * 3**i <= n//2):
            while (2**gap * 3**i <= n//2):
                gaps.append(2**gap * 3**i)
                i = i + 1 
            i = 0
            gap = gap + 1
        gaps = sorted(gaps)
        gaps = gaps[::-1]
    for x in gaps:
        for i in range (x,n,1):
            j = i
            while (j - x >= 0 and A[j-x] > A[j]):
                A[j-x],A[j] = A[j],A[j-x]
                j = j - x
    return A

def QuickSort(A):
    n = len(A)
    if (n <= 1):
        return A
    else:
        pivot = A[n//2]
        left = []
        right = []
        middle = []
        for x in A:
            if ( x < pivot ):
                left.append(x)
            elif x > pivot:
                right.append(x)
            else:
                middle.append(x)
        return QuickSort(left) + middle + QuickSort(right)



def heapify(A, n, i):
    large = i
    left = 2*i + 1
    right = 2*i + 2
    if left < n and A[i] < A[left]:
        large = i
    if right < n and A[large] < A[right]:
        large = right
    if large != i:
        A[i], A[large] = A[large], A[i]
        heapify(A, n, large)

def HeapSort(A):
	n = len(A)
	for i in range(n // 2 - 1, -1, -1):
		heapify(A, n, i)
	for i in range(n - 1, 0, -1):
		A[i], A[0] = A[0], A[i]
		heapify(A, i, 0)



def printArray(A,val,value,step):
    n = 0
    if val == 1:
        print("SelectSort Array")
        if value == 1:
            print("Random array:")
        elif value == 2:
            print("Sorted array:")
        elif value == 3:
            print("Reverse sorted array:")
        else:
            print("Sorted(90/10) array:")
        for x in A:
            print("Time(n =",n,"):",x)
            n += step
    elif val == 2:
        print("InsertSort Array")
        if value == 1:
            print("Random array:")
        elif value == 2:
            print("Sorted array:")
        elif value == 3:
            print("Reverse sorted array:")
        else:
            print("Sorted(90/10) array:")
        for x in A:
            print("Time(n =",n,"):",x)
            n += step
    elif val == 3:
        print("BubbleSort Array")
        if value == 1:
            print("Random array:")
        elif value == 2:
            print("Sorted array:")
        elif value == 3:
            print("Reverse sorted array:")
        else:
            print("Sorted(90/10) array:")
        for x in A:
            print("Time(n =",n,"):",x)
            n += step
    elif val == 4:
        print("MergeSort Array")
        if value == 1:
            print("Random array:")
        elif value == 2:
            print("Sorted array:")
        elif value == 3:
            print("Reverse sorted array:")
        else:
            print("Sorted(90/10) array:")
        for x in A:
            print("Time(n =",n,"):",x)
            n += step
    elif val == 5:
        print("ShellSort (Shell) Array")
        if value == 1:
            print("Random array:")
        elif value == 2:
            print("Sorted array:")
        elif value == 3:
            print("Reverse sorted array:")
        else:
            print("Sorted(90/10) array:")
        for x in A:
            print("Time(n =",n,"):",x)
            n += step
    elif val == 6:
        print("ShellSort (Hibard) Array")
        if value == 1:
            print("Random array:")
        elif value == 2:
            print("Sorted array:")
        elif value == 3:
            print("Reverse sorted array:")
        else:
            print("Sorted(90/10) array:")
        for x in A:
            print("Time(n =",n,"):",x)
            n += step
    elif val == 7:
        print("ShellSort (Pratta) Array")
        if value == 1:
            print("Random array:")
        elif value == 2:
            print("Sorted array:")
        elif value == 3:
            print("Reverse sorted array:")
        else:
            print("Sorted(90/10) array:")
        for x in A:
            print("Time(n =",n,"):",x)
            n += step
    elif val == 8:
        print("QuickSort Array")
        if value == 1:
            print("Random array:")
        elif value == 2:
            print("Sorted array:")
        elif value == 3:
            print("Reverse sorted array:")
        else:
            print("Sorted(90/10) array:")
        for x in A:
            print("Time(n =",n,"):",x)
            n += step
    else:
        print("HeapSort Array")
        if value == 1:
            print("Random array:")
        elif value == 2:
            print("Sorted array:")
        elif value == 3:
            print("Reverse sorted array:")
        else:
            print("Sorted(90/10) array:")
        for x in A:
            print("Time(n =",n,"):",x)
            n += step

def Generator(val,value,size,step):
    times = []
    if value == 1:
        if val == 1:
            for n in range (0,size+1,step):
                m = [random.randint(1,1000) for _ in range(n)]
                StartTime = time.time()
                SelectionSort(m)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
        if val == 2:
            for n in range (0,size+1,step):
                m = [random.randint(1,1000) for _ in range(n)]
                StartTime = time.time()
                InsertionSort(m)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
        if val == 3:
            for n in range (0,size+1,step):
                m = [random.randint(1,1000) for _ in range(n)]
                StartTime = time.time()
                BubbleSort(m)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
        if val == 4:
            for n in range (0,size+1,step):
                m = [random.randint(1,1000) for _ in range(n)]
                StartTime = time.time()
                MergeSort(m)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
        if val == 5:
            for n in range (0,size+1,step):
                m = [random.randint(1,1000) for _ in range(n)]
                StartTime = time.time()
                ShellSort(m,1)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
        if val == 6:
            for n in range (0,size+1,step):
                m = [random.randint(1,1000) for _ in range(n)]
                StartTime = time.time()
                ShellSort(m,2)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
        if val == 7:
            for n in range (0,size+1,step):
                m = [random.randint(1,1000) for _ in range(n)]
                StartTime = time.time()
                ShellSort(m,3)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
        if val == 8:
            for n in range (0,size+1,step):
                m = [random.randint(1,1000) for _ in range(n)]
                StartTime = time.time()
                QuickSort(m)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
        if val == 9:
            for n in range (0,size+1,step):
                m = [random.randint(1,1000) for _ in range(n)]
                StartTime = time.time()
                HeapSort(m)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
    elif value == 2:
        if val == 1:
            for n in range (0,size+1,step):
                m = sorted([random.randint(1,1000) for _ in range(n)])
                StartTime = time.time()
                SelectionSort(m)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
        if val == 2:
            for n in range (0,size+1,step):
                m = sorted([random.randint(1,1000) for _ in range(n)])
                StartTime = time.time()
                InsertionSort(m)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
        if val == 3:
            for n in range (0,size+1,step):
                m = sorted([random.randint(1,1000) for _ in range(n)])
                StartTime = time.time()
                BubbleSort(m)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
        if val == 4:
            for n in range (0,size+1,step):
                m = sorted([random.randint(1,1000) for _ in range(n)])
                StartTime = time.time()
                MergeSort(m)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
        if val == 5:
            for n in range (0,size+1,step):
                m = sorted([random.randint(1,1000) for _ in range(n)])
                StartTime = time.time()
                ShellSort(m,1)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
        if val == 6:
            for n in range (0,size+1,step):
                m = sorted([random.randint(1,1000) for _ in range(n)])
                StartTime = time.time()
                ShellSort(m,2)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
        if val == 7:
            for n in range (0,size+1,step):
                m = sorted([random.randint(1,1000) for _ in range(n)])
                StartTime = time.time()
                ShellSort(m,3)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
        if val == 8:
            for n in range (0,size+1,step):
                m = sorted([random.randint(1,1000) for _ in range(n)])
                StartTime = time.time()
                QuickSort(m)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
        if val == 9:
            for n in range (0,size+1,step):
                m = sorted([random.randint(1,1000) for _ in range(n)])
                StartTime = time.time()
                HeapSort(m)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
    elif value == 3:
        if val == 1:
            for n in range (0,size+1,step):
                m = sorted([random.randint(1,1000) for _ in range(n)])
                m = m[::-1]
                StartTime = time.time()
                SelectionSort(m)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
        if val == 2:
            for n in range (0,size+1,step):
                m = sorted([random.randint(1,1000) for _ in range(n)])
                m = m[::-1]
                StartTime = time.time()
                InsertionSort(m)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
        if val == 3:
            for n in range (0,size+1,step):
                m = sorted([random.randint(1,1000) for _ in range(n)])
                m = m[::-1]
                StartTime = time.time()
                BubbleSort(m)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
        if val == 4:
            for n in range (0,size+1,step):
                m = sorted([random.randint(1,1000) for _ in range(n)])
                m = m[::-1]
                StartTime = time.time()
                MergeSort(m)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
        if val == 5:
            for n in range (0,size+1,step):
                m = sorted([random.randint(1,1000) for _ in range(n)])
                m = m[::-1]
                StartTime = time.time()
                ShellSort(m,1)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
        if val == 6:
            for n in range (0,size+1,step):
                m = sorted([random.randint(1,1000) for _ in range(n)])
                m = m[::-1]
                StartTime = time.time()
                ShellSort(m,2)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
        if val == 7:
            for n in range (0,size+1,step):
                m = sorted([random.randint(1,1000) for _ in range(n)])
                m = m[::-1]
                StartTime = time.time()
                ShellSort(m,3)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
        if val == 8:
            for n in range (0,size+1,step):
                m = sorted([random.randint(1,1000) for _ in range(n)])
                m = m[::-1]
                StartTime = time.time()
                QuickSort(m)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
        if val == 9:
            for n in range (0,size+1,step):
                m = sorted([random.randint(1,1000) for _ in range(n)])
                m = m[::-1]
                StartTime = time.time()
                HeapSort(m)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
    elif value == 4:
        if val == 1:
            for n in range (0,size+1,step):
                m = sorted([random.randint(1,1000) for _ in range(n)])
                Fp = m[:int(len(m)*0.9)]
                Sp = m[int(len(m)*0.9):]
                random.shuffle(Sp)
                m = np.concatenate((Fp,Sp))
                StartTime = time.time()
                SelectionSort(m)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
        if val == 2:
            for n in range (0,size+1,step):
                m = sorted([random.randint(1,1000) for _ in range(n)])
                Fp = m[:int(len(m)*0.9)]
                Sp = m[int(len(m)*0.9):]
                random.shuffle(Sp)
                m = np.concatenate((Fp,Sp))
                StartTime = time.time()
                InsertionSort(m)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
        if val == 3:
            for n in range (0,size+1,step):
                m = sorted([random.randint(1,1000) for _ in range(n)])
                Fp = m[:int(len(m)*0.9)]
                Sp = m[int(len(m)*0.9):]
                random.shuffle(Sp)
                m = np.concatenate((Fp,Sp))
                StartTime = time.time()
                BubbleSort(m)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
        if val == 4:
            for n in range (0,size+1,step):
                m = sorted([random.randint(1,1000) for _ in range(n)])
                Fp = m[:int(len(m)*0.9)]
                Sp = m[int(len(m)*0.9):]
                random.shuffle(Sp)
                m = np.concatenate((Fp,Sp))
                StartTime = time.time()
                MergeSort(m)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
        if val == 5:
            for n in range (0,size+1,step):
                m = sorted([random.randint(1,1000) for _ in range(n)])
                Fp = m[:int(len(m)*0.9)]
                Sp = m[int(len(m)*0.9):]
                random.shuffle(Sp)
                m = np.concatenate((Fp,Sp))
                StartTime = time.time()
                ShellSort(m,1)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
        if val == 6:
            for n in range (0,size+1,step):
                m = sorted([random.randint(1,1000) for _ in range(n)])
                Fp = m[:int(len(m)*0.9)]
                Sp = m[int(len(m)*0.9):]
                random.shuffle(Sp)
                m = np.concatenate((Fp,Sp))
                StartTime = time.time()
                ShellSort(m,2)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
        if val == 7:
            for n in range (0,size+1,step):
                m = sorted([random.randint(1,1000) for _ in range(n)])
                Fp = m[:int(len(m)*0.9)]
                Sp = m[int(len(m)*0.9):]
                random.shuffle(Sp)
                m = np.concatenate((Fp,Sp))
                StartTime = time.time()
                ShellSort(m,3)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
        if val == 8:
            for n in range (0,size+1,step):
                m = sorted([random.randint(1,1000) for _ in range(n)])
                Fp = m[:int(len(m)*0.9)]
                Sp = m[int(len(m)*0.9):]
                random.shuffle(Sp)
                m = np.concatenate((Fp,Sp))
                StartTime = time.time()
                QuickSort(m)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
        if val == 9:
            for n in range (0,size+1,step):
                m = sorted([random.randint(1,1000) for _ in range(n)])
                Fp = m[:int(len(m)*0.9)]
                Sp = m[int(len(m)*0.9):]
                random.shuffle(Sp)
                m = np.concatenate((Fp,Sp))
                StartTime = time.time()
                HeapSort(m)
                EndTime = time.time()
                times.append(EndTime - StartTime)
            printArray(times,val,value,step)
    return times

def DrawGraph(val,value):
    if (val==1 or val==2 or val==3):
        max_size = 10000
        step = 500
    else:
        max_size = 100000
        step = 5000
    sizes = list(range(0, max_size + step, step))
    times = Generator(val,value,max_size,step)
    times = np.array(times)
    sizes = np.array(sizes)
    poly = PolynomialFeatures(degree=2)
    N_b_poly = poly.fit_transform(sizes.reshape(-1, 1))
    model = LinearRegression()
    model.fit(N_b_poly, times)
    y_predict = model.predict(N_b_poly)
    coeff = model.coef_
    intercept = model.intercept_
    print("Уравнение полинома: y = {:.4e} * n^2 + {:.4e} * n + {:.4e} ".format(coeff[2], coeff[1], intercept))
    plt.figure(figsize=(12, 6))
    plt.scatter(sizes, times, marker='o', linestyle='-', color='y', label='Время выполнения')
    plt.plot(sizes, y_predict, color='y', label='Регрессионная кривая')
    if val == 1:
        if (value == 1):
            plt.title('Тестирование SelectSort на случайном массиве с регрессией')
        elif (value == 2):
            plt.title('Тестирование SelectSort на отсортированном массиве с регрессией')
        elif (value == 3):
            plt.title('Тестирование SelectSort на обратно отсортированном массиве с регрессией')
        else:
            plt.title('Тестирование SelectSort на массиве отсортированном в соотношении 90/10 с регрессией')
    elif val == 2:
        if (value == 1):
            plt.title('Тестирование InsertSort на случайном массиве с регрессией')
        elif (value == 2):
            plt.title('Тестирование InsertSort на отсортированном массиве с регрессией')
        elif (value == 3):
            plt.title('Тестирование InsertSort на обратно отсортированном массиве с регрессией')
        else:
            plt.title('Тестирование InsertSort на массиве отсортированном в соотношении 90/10 с регрессией')
    elif val == 3:
        if (value == 1):
            plt.title('Тестирование BubbleSort на случайном массиве с регрессией')
        elif (value == 2):
            plt.title('Тестирование BubbleSort на отсортированном массиве с регрессией')
        elif (value == 3):
            plt.title('Тестирование BubbleSort на обратно отсортированном массиве с регрессией')
        else:
            plt.title('Тестирование BubbleSort на массиве отсортированном в соотношении 90/10 с регрессией')
    elif val == 4:
        if (value == 1):
            plt.title('Тестирование MergeSort на случайном массиве с регрессией')
        elif (value == 2):
            plt.title('Тестирование MergeSort на отсортированном массиве с регрессией')
        elif (value == 3):
            plt.title('Тестирование MergeSort на обратно отсортированном массиве с регрессией')
        else:
            plt.title('Тестирование MergeSort на массиве отсортированном в соотношении 90/10 с регрессией')
    elif val == 5:
        if (value == 1):
            plt.title('Тестирование ShellSort (Shell) на случайном массиве с регрессией')
        elif (value == 2):
            plt.title('Тестирование ShellSort (Shell) на отсортированном массиве с регрессией')
        elif (value == 3):
            plt.title('Тестирование ShellSort (Shell) на обратно отсортированном массиве с регрессией')
        else:
            plt.title('Тестирование ShellSort (Shell) на массиве отсортированном в соотношении 90/10 с регрессией')
    elif val == 6:
        if (value == 1):
            plt.title('Тестирование ShellSort (Hibard) на случайном массиве с регрессией')
        elif (value == 2):
            plt.title('Тестирование ShellSort (Hibard) на отсортированном массиве с регрессией')
        elif (value == 3):
            plt.title('Тестирование ShellSort (Hibard) на обратно отсортированном массиве с регрессией')
        else:
            plt.title('Тестирование ShellSort (Hibard) на массиве отсортированном в соотношении 90/10 с регрессией')
    elif val == 7:
        if (value == 1):
            plt.title('Тестирование ShellSort (Pratta) на случайном массиве с регрессией')
        elif (value == 2):
            plt.title('Тестирование ShellSort (Pratta) на отсортированном массиве с регрессией')
        elif (value == 3):
            plt.title('Тестирование ShellSort (Pratta) на обратно отсортированном массиве с регрессией')
        else:
            plt.title('Тестирование ShellSort (Pratta) на массиве отсортированном в соотношении 90/10 с регрессией')
    elif val == 8:
        if (value == 1):
            plt.title('Тестирование QuickSort на случайном массиве с регрессией')
        elif (value == 2):
            plt.title('Тестирование QuickSort на отсортированном массиве с регрессией')
        elif (value == 3):
            plt.title('Тестирование QuickSort на обратно отсортированном массиве с регрессией')
        else:
            plt.title('Тестирование QuickSort на массиве отсортированном в соотношении 90/10 с регрессией')
    else:
        if (value == 1):
            plt.title('Тестирование HeapSort на случайном массиве с регрессией')
        elif (value == 2):
            plt.title('Тестирование HeapSort на отсортированном массиве с регрессией')
        elif (value == 3):
            plt.title('Тестирование HeapSort на обратно отсортированном массиве с регрессией')
        else:
            plt.title('Тестирование HeapSort на массиве отсортированном в соотношении 90/10 с регрессией')
    plt.xlabel('Размер массива (n)')
    plt.ylabel('Время выполнения (t)')
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.grid()
    plt.legend()
    plt.show()

setrecursionlimit(10000)
random.seed(566)
print("1 - Random\n2 - Sorted\n3 - Resorted\n4 - 90/10")
y = int(input())
print("1 - SelectS\n2 - InsertS\n3 - BubbleS\n4 - MergeS\n5 - ShellS(Shell)\n6 - ShellS(Hibard)\n7 - ShellS(Pratt)\n8 - QS\n9 - HeapS")
x = int(input())
DrawGraph(x,y)
