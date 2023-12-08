import subprocess
import math

def optimal_learning_rate():
    largest_rate = 0
    inner_learning_rate = 0.4
    base = "0."
    math.floor(math.log(inner_learning_rate, 10))
    while True:
        print(f"inner_learning_rate: {inner_learning_rate}")

        # a = subprocess.run(["python", "print('Y')"], capture_output=True)
        a = subprocess.run(["python", "tools/maml_demo.py", "--inner_lr", f"{inner_learning_rate}"], capture_output=True)

        if a.stderr and len(a.stderr) >12 and a.stderr[-12:] == b'ValueError\r\n':
            inner_learning_rate /= 2
        else:
            if inner_learning_rate > largest_rate:
                largest_rate = inner_learning_rate
                with open("lr.txt", "w") as f:
                    f.write(str(largest_rate))
            elif inner_learning_rate == largest_rate:
                break
            increase = base + "0" * abs(math.floor(math.log(inner_learning_rate, 10)) + 1) + str(1)
            inner_learning_rate += float(increase)

def optimal_batch():
        
        for meta_batch_size in [16,32,64]:
            a = subprocess.run(["python", "tools/maml_demo.py", "--meta_batch_size", f"{meta_batch_size}"])

optimal_batch()