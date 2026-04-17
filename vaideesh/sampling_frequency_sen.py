from pyteensy import TeensyPort
import time

enc = TeensyPort()
enc.start()
time.sleep(1.0)  # let serial settle

prev1, prev2 = None, None
changes = 0
t0 = time.time()
duration = 5.0

while time.time() - t0 < duration:
    e1, e2 = enc.enc1, enc.enc2
    if e1 != prev1 or e2 != prev2:
        print(f"  t={time.time()-t0:.4f}s  enc1={e1}  enc2={e2}")
        prev1, prev2 = e1, e2
        changes += 1

print(f"\nUnique updates in {duration}s : {changes}")
print(f"Effective serial rate        : {changes/duration:.1f} Hz")