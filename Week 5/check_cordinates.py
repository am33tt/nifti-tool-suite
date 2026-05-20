import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

img = nib.load(r"D:\CT_output\MS_Probe1\MS_Probe4.nii")
data = img.dataobj

print("Volume shape (x, y, z):", img.shape)

# Z profile — finds cube boundaries
cx, cy = img.shape[0]//2, img.shape[1]//2
print("Sampling Z profile at center x,y:", cx, cy)
z_col = np.array(data[cx-100:cx+100, cy-100:cy+100, :], dtype=np.float32).mean(axis=(0,1))

plt.figure(figsize=(14,4))
plt.plot(z_col)
plt.xlabel("Z slice index")
plt.ylabel("Mean intensity")
plt.title("Z profile — cube interfaces appear as dips")
plt.grid(True)
plt.savefig("z_profile.png", dpi=150)
plt.close()

# X profile — finds left/right extent of cube
mid_z = img.shape[2]//2
x_col = np.array(data[:, cy-100:cy+100, mid_z-50:mid_z+50], dtype=np.float32).mean(axis=(1,2))

plt.figure(figsize=(14,4))
plt.plot(x_col)
plt.xlabel("X slice index")
plt.ylabel("Mean intensity")
plt.title("X profile — cube left/right edges")
plt.grid(True)
plt.savefig("x_profile.png", dpi=150)
plt.close()

# Y profile — finds front/back extent
y_col = np.array(data[cx-100:cx+100, :, mid_z-50:mid_z+50], dtype=np.float32).mean(axis=(0,2))

plt.figure(figsize=(14,4))
plt.plot(y_col)
plt.xlabel("Y slice index")
plt.ylabel("Mean intensity")
plt.title("Y profile — cube front/back edges")
plt.grid(True)
plt.savefig("y_profile.png", dpi=150)
plt.close()

print("Done — check z_profile.png, x_profile.png, y_profile.png")