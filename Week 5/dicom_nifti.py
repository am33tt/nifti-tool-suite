import nibabel as nib
import numpy as np

OUTPUT_PATH = r"D:\CT_output\MS_Probe1\MS_Probe4.nii"

cols, rows, n_slices = 3052, 3052, 5868
dtype = np.int16
HEADER_SIZE = 352

hdr = nib.Nifti1Header.from_fileobj(open(OUTPUT_PATH, 'rb'))
print("Current shape:", hdr.get_data_shape())

hdr.set_data_shape((cols, rows, n_slices))
hdr.set_data_dtype(dtype)
hdr['vox_offset'] = HEADER_SIZE

# Serialize via the underlying numpy structured array
hdr_bytes = hdr.structarr.tobytes()  # this always works
print(f"Header bytes: {len(hdr_bytes)}")  # should be 348

pad = b'\x00' * (HEADER_SIZE - len(hdr_bytes))

with open(OUTPUT_PATH, 'r+b') as f:
    f.seek(0)
    f.write(hdr_bytes + pad)

print("Header patched.")

# Verify
img = nib.load(OUTPUT_PATH)
print("Shape:", img.shape)
print("Dtype:", img.get_data_dtype())
print("Slice 0 peek:", img.dataobj[:, :, 0].min(), img.dataobj[:, :, 0].max())