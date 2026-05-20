# import pydicom
# from pathlib import Path

# dicom_dir = Path(r"G:\multiscan_dcm")
# files = sorted(dicom_dir.rglob("*.dcm"))

# # Check first file for available sorting tags
# ds = pydicom.dcmread(files[0], stop_before_pixels=True)

# tags_to_check = [
#     "InstanceNumber",
#     "SliceLocation",
#     "ImagePositionPatient",
#     "AcquisitionNumber",
#     "SliceThickness",
#     "ZLocation",
# ]

# print("Available sorting tags:")
# for tag in tags_to_check:
#     val = ds.get(tag, "NOT FOUND")
#     print(f"  {tag}: {val}")

# # # # ======================================================
import nibabel as nib
import numpy as np
import pydicom, glob

PATH = r'D:\CT_output\MS_Probe1\MS_Probe4.nii'
HEADER_SIZE = 352
cols, rows, n_slices = 3052, 3052, 5868
dtype = np.int16

# Rebuild affine
files = glob.glob(r'F:\CT 012-25 (Christmann)\Multiscan Würfel-2\multiscan_dcm\*.dcm')
slices = []
for f in files:
    try:
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        z = float(ds.ImagePositionPatient[2])
        slices.append((z, f))
    except:
        pass
slices.sort(key=lambda x: x[0])

ds0 = pydicom.dcmread(slices[0][1])
ipp = np.array(ds0.ImagePositionPatient, dtype=float)
iop = np.array(ds0.ImageOrientationPatient, dtype=float)
ps  = np.array(ds0.PixelSpacing, dtype=float)
z_spacing = abs(slices[1][0] - slices[0][0])

row_cos = iop[:3]
col_cos = iop[3:]
normal  = np.cross(row_cos, col_cos)

affine = np.eye(4)
affine[:3, 0] = row_cos * ps[1]
affine[:3, 1] = col_cos * ps[0]
affine[:3, 2] = normal  * z_spacing
affine[:3, 3] = ipp

# Build and write correct header
header = nib.Nifti1Header()
header.set_data_shape((cols, rows, n_slices))
header.set_data_dtype(dtype)
header.set_qform(affine, code=1)
header.set_sform(affine, code=1)
header['vox_offset'] = HEADER_SIZE

header_bytes = header.structarr.tobytes()
assert len(header_bytes) == 348, f"Header size mismatch: {len(header_bytes)}"

with open(PATH, 'r+b') as f:
    f.seek(0)
    f.write(header_bytes)

print("Header patched!")

# Verify
img = nib.load(PATH)
print("Shape:", img.header.get_data_shape())
print("Dtype:", img.header.get_data_dtype())

mid = n_slices // 2
slab = np.array(img.dataobj[:, :, mid])
print(f"Mid-slice min/max: {slab.min()} / {slab.max()}")