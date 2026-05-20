import pydicom
from pathlib import Path

dicom_dir = Path(r"G:\multiscan_dcm")
files = sorted(dicom_dir.rglob("*.dcm"))

def get_z(f):
    ds = pydicom.dcmread(f, stop_before_pixels=True)
    return float(ds.ImagePositionPatient[2])

files_sorted = sorted(files, key=get_z)

# Print z position of every 250th slice
print("Slice index | Z position")
for i in range(0, len(files_sorted), 250):
    ds = pydicom.dcmread(files_sorted[i], stop_before_pixels=True)
    z = float(ds.ImagePositionPatient[2])
    print(f"  {i:5d}     |  {z:.4f} mm")