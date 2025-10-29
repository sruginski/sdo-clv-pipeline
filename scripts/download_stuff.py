from sdo_clv_pipeline.sdo_download import *
import datetime as dt

data_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "data"))
# data_dir = os.path.abspath("/mnt/ceph/users/mpalumbo/new_sdo_data")
print(data_dir)
download_data(series="720", email="mlp95@psu.edu", outdir=data_dir, 
              start="2014/01/15", end="2014/01/15", 
              sample=24, overwrite=False, progress=True)
